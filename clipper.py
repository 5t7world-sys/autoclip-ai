"""AutoClip AI - video processing core.

All heavy lifting for AutoClip AI lives here: downloading a YouTube video,
transcribing it with Whisper, asking an LLM (and optionally GPT-4o vision)
which moments are worth clipping, cutting the clips with FFmpeg, auto-framing
them to vertical, burning styled captions and packaging everything as a ZIP.

Every public function is defensive: if an optional dependency (pycaps, mediapipe)
or an API call is unavailable, the module degrades gracefully to a working
FFmpeg-only fallback instead of crashing.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # yt-dlp
    import yt_dlp  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    yt_dlp = None

try:  # openai>=1.0
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # mediapipe face detection
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None

try:  # pycaps-ai
    from pycaps import TemplateLoader  # type: ignore
except Exception:  # pragma: no cover
    TemplateLoader = None  # type: ignore

try:  # PIL is only needed for frame scoring
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHISPER_MAX_BYTES = 24 * 1024 * 1024          # Whisper API hard limit is 25 MB
MAX_VIDEO_SECONDS = 30 * 60                    # >30 min triggers a warning in the UI
MIN_CLIP_SECONDS = 20.0
MAX_CLIP_SECONDS = 60.0
DEFAULT_CLIP_SECONDS = 35.0

# Model names, overridable from the environment (.env is loaded by app.py).
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _env_int(name: str, fallback: int) -> int:
    """Read an int from the environment, ignoring malformed values."""
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


DEFAULT_MAX_HEIGHT = _env_int("AUTOCLIP_MAX_HEIGHT", 1080)
DEFAULT_OUTPUT_DIR = os.environ.get("AUTOCLIP_OUTPUT_DIR", "output")

# Caption styles exposed in the UI -> (pycaps template, ASS style key)
CAPTION_STYLES = {
    "Viral Bold": {"pycaps": "explosive", "ass": "viral_bold"},
    "Minimal Clean": {"pycaps": "minimalist", "ass": "minimal_clean"},
    "Karaoke Highlight": {"pycaps": "word-focus", "ass": "karaoke"},
}

ASPECT_RATIOS = {
    "16:9 (YouTube)": (16, 9),
    "9:16 (Shorts/Reels)": (9, 16),
    "1:1 (Instagram)": (1, 1),
}


class ClipperError(RuntimeError):
    """User-facing processing error (already human readable)."""


@dataclass
class ClipResult:
    """One generated clip plus its metadata."""

    index: int
    path: str
    start: float
    end: float
    reason: str
    viral_score: int
    srt_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = self.duration
        return data


# ---------------------------------------------------------------------------
# Tooling helpers
# ---------------------------------------------------------------------------
def which(binary: str) -> Optional[str]:
    """Return the absolute path of an executable, or None if missing."""
    return shutil.which(binary)


def check_ffmpeg() -> tuple[bool, str]:
    """Verify ffmpeg/ffprobe availability.

    Returns (ok, message). ``message`` is a ready-to-display install hint.
    """
    ffmpeg, ffprobe = which("ffmpeg"), which("ffprobe")
    if ffmpeg and ffprobe:
        return True, "FFmpeg found"
    hint = (
        "FFmpeg is not installed (or not on PATH).\n"
        "• Windows:  winget install Gyan.FFmpeg   (then reopen the terminal)\n"
        "• macOS:    brew install ffmpeg\n"
        "• Linux:    sudo apt install ffmpeg\n"
        "• Conda:    conda install -c conda-forge ffmpeg"
    )
    return False, hint


def _run(
    cmd: list[str],
    timeout: Optional[int] = 3600,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output and raising a friendly error."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise ClipperError(f"Command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClipperError(f"Command timed out after {timeout}s: {cmd[0]}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        tail = " | ".join(err[-6:]) if err else "unknown error"
        raise ClipperError(f"{os.path.basename(cmd[0])} failed: {tail}")
    return proc


def ffprobe_duration(video_path: str) -> float:
    """Return media duration in seconds (0.0 when unknown)."""
    if not which("ffprobe"):
        return 0.0
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    try:
        out = _run(cmd, timeout=120).stdout.decode("utf-8", "ignore").strip()
        return float(out)
    except (ClipperError, ValueError):
        return 0.0


def ffprobe_size(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream, (0, 0) on failure."""
    if not which("ffprobe"):
        return 0, 0
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path,
    ]
    try:
        out = _run(cmd, timeout=120).stdout.decode("utf-8", "ignore").strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except (ClipperError, ValueError):
        return 0, 0


def ensure_dir(path: str) -> str:
    """Create ``path`` (file or directory parents) if it does not exist."""
    directory = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
YOUTUBE_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?"
    r"(youtube\.com/(watch\?|shorts/|live/|embed/|v/)|youtu\.be/)",
    re.IGNORECASE,
)
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")


def is_valid_youtube_url(url: str) -> bool:
    """Cheap syntactic check for a YouTube URL with a video id (no network call)."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not YOUTUBE_RE.match(url):
        return False

    tail = re.split(r"https?://", url, maxsplit=1)[-1]
    for marker in ("shorts/", "live/", "embed/", "v/"):
        if marker in tail:
            candidate = tail.split(marker, 1)[1].split("?")[0].split("&")[0].split("/")[0]
            if candidate.lower() in ("videoseries", "playlist"):
                return False
            return bool(VIDEO_ID_RE.match(candidate))
    if "youtu.be/" in tail:
        candidate = tail.split("youtu.be/", 1)[1].split("?")[0].split("&")[0].split("/")[0]
        return bool(VIDEO_ID_RE.match(candidate))
    if "watch" in tail or "?" in tail:
        match = re.search(r"[?&]v=([^&#\s]+)", url)
        return bool(match and VIDEO_ID_RE.match(match.group(1)))
    return False


def download_video(
    url: str,
    output_path: str,
    max_height: int = DEFAULT_MAX_HEIGHT,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Download a YouTube video to ``output_path`` as MP4."""
    if yt_dlp is None:
        raise ClipperError("yt-dlp is not installed. Run: pip install yt-dlp")
    if not is_valid_youtube_url(url):
        raise ClipperError(
            "That does not look like a YouTube URL. "
            "Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
    if not which("ffmpeg"):
        raise ClipperError("FFmpeg is required to merge video+audio. " + check_ffmpeg()[1])

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]
    template = os.path.join(out_dir, base + ".%(ext)s")

    def hook(data: dict) -> None:
        if progress_cb is None:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            done = data.get("downloaded_bytes") or 0
            pct = (done / total * 100.0) if total else 0.0
            speed = data.get("speed") or 0
            speed_mb = speed / 1_048_576 if speed else 0.0
            progress_cb(pct, f"Downloading… {pct:5.1f}%  ({speed_mb:.1f} MB/s)")
        elif status == "finished":
            progress_cb(100.0, "Download finished, merging streams…")

    opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}][acodec!=none][vcodec!=none]/"
            f"best[acodec!=none][vcodec!=none]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": False,
        "concurrent_fragment_downloads": 4,
        "retries": 3,
        "socket_timeout": 30,
        "progress_hooks": [hook],
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "geo_bypass": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise ClipperError("yt-dlp returned no video information.")
            if info.get("_type") == "playlist":
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise ClipperError("The playlist is empty.")
                info = entries[0]
            requested = ydl.prepare_filename(info)
    except ClipperError:
        raise
    except Exception as exc:
        msg = str(exc)
        if "Private video" in msg or "Sign in" in msg:
            raise ClipperError("This video is private or age-restricted. Try another URL.") from exc
        if "Video unavailable" in msg:
            raise ClipperError("Video unavailable (removed or region locked).") from exc
        raise ClipperError(f"Download failed: {msg}") from exc

    stem = requested.split("%(ext)s")[0]
    candidates = [
        requested if "%(ext)s" not in requested else "",
        stem + ".mp4",
        stem + ".mkv",
        stem + ".webm",
        output_path,
    ]
    found = next((c for c in candidates if c and os.path.isfile(c)), None)
    if found is None:
        matches = [
            os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
            if f.startswith(base + ".") and f.lower().endswith((".mp4", ".mkv", ".webm"))
        ]
        found = matches[0] if matches else None
    if found is None:
        raise ClipperError("Download finished but the output file could not be found.")

    if os.path.abspath(found) != os.path.abspath(output_path):
        if found.lower().endswith(".mp4"):
            shutil.move(found, output_path)
        else:
            _run(["ffmpeg", "-y", "-i", found, "-c", "copy", "-movflags", "+faststart", output_path])
            try:
                os.remove(found)
            except OSError:
                pass
    try:
        has_audio = bool(_run([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", output_path,
        ], timeout=120).stdout.strip())
    except ClipperError:
        has_audio = True
    if not has_audio:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise ClipperError(
            "The downloaded file has no audio track, so it cannot be transcribed. "
            "Try a different URL or video."
        )
    if progress_cb:
        progress_cb(100.0, "Download complete")
    return output_path
  
# ---------------------------------------------------------------------------
# 2. Transcription (Whisper API)
# ---------------------------------------------------------------------------
def transcribe_video(video_path: str, model_size: str = "base", api_key: str = "") -> list:
    """Extract audio and transcribe it using OpenAI Whisper API."""
    import tempfile
    if not which("ffmpeg"):
        raise ClipperError("FFmpeg is required to extract audio.")
    if not OpenAI:
        raise ClipperError("openai package is missing. Run: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ClipperError("OpenAI API key is required for transcription.")

    # 1. Audio Extract
    tmp_dir = tempfile.mkdtemp(prefix="autoclip_audio_")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-ac", "1", "-ar", "16000", "-b:a", "32k",
        "-map_metadata", "-1", audio_path,
    ]
    _run(cmd)

    # 2. Whisper API Call
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as fh:
        try:
            resp = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), fh.read()),
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        except Exception as exc:
            raise ClipperError(f"Whisper API failed: {exc}")

    # 3. Format Output
    segments = []
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    for seg in data.get("segments") or []:
        segments.append({
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "text": str(seg.get("text", "")).strip(),
        })
    return segments
