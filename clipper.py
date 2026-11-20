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

    # youtu.be/<id>?…  or  youtube.com/(shorts|live|embed|v)/<id>?…
    tail = re.split(r"https?://", url, maxsplit=1)[-1]
    for marker in ("shorts/", "live/", "embed/", "v/"):
        if marker in tail:
            candidate = tail.split(marker, 1)[1].split("?")[0].split("&")[0].split("/")[0]
            if candidate.lower() in ("videoseries", "playlist"):
                return False        # playlist embed, not a single video
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
    """Download a YouTube video to ``output_path`` as MP4.

    Uses yt-dlp with ``bestvideo[height<=1080]+bestaudio/best`` merged to MP4.
    ``progress_cb(percent, message)`` is called while downloading.
    Returns the path of the downloaded file.
    """
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
    except Exception as exc:  # yt-dlp raises many exception types
        msg = str(exc)
        if "Private video" in msg or "Sign in" in msg:
            raise ClipperError("This video is private or age-restricted. Try another URL.") from exc
        if "Video unavailable" in msg:
            raise ClipperError("Video unavailable (removed or region locked).") from exc
        raise ClipperError(f"Download failed: {msg}") from exc

    # yt-dlp may report .webm/.mkv even after merging, so resolve what exists.
    # prepare_filename() can also return the unexpanded "%(ext)s" template.
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
            # Remux exotic containers so downstream steps always see MP4.
            _run(["ffmpeg", "-y", "-i", found, "-c", "copy", "-movflags", "+faststart", output_path])
            try:
                os.remove(found)
            except OSError:
                pass
    # Rare, but some formats only ship a separate audio track.
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
# 2. Transcription
# ---------------------------------------------------------------------------
def _extract_audio(video_path: str, audio_path: str, max_bytes: int = WHISPER_MAX_BYTES) -> str:
    """Extract mono 16 kHz MP3 audio, shrinking bitrate if the file is too big."""
    if not which("ffmpeg"):
        raise ClipperError("FFmpeg is required to extract audio.")
    duration = ffprobe_duration(video_path)

    # 32 kbps mono is plenty for speech; lower it for very long videos.
    bitrate = "32k"
    if duration > 0:
        budget_bps = (max_bytes * 8) / max(duration, 1.0)
        if budget_bps < 32_000:
            bitrate = f"{max(8, int(budget_bps / 1000) - 2)}k"

    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-ac", "1", "-ar", "16000", "-b:a", bitrate,
        "-map_metadata", "-1", audio_path,
    ]
    _run(cmd)
    if os.path.getsize(audio_path) > max_bytes:
        # Last resort: compress harder (to a temp file — never overwrite the input).
        harder = audio_path + ".small.mp3"
        _run([
            "ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
            "-b:a", "12k", harder,
        ])
        shutil.move(harder, audio_path)
    return audio_path


def _split_audio(audio_path: str, chunk_seconds: int = 1200) -> list[tuple[str, float]]:
    """Split an oversized audio file into chunks; returns [(path, offset_seconds)].

    Chunks are re-encoded rather than stream-copied: MP3 seek points are not
    frame-exact, and every transcript timestamp is shifted by the chunk offset,
    so a few hundred milliseconds of drift would slide all captions.
    """
    total = ffprobe_duration(audio_path)
    if total <= chunk_seconds:
        return [(audio_path, 0.0)]
    out_dir = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    chunks: list[tuple[str, float]] = []
    start = 0.0
    i = 0
    while start < total:
        chunk = os.path.join(out_dir, f"{stem}_part{i:03d}.mp3")
        _run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(chunk_seconds),
            "-i", audio_path, "-ac", "1", "-ar", "16000", "-b:a", "32k", chunk,
        ])
        if os.path.getsize(chunk) > 0:
            chunks.append((chunk, start))
        start += chunk_seconds
        i += 1
    return chunks


def _whisper_response(audio_path: str, model_size: str, api_key: str) -> dict:
    """Call the Whisper endpoint once and return the raw JSON response."""
    if OpenAI is None:
        raise ClipperError("openai package missing. Run: pip install openai")
    client = OpenAI(api_key=api_key, timeout=900.0, max_retries=2)
    with open(audio_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), fh.read()),
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)


def transcribe_video(
    video_path: str,
    model_size: str = "base",
    api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[dict]:
    """Transcribe ``video_path`` and return word-level segments.

    The hosted Whisper endpoint exposes one model, so ``model_size`` is used as an
    accuracy / chunking hint: ``"small"`` (the "high accuracy" toggle) transcribes
    in shorter audio chunks so long, quiet passages keep their context, while
    ``"base"`` transcribes in large chunks for speed. Both request word-level
    timestamps.

    Returns:
        [{"start": float, "end": float, "text": str,
          "words": [{"start": float, "end": float, "word": str}, ...]}, ...]
    """
    if not os.path.isfile(video_path):
        raise ClipperError(f"Video not found: {video_path}")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ClipperError("OpenAI API key is required for transcription.")

    model_size = (model_size or "base").lower()
    chunk_seconds = 600 if model_size == "small" else 1200
    tmp_dir = tempfile.mkdtemp(prefix="autoclip_audio_")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    try:
        if progress_cb:
            progress_cb(5, "Extracting audio…")
        _extract_audio(video_path, audio_path)

        if progress_cb:
            progress_cb(20, f"Sending audio to Whisper ({model_size})…")
        needs_split = os.path.getsize(audio_path) > WHISPER_MAX_BYTES * 0.95
        chunks = _split_audio(audio_path, chunk_seconds) if needs_split else [(audio_path, 0.0)]

        segments: list[dict] = []
        for i, (chunk, offset) in enumerate(chunks):
            if progress_cb:
                pct = 20 + (70 * i / max(len(chunks), 1))
                progress_cb(pct, f"Transcribing chunk {i + 1}/{len(chunks)}…")
            data = _whisper_response(chunk, model_size, api_key)
            lang = data.get("language")
            for seg in data.get("segments") or []:
                words = [
                    {
                        "start": float(offset + (w.get("start") or 0.0)),
                        "end": float(offset + (w.get("end") or w.get("start") or 0.0)),
                        "word": str(w.get("word", "")).strip(),
                    }
                    for w in (seg.get("words") or [])
                    if str(w.get("word", "")).strip()
                ]
                text = str(seg.get("text", "")).strip()
                if not text and not words:
                    continue
                segments.append({
                    "start": float(offset + (seg.get("start") or 0.0)),
                    "end": float(offset + (seg.get("end") or seg.get("start") or 0.0)),
                    "text": text,
                    "words": words,
                    "language": lang,
                })
            # Whisper sometimes returns a flat "words" list without segments.
            if not data.get("segments") and data.get("words"):
                for w in data["words"]:
                    segments.append({
                        "start": float(offset + (w.get("start") or 0.0)),
                        "end": float(offset + (w.get("end") or 0.0)),
                        "text": str(w.get("word", "")).strip(),
                        "words": [{
                            "start": float(offset + (w.get("start") or 0.0)),
                            "end": float(offset + (w.get("end") or 0.0)),
                            "word": str(w.get("word", "")).strip(),
                        }],
                        "language": data.get("language"),
                    })

        if progress_cb:
            progress_cb(95, "Cleaning transcript…")
        segments = _clean_segments(segments)
        if not segments:
            raise ClipperError(
                "Whisper returned an empty transcript. "
                "The video may have no speech (music only) or be too noisy."
            )
        if progress_cb:
            progress_cb(100, f"Transcribed {len(segments)} segments")
        return segments
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _clean_segments(segments: Iterable[dict]) -> list[dict]:
    """Sort, drop zero-length entries and normalise word lists."""
    out: list[dict] = []
    for seg in sorted(segments, key=lambda s: s.get("start", 0.0)):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end <= start:
            end = start + 0.4
        words = [
            w for w in (seg.get("words") or [])
            if w.get("word") and float(w.get("end", 0)) >= float(w.get("start", 0))
        ]
        text = (seg.get("text") or " ".join(w["word"] for w in words)).strip()
        if not text:
            continue
        out.append({**seg, "start": start, "end": end, "text": text, "words": words})
    return out


def transcript_text(segments: list[dict], max_chars: int = 60_000) -> str:
    """Render segments as ``[mm:ss] text`` lines for LLM prompts."""
    lines = []
    size = 0
    for seg in segments:
        line = f"[{_fmt_ts(seg['start'])}] {seg['text']}"
        size += len(line) + 1
        if size > max_chars:
            lines.append("[… transcript truncated …]")
            break
        lines.append(line)
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ---------------------------------------------------------------------------
# 3. Highlight detection
# ---------------------------------------------------------------------------
def _client(api_key: str):
    if OpenAI is None:
        raise ClipperError("openai package missing. Run: pip install openai")
    if not api_key:
        raise ClipperError("OpenAI API key is required.")
    return OpenAI(api_key=api_key, timeout=180.0, max_retries=2)


def _extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating markdown fences and prose."""
    if not text:
        raise ClipperError("The model returned an empty response.")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Grab the first balanced array or object.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ClipperError("Could not parse the model's JSON answer.")


def _coerce_highlights(raw: Any) -> list[dict]:
    """Normalise arbitrary model output into highlight dicts."""
    if isinstance(raw, dict):
        for key in ("clips", "highlights", "results", "segments", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        score = item.get("viral_score", item.get("score", 70))
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 70
        out.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "reason": str(item.get("reason") or item.get("why") or "Engaging moment").strip(),
            "viral_score": max(1, min(100, score)),
        })
    return out


def _clamp_clip(highlight: dict, video_duration: float) -> Optional[dict]:
    """Keep a highlight inside the video and within 20-60 seconds."""
    start = max(0.0, float(highlight["start"]))
    end = float(highlight["end"])
    if video_duration > 0:
        end = min(end, video_duration)
    if end - start < MIN_CLIP_SECONDS:
        end = start + MIN_CLIP_SECONDS
        if video_duration > 0:
            end = min(end, video_duration)
            start = max(0.0, end - MIN_CLIP_SECONDS)
    if end - start > MAX_CLIP_SECONDS:
        end = start + MAX_CLIP_SECONDS
    if video_duration > 0:
        end = min(end, video_duration)
        if end <= start:
            return None
    if end - start < 3.0:
        return None
    return {**highlight, "start": round(start, 2), "end": round(end, 2)}


def _dedupe_highlights(highlights: list[dict]) -> list[dict]:
    """Drop overlapping candidates, keeping the higher viral score."""
    ordered = sorted(highlights, key=lambda h: h.get("viral_score", 0), reverse=True)
    kept: list[dict] = []
    for cand in ordered:
        clash = False
        for k in kept:
            overlap = min(cand["end"], k["end"]) - max(cand["start"], k["start"])
            shortest = min(cand["end"] - cand["start"], k["end"] - k["start"])
            if shortest > 0 and overlap / shortest > 0.25:
                clash = True
                break
        if not clash:
            kept.append(cand)
    return sorted(kept, key=lambda h: h["start"])


def detect_highlights_llm(
    segments: list[dict],
    num_clips: int,
    api_key: str,
    video_duration: float = 0.0,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Ask GPT-4o-mini which moments in the transcript are most clip-worthy.

    Args:
        segments: word-level transcript from :func:`transcribe_video`.
        num_clips: how many clips the user wants (1-10).
        api_key: OpenAI key.
        video_duration: total length in seconds, used to clamp timestamps.
        model: chat model name.

    Returns:
        [{"start": float, "end": float, "reason": str, "viral_score": int}, ...]
    """
    if not segments:
        raise ClipperError("Cannot detect highlights: the transcript is empty.")
    num_clips = max(1, min(10, int(num_clips)))
    client = _client(api_key)

    if video_duration <= 0:
        video_duration = max(seg["end"] for seg in segments)

    system = (
        "You are a short-form video editor who has produced viral YouTube Shorts, "
        "TikToks and Reels. You read transcripts with timestamps and pick the exact "
        "moments that would perform best as standalone clips."
    )
    user = f"""Video length: {video_duration:.0f} seconds.

Transcript with [start time] for every line:
{transcript_text(segments)}

Task: choose exactly {num_clips} non-overlapping segments that would work as
standalone short-form clips.

Rules for every clip:
- start and end are seconds from the start of the video (numbers, not timestamps)
- duration between {int(MIN_CLIP_SECONDS)} and {int(MAX_CLIP_SECONDS)} seconds
- the clip must start on a complete thought and end on a punchline/point
- prefer hooks, strong opinions, surprising facts, emotional peaks, funny beats,
  concrete numbers and quotable one-liners
- clips must not overlap each other
- viral_score is an integer 1-100 estimating short-form potential
- reason is one short sentence (max 140 characters) explaining the pick

Return ONLY a JSON array, no prose, no markdown:
[{{"start": 12.5, "end": 47.0, "reason": "…", "viral_score": 88}}]"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user + '\n\nWrap the array in {"clips": [...]}.'},
            ],
        )
        content = resp.choices[0].message.content or ""
    except Exception as exc:  # network / quota / auth
        raise ClipperError(f"Highlight detection failed: {exc}") from exc

    highlights = _coerce_highlights(_extract_json(content))
    highlights = [h for h in (_clamp_clip(h, video_duration) for h in highlights) if h]
    highlights = _dedupe_highlights(highlights)[:num_clips]
    if not highlights:
        raise ClipperError("The model did not return any usable clips. Try again or fewer clips.")
    return highlights


def _sample_frames(video_path: str, every: int = 5, max_frames: int = 60) -> list[tuple[str, float]]:
    """Extract downscaled JPEGs spread across the whole video.

    One frame is taken every ``every`` seconds, but the interval is stretched
    when needed so ``max_frames`` still covers the entire video (a 30-minute
    video gets frames from start to end, not just the first 80 seconds).

    Returns [(path, timestamp)].
    """
    if not which("ffmpeg"):
        raise ClipperError("FFmpeg is required for frame extraction.")
    duration = ffprobe_duration(video_path)
    if duration <= 0:
        raise ClipperError("Could not read the video duration for vision analysis.")

    step = max(1.0, float(every))
    if duration / step > max_frames:
        step = duration / max_frames
    count = max(1, min(max_frames, int(duration / step) + 1))

    tmp_dir = tempfile.mkdtemp(prefix="autoclip_frames_")
    pattern = os.path.join(tmp_dir, "frame_%04d.jpg")
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{step:.4f},scale=640:-2",
        "-frames:v", str(count), "-q:v", "4", pattern,
    ], timeout=900)

    files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".jpg"))
    samples: list[tuple[str, float]] = [
        (os.path.join(tmp_dir, name), min(duration, i * step))
        for i, name in enumerate(files)
    ]
    if not samples:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ClipperError("No frames could be extracted from the video.")
    return samples


def _frame_energy(frame_path: str) -> float:
    """Cheap visual-interest proxy (contrast + colourfulness), 0-100."""
    if Image is None:
        return 50.0
    try:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((160, 160))
        pixels = list(img.getdata())
        n = len(pixels)
        if not n:
            return 50.0
        lum = [(r * 299 + g * 587 + b * 114) // 1000 for r, g, b in pixels]
        mean = sum(lum) / n
        variance = sum((v - mean) ** 2 for v in lum) / n
        saturation = sum(max(p) - min(p) for p in pixels) / n
        contrast = math.sqrt(variance)          # ~0-128
        score = (contrast / 64.0) * 55.0 + (saturation / 255.0) * 45.0
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def _encode_image(frame_path: str, max_side: int = 512) -> Optional[str]:
    """Base64-encode a JPEG, downscaling it first to keep the request small."""
    import base64
    import io

    if Image is None:
        with open(frame_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    try:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def detect_highlights_vision(
    video_path: str,
    api_key: str,
    every: int = 5,
    max_frames: int = 16,
    model: str = DEFAULT_MODEL,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[dict]:
    """Score the visual interest of the video with GPT-4o vision.

    Frames are sampled across the whole video (one every ``every`` seconds,
    stretched when the video is long) and scored by the vision model in small
    batches. Returns per-timestamp scores:
        [{"start": float, "end": float, "visual_score": int, "note": str}, ...]
    Falls back to a local contrast/saturation heuristic if the API call fails, so
    AI Vision Mode never breaks the pipeline.
    """
    if progress_cb:
        progress_cb(5, f"Extracting 1 frame every {every}s…")
    samples = _sample_frames(video_path, every=every, max_frames=max_frames)
    frame_dir = os.path.dirname(samples[0][0]) if samples else ""
    try:
        scores = _score_frames(samples, api_key, every, model, progress_cb)
        # Re-anchor the per-frame scores onto the real timestamps, and make each
        # window reach the next frame so merge_highlights sees no gaps.
        stamps = [ts for _path, ts in samples]
        best: dict[int, dict] = {}
        for item in scores:
            idx = min(range(len(stamps)), key=lambda i: abs(stamps[i] - item["start"]))
            current = best.get(idx)
            if current is None or item["visual_score"] > current["visual_score"]:
                best[idx] = item
        anchored: list[dict] = []
        for idx, item in sorted(best.items()):
            item["start"] = stamps[idx]
            item["end"] = stamps[idx + 1] if idx + 1 < len(stamps) else stamps[idx] + every
            anchored.append(item)
        return anchored
    finally:
        if frame_dir:
            shutil.rmtree(frame_dir, ignore_errors=True)


def _score_frames(
    samples: list[tuple[str, float]],
    api_key: str,
    every: int,
    model: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[dict]:
    """Score pre-extracted frames (vision API with a local heuristic fallback)."""
    local_scores = [(ts, _frame_energy(path)) for path, ts in samples]

    client = _client(api_key)
    results: list[dict] = []
    batch_size = 4
    total_batches = max(1, math.ceil(len(samples) / batch_size))

    for b in range(total_batches):
        batch = samples[b * batch_size:(b + 1) * batch_size]
        if not batch:
            break
        if progress_cb:
            progress_cb(
                10 + 80 * b / total_batches,
                f"Vision analysis: batch {b + 1}/{total_batches}…",
            )
        # Only include frames that could actually be encoded, and remember the
        # index the model sees for each of them.
        sent: list[tuple[str, float]] = []
        content: list[dict] = [{
            "type": "text",
            "text": (
                "These frames are consecutive samples from one video, in order. "
                "For EACH frame rate how visually interesting it is for a short-form "
                "clip (faces, action, motion, on-screen text, emotion, appealing "
                "composition). Reply ONLY as JSON: "
                '{"frames": [{"index": 0, "score": 85, "note": "speaker close-up"}]} '
                "where score is 1-100 and note is under 12 words."
            ),
        }]
        for path, ts in batch:
            b64 = _encode_image(path)
            if not b64:
                continue
            content.append({"type": "text", "text": f"Frame {len(sent)} (t={ts:.0f}s):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })
            sent.append((path, ts))

        if not sent:
            for path, ts in batch:
                results.append({
                    "start": ts,
                    "end": ts + every,
                    "visual_score": int(round(dict(local_scores).get(ts, 50.0))),
                    "note": "local heuristic (frame unreadable)",
                })
            continue

        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            parsed = _extract_json(resp.choices[0].message.content or "")
            frames = parsed.get("frames", []) if isinstance(parsed, dict) else []
            for item in frames:
                try:
                    idx = int(item.get("index", -1))
                    score = int(round(float(item.get("score", 50))))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(sent):
                    results.append({
                        "start": sent[idx][1],
                        "end": sent[idx][1] + every,
                        "visual_score": max(1, min(100, score)),
                        "note": str(item.get("note", ""))[:80],
                    })
        except Exception:
            # Vision failed for this batch: keep the local heuristic for it.
            for path, ts in batch:
                results.append({
                    "start": ts,
                    "end": ts + every,
                    "visual_score": int(round(dict(local_scores).get(ts, 50.0))),
                    "note": "local heuristic (vision unavailable)",
                })

    if not results:
        results = [
            {"start": ts, "end": ts + every, "visual_score": int(round(score)),
             "note": "local heuristic"}
            for ts, score in local_scores
        ]

    if progress_cb:
        progress_cb(100, f"Scored {len(results)} frames")
    return sorted(results, key=lambda r: r["start"])


def merge_highlights(
    llm_results: list[dict],
    vision_results: Optional[list[dict]],
    num_clips: int = 5,
    video_duration: float = 0.0,
) -> list[dict]:
    """Combine transcript picks with visual scores into the final clip list.

    Each LLM clip gets the mean visual score of the frames it covers and a
    blended ``final_score`` (70% LLM / 30% vision). With no vision data the LLM
    score is used unchanged.
    """
    if not llm_results:
        return []
    merged: list[dict] = []

    for clip in llm_results:
        vision_scores = [
            v["visual_score"] for v in (vision_results or [])
            if v["end"] > clip["start"] and v["start"] < clip["end"]
        ]
        visual = sum(vision_scores) / len(vision_scores) if vision_scores else None
        base = float(clip.get("viral_score", 70))
        final = base * 0.7 + visual * 0.3 if visual is not None else base

        reason = clip.get("reason", "").strip()
        if visual is not None and vision_results:
            best = max(
                (v for v in vision_results if v["end"] > clip["start"] and v["start"] < clip["end"]),
                key=lambda v: v["visual_score"],
                default=None,
            )
            if best and best.get("note") and "local heuristic" not in best["note"]:
                reason = f"{reason} Visual: {best['note']}."

        merged.append({
            "start": float(clip["start"]),
            "end": float(clip["end"]),
            "reason": reason or "Engaging moment",
            "viral_score": int(round(base)),
            "visual_score": int(round(visual)) if visual is not None else None,
            "final_score": int(round(max(1, min(100, final)))),
        })

    merged.sort(key=lambda c: c["final_score"], reverse=True)
    merged = merged[:max(1, int(num_clips))]

    # Re-clamp (keeps final_score / visual_score), then present in chronological
    # order for a natural clip list.
    if video_duration > 0:
        merged = [m for m in (_clamp_clip(m, video_duration) for m in merged) if m]
    return sorted(merged, key=lambda c: c["start"])


def fallback_highlights(
    segments: list[dict],
    num_clips: int = 5,
    video_duration: float = 0.0,
) -> list[dict]:
    """Offline highlight picker used when no API key is available.

    Scores transcript windows by speech energy (word density, questions,
    numbers, emphasis words, exclamations) and returns the top windows.
    """
    if not segments:
        return []
    text_of = lambda s: s["text"].lower()
    emphasis = {
        "secret", "never", "always", "best", "worst", "biggest", "insane",
        "crazy", "shocking", "actually", "truth", "mistake", "money", "free",
        "watch", "look", "listen", "important", "remember", "because", "but",
    }

    windows: list[dict] = []
    n = len(segments)
    i = 0
    while i < n:
        start = segments[i]["start"]
        j = i
        while j < n and segments[j]["end"] - start < DEFAULT_CLIP_SECONDS:
            j += 1
        chunk = segments[i:max(j, i + 1)]
        dur = chunk[-1]["end"] - start
        if dur < MIN_CLIP_SECONDS and j < n:
            i += 1
            continue
        text = " ".join(text_of(s) for s in chunk)
        words = len(text.split())
        score = 45.0
        score += min(20.0, words / max(dur, 1.0) * 6)          # words per second
        score += 6 * sum(1 for w in emphasis if f" {w} " in f" {text} ")
        score += 8 if "?" in text else 0
        score += 6 * len(re.findall(r"\b\d{2,}\b", text))       # concrete numbers
        score += 6 if "!" in text else 0
        windows.append({
            "start": round(start, 2),
            "end": round(min(chunk[-1]["end"], start + MAX_CLIP_SECONDS), 2),
            "reason": "High speech energy and distinctive phrasing (offline heuristic).",
            "viral_score": int(max(20, min(95, score))),
        })
        i = max(i + 1, j)

    if video_duration > 0:
        windows = [w for w in (_clamp_clip(w, video_duration) for w in windows) if w]
    windows = _dedupe_highlights(windows)
    windows.sort(key=lambda w: w["viral_score"], reverse=True)
    picked = windows[:max(1, int(num_clips))]
    return sorted(picked, key=lambda w: w["start"])


# ---------------------------------------------------------------------------
# 4. Cutting and framing
# ---------------------------------------------------------------------------
_FILTER_CACHE: dict[str, bool] = {}


def has_ffmpeg_filter(name: str) -> bool:
    """Return True when the local FFmpeg build exposes a filter (e.g. subtitles)."""
    if name in _FILTER_CACHE:
        return _FILTER_CACHE[name]
    ok = False
    try:
        out = _run(["ffmpeg", "-hide_banner", "-filters"], timeout=60).stdout.decode("utf-8", "ignore")
        ok = bool(re.search(rf"^\s*[TSC.]+\s+{re.escape(name)}\s", out, re.MULTILINE))
    except ClipperError:
        ok = False
    _FILTER_CACHE[name] = ok
    return ok


def _parse_aspect(aspect_ratio: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(aspect_ratio, tuple):
        return aspect_ratio
    if aspect_ratio in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_ratio]
    match = re.search(r"(\d+)\s*:\s*(\d+)", str(aspect_ratio))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 16, 9


def _crop_geometry(w: int, h: int, ar: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    """Largest centered crop of size ``ar`` inside ``w x h``; None if already matching."""
    ar_w, ar_h = ar
    if w <= 0 or h <= 0:
        return None
    target = ar_w / ar_h
    current = w / h
    if abs(current - target) < 0.01:
        return None
    if current > target:            # too wide -> trim the sides
        cw, ch = int(round(h * target)), h
    else:                           # too tall -> trim top/bottom
        cw, ch = w, int(round(w / target))
    cw -= cw % 2
    ch -= ch % 2
    if cw <= 2 or ch <= 2:
        return None
    return cw, ch, (w - cw) // 2, (h - ch) // 2


def _target_size(ar: tuple[int, int]) -> tuple[int, int]:
    """Canonical export size for an aspect ratio."""
    ar_w, ar_h = ar
    if ar_w == ar_h:
        return 1080, 1080
    if ar_h > ar_w:                 # vertical
        return 1080, 1920
    return 1920, 1080


def cut_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    aspect_ratio: str = "16:9 (YouTube)",
    fast: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    stream_copy: bool = False,
) -> str:
    """Cut ``start``-``end`` out of ``video_path`` and crop it to ``aspect_ratio``.

    Pass ``aspect_ratio="source"`` to keep the original shape (only used by
    auto-framing). Otherwise the clip is cropped to the target ratio and scaled
    to a standard size (1080x1920 for 9:16, 1080x1080 for 1:1, up to 1920x1080
    for 16:9).

    Cuts are always frame-accurate: ``-ss`` sits before ``-i`` (fast seek) but the
    clip is re-encoded, so the output really starts at ``start``. Stream-copying
    would snap to the nearest keyframe and silently shift every caption, so it is
    only available by explicitly passing ``stream_copy=True``. ``fast=True``
    switches to a quicker x264 preset (same accuracy, slightly larger file).

    Returns the output path.
    """
    if not os.path.isfile(video_path):
        raise ClipperError(f"Source video not found: {video_path}")
    start = max(0.0, float(start))
    end = float(end)
    if end <= start:
        raise ClipperError("Invalid clip range: end must be after start.")
    duration = end - start
    ensure_dir(output_path)
    if progress_cb:
        progress_cb(5, f"Cutting {start:.1f}s → {end:.1f}s ({duration:.1f}s)…")

    src_w, src_h = ffprobe_size(video_path)
    if str(aspect_ratio).lower() in ("source", "original", "none"):
        ar = None
        crop = None
    else:
        ar = _parse_aspect(aspect_ratio)
        crop = _crop_geometry(src_w, src_h, ar)

    use_copy = (
        stream_copy
        and crop is None
        and (ar is None or ar == (16, 9))
        and not (src_h and src_h > 1080)
    )
    if use_copy:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", video_path, "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", output_path,
        ]
    else:
        filters: list[str] = []
        if crop:
            cw, ch, cx, cy = crop
            filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        if ar is None:
            if src_h and src_h > 1080:
                filters.append("scale=-2:1080")     # keep source shape, cap at 1080p
        else:
            tw, th = _target_size(ar)
            if crop:
                # Crop already matches the target ratio, so scaling cannot distort.
                filters.append(f"scale={tw}:{th}")
            elif src_w and src_h and (src_w > tw or src_h > th):
                filters.append(f"scale={tw}:{th}")  # downscale only, never upscale
        filters.append("setsar=1")
        filters.append("format=yuv420p")

        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", video_path,
            "-vf", ",".join(filters),
            "-c:v", "libx264", "-preset", "veryfast" if fast else "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
            "-movflags", "+faststart", output_path,
        ]
    _run(cmd, timeout=3600)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise ClipperError("FFmpeg produced an empty clip. The source range may be corrupt.")
    if progress_cb:
        progress_cb(100, "Clip cut")
    return output_path


def _sample_frames_for_faces(video_path: str, count: int = 12) -> list[str]:
    """Extract up to ``count`` evenly spaced JPEG frames for face scanning."""
    duration = ffprobe_duration(video_path)
    if duration <= 0:
        return []
    tmp_dir = tempfile.mkdtemp(prefix="autoclip_faces_")
    step = max(0.5, duration / count)
    out = []
    for i in range(count):
        ts = min(duration - 0.1, i * step)
        path = os.path.join(tmp_dir, f"f{i:02d}.jpg")
        try:
            _run([
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5", path,
            ], timeout=120)
        except ClipperError:
            continue
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            out.append(path)
    return out


def _face_center_mediapipe(frame_path: str) -> Optional[tuple[float, float, float]]:
    """Return (cx, cy, area) in 0-1 coordinates for the largest face, or None."""
    if mp is None:
        return None
    try:
        import numpy as np  # bundled with mediapipe

        if Image is None:
            return None
        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        ) as detector:
            result = detector.process(arr)
        if not result.detections:
            return None
        best = max(
            result.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        box = best.location_data.relative_bounding_box
        cx = min(1.0, max(0.0, box.xmin + box.width / 2))
        cy = min(1.0, max(0.0, box.ymin + box.height / 2))
        return cx, cy, max(0.0, box.width * box.height)
    except Exception:
        return None


def _face_center_heuristic(frame_path: str) -> Optional[tuple[float, float, float]]:
    """Skin-tone centroid fallback when mediapipe is unavailable."""
    if Image is None:
        return None
    try:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((96, 96))
        w, h = img.size
        pixels = img.load()
        total = 0.0
        sx = sy = 0.0
        for y in range(h):
            for x in range(w):
                r, g, b = pixels[x, y]
                if r > 95 and g > 40 and b > 20 and r > g and r > b and (max(r, g, b) - min(r, g, b)) > 18:
                    weight = 1.0
                    sx += x * weight
                    sy += y * weight
                    total += weight
        if total < w * h * 0.01:      # less than 1% skin-like pixels
            return None
        return sx / total / w, sy / total / h, min(1.0, total / (w * h))
    except Exception:
        return None


def apply_auto_framing(
    clip_path: str,
    output_path: str,
    aspect_ratio: str = "9:16 (Shorts/Reels)",
) -> str:
    """Re-crop a clip around the detected faces ("auto-framing").

    Face positions are detected with mediapipe on sampled frames (skin-tone
    centroid as fallback). The median face position determines the crop window,
    so the speaker stays centered instead of being cut by a blind center crop.
    When no face is found the clip is center-cropped to the target ratio, so the
    output always has the requested shape.

    Returns the output path (or ``clip_path`` if no reframe was needed).
    """
    if not os.path.isfile(clip_path):
        raise ClipperError(f"Clip not found: {clip_path}")
    ar = _parse_aspect(aspect_ratio)
    if ar[0] >= ar[1]:
        return clip_path          # auto-framing only makes sense for portrait/square

    w, h = ffprobe_size(clip_path)
    if w <= 0 or h <= 0:
        return clip_path

    target = ar[0] / ar[1]
    if abs(w / h - target) < 0.01:
        return clip_path          # already the right shape (e.g. after cut_clip)

    def crop_size() -> tuple[int, int]:
        if w / h > target:
            cw, ch = int(round(h * target)), h
        else:
            cw, ch = w, int(round(w / target))
        return cw - cw % 2, ch - ch % 2

    cw, ch = crop_size()
    if cw <= 2 or ch <= 2 or (cw >= w and ch >= h):
        return clip_path

    frames = _sample_frames_for_faces(clip_path, count=10)
    centers: list[tuple[float, float, float]] = []
    for frame in frames:
        center = _face_center_mediapipe(frame) or _face_center_heuristic(frame)
        if center:
            centers.append(center)
    if frames:
        shutil.rmtree(os.path.dirname(frames[0]), ignore_errors=True)

    if centers:
        # Weighted median resists a single false positive.
        centers.sort(key=lambda c: c[0])
        total_weight = sum(c[2] for c in centers) or 1.0
        acc = 0.0
        face_x = centers[len(centers) // 2][0]
        for cx, _cy, area in centers:
            acc += area
            if acc >= total_weight / 2:
                face_x = cx
                break
        face_y = sorted(c[1] for c in centers)[len(centers) // 2]
        cx = int(round(face_x * w - cw / 2))
        cy = int(round(face_y * h - ch / 2))
        # Keep the face in the upper-middle third with a gentle bias upward.
        cy = int(round(cy - ch * 0.05))
    else:
        cx = (w - cw) // 2
        cy = (h - ch) // 2

    cx = max(0, min(w - cw, cx))
    cy = max(0, min(h - ch, cy))

    ensure_dir(output_path)
    tw, th = _target_size(ar)
    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={tw}:{th},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        _run(cmd, timeout=1800)
    except ClipperError:
        return clip_path
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        return clip_path
    return output_path


# ---------------------------------------------------------------------------
# 5. Captions
# ---------------------------------------------------------------------------
def _ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_escape(text: str) -> str:
    text = text.replace("\\", "").replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", " ", text).strip()


def _words_relative(words: list[dict], offset: float) -> list[dict]:
    out = []
    for w in words or []:
        start = float(w.get("start", 0.0)) - offset
        end = float(w.get("end", start)) - offset
        start = max(0.0, start)
        end = max(start + 0.06, end)
        text = _ass_escape(str(w.get("word", "")))
        if text:
            out.append({"start": start, "end": end, "word": text})
    return out


def _words_from_segments(segments: list[dict], offset: float) -> list[dict]:
    """Word list for a clip, using Whisper words or a proportional split."""
    collected: list[dict] = []
    for seg in segments or []:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end <= offset:
            continue
        words = _words_relative(seg.get("words") or [], offset)
        if words:
            collected.extend(words)
            continue
        tokens = str(seg.get("text", "")).split()
        if not tokens:
            continue
        span = max(0.2, seg_end - seg_start)
        step = span / len(tokens)
        for i, token in enumerate(tokens):
            start = max(0.0, seg_start + i * step - offset)
            collected.append({
                "start": start,
                "end": start + step * 0.92,
                "word": _ass_escape(token),
            })
    collected.sort(key=lambda w: w["start"])
    return [w for w in collected if w["end"] > 0]


def _group_caption_words(
    words: list[dict], max_words: int, max_chars: int, max_dur: float
) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        current.append(word)
        text_len = sum(len(w["word"]) + 1 for w in current) - 1
        span = current[-1]["end"] - current[0]["start"]
        ended = word["word"].rstrip().endswith((".", "!", "?", ",", ";", ":"))
        if (len(current) >= max_words or text_len >= max_chars or span >= max_dur
                or (ended and len(current) >= 2)):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


_CAPTION_PRESETS = {
    "viral_bold": {
        "font": "Arial Black",
        "size_ratio": 0.062,
        "primary": "&H00FFFFFF",
        "highlight": "&H0000E5FF",
        "outline": 0x000000,
        "outline_w": 6,
        "shadow": 2,
        "bold": -1,
        "upper": True,
        "max_words": 3,
        "max_chars": 22,
        "max_dur": 2.0,
        "margin_ratio": 0.16,
        "karaoke": False,
        "pop": True,
    },
    "minimal_clean": {
        "font": "Arial",
        "size_ratio": 0.040,
        "primary": "&H00FFFFFF",
        "highlight": "&H00FFFFFF",
        "outline": 0x101010,
        "outline_w": 2,
        "shadow": 1,
        "bold": 0,
        "upper": False,
        "max_words": 8,
        "max_chars": 42,
        "max_dur": 4.0,
        "margin_ratio": 0.10,
        "karaoke": False,
        "pop": False,
    },
    "karaoke": {
        "font": "Arial",
        "size_ratio": 0.055,
        "primary": "&H0000D7FF",     # gold, revealed as the word is spoken
        "secondary": "&H00FFFFFF",   # white before it is spoken
        "outline": 0x000000,
        "outline_w": 5,
        "shadow": 0,
        "bold": -1,
        "upper": False,
        "max_words": 6,
        "max_chars": 30,
        "max_dur": 3.5,
        "margin_ratio": 0.14,
        "karaoke": True,
        "pop": False,
    },
}


def build_ass(
    words: list[dict],
    ass_path: str,
    style: str = "Viral Bold",
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Render word-timed captions into an Advanced SubStation Alpha file.

    ``style`` accepts a display name ("Viral Bold", "Minimal Clean",
    "Karaoke Highlight") or an internal key ("viral_bold", "minimal_clean",
    "karaoke"). ``words`` must already be relative to the clip start.
    """
    key = CAPTION_STYLES.get(style, {}).get("ass", str(style).lower().replace(" ", "_"))
    preset = _CAPTION_PRESETS.get(key, _CAPTION_PRESETS["viral_bold"])

    font_size = max(18, int(round(height * preset["size_ratio"])))
    margin_v = max(10, int(round(height * preset["margin_ratio"])))
    outline = f"&H00{preset['outline']:06X}"
    secondary = preset.get("secondary", preset["highlight"])
    back = "&H80000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {int(width)}
PlayResY: {int(height)}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{preset['font']},{font_size},{preset['primary']},{secondary},{outline},{back},{preset['bold']},0,0,0,100,100,0,0,1,{preset['outline_w']},{preset['shadow']},2,{int(width*0.06)},{int(width*0.06)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    groups = _group_caption_words(
        words, preset["max_words"], preset["max_chars"], preset["max_dur"]
    )

    lines: list[str] = []
    for group in groups:
        if not group:
            continue
        start = group[0]["start"]
        end = max(group[-1]["end"], start + 0.25)
        if preset["karaoke"]:
            parts = []
            for word in group:
                centis = max(4, int(round((word["end"] - word["start"]) * 100)))
                parts.append(f"{{\\k{centis}}}{word['word']}")
            text = " ".join(parts)
        else:
            tokens = []
            for word in group:
                token = word["word"].upper() if preset["upper"] else word["word"]
                tokens.append(token)
            if preset["pop"] and tokens:
                # Highlight the longest word — usually the punchy one.
                longest = max(range(len(tokens)), key=lambda i: len(tokens[i]))
                tokens[longest] = (
                    f"{{\\c{preset['highlight']}&}}{tokens[longest]}"
                    f"{{\\c{preset['primary']}&}}"
                )
            text = " ".join(tokens)
            if preset["pop"]:
                text = f"{{\\fad(60,40)\\fscx108\\fscy108\\t(0,110,\\fscx100\\fscy100)}}{text}"
            else:
                text = f"{{\\fad(90,90)}}{text}"
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )

    ensure_dir(ass_path)
    with open(ass_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n".join(lines) + ("\n" if lines else ""))
    return ass_path


def _write_srt_from_words(words: list[dict], srt_path: str) -> str:
    groups = _group_caption_words(words, 8, 42, 4.0)
    blocks = []
    for i, group in enumerate(groups, start=1):
        blocks.append(
            f"{i}\n{_srt_time(group[0]['start'])} --> {_srt_time(group[-1]['end'])}\n"
            + " ".join(w["word"] for w in group)
        )
    ensure_dir(srt_path)
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(blocks) + "\n")
    return srt_path


def save_srt(segments: list[dict], srt_path: str, offset: float = 0.0) -> str:
    """Write an SRT for ``segments`` shifted by ``-offset`` (clip-relative)."""
    words = _words_from_segments(segments, offset)
    if words:
        return _write_srt_from_words(words, srt_path)

    blocks = []
    index = 1
    for seg in segments or []:
        start = max(0.0, float(seg["start"]) - offset)
        end = max(start + 0.3, float(seg["end"]) - offset)
        text = _ass_escape(str(seg.get("text", "")))
        if not text:
            continue
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}")
        index += 1
    ensure_dir(srt_path)
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(blocks) + ("\n" if blocks else ""))
    return srt_path


def _burn_ass(clip_path: str, output_path: str, ass_path: str) -> str:
    """Burn an .ass file into a clip using FFmpeg's libass filter.

    FFmpeg runs with ``cwd`` set to the ASS file's directory and the filter gets
    only a bare filename, which sidesteps every path-escaping pitfall in filter
    graphs (spaces, colons, commas, brackets in the input path).
    """
    if not has_ffmpeg_filter("subtitles") and not has_ffmpeg_filter("ass"):
        raise ClipperError(
            "This FFmpeg build has no libass support (subtitles/ass filter). "
            "Install a full build: brew install ffmpeg / winget install Gyan.FFmpeg / "
            "sudo apt install ffmpeg"
        )
    filter_name = "subtitles" if has_ffmpeg_filter("subtitles") else "ass"

    work_dir = os.path.dirname(os.path.abspath(ass_path))
    local_ass = os.path.basename(ass_path)
    local_out = "_captioned_" + os.path.basename(output_path)
    produced = os.path.join(work_dir, local_out)
    if os.path.isfile(produced):
        os.remove(produced)

    cmd = [
        "ffmpeg", "-y", "-i", os.path.abspath(clip_path),
        "-vf", f"{filter_name}={local_ass}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",       # re-encode: copy can self-overwrite
        "-movflags", "+faststart",
        local_out,
    ]
    _run(cmd, timeout=3600, cwd=work_dir)

    if not os.path.isfile(produced) or os.path.getsize(produced) == 0:
        raise ClipperError("Caption burning produced no output.")
    ensure_dir(output_path)
    if os.path.abspath(produced) != os.path.abspath(output_path):
        shutil.move(produced, output_path)
    return output_path


def _add_captions_pycaps(
    clip_path: str, output_path: str, style: str, srt_path: Optional[str]
) -> str:
    """Caption a clip with pycaps-ai templates (real API, v0.2+)."""
    if TemplateLoader is None:
        raise ClipperError("pycaps-ai is not installed.")
    template = CAPTION_STYLES.get(style, {}).get("pycaps", "default")
    if os.path.exists(output_path):
        os.remove(output_path)

    loader = TemplateLoader(template).with_input_video(clip_path)
    builder = loader.load(False)
    builder.with_output_video(output_path)
    if srt_path and os.path.isfile(srt_path):
        try:
            builder.with_transcription_file(srt_path, "srt")
        except Exception:
            pass
    builder.build().run()
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise ClipperError("pycaps produced no output file.")
    return output_path


def add_captions(
    clip_path: str,
    output_path: str,
    style: str = "Viral Bold",
    words: Optional[list[dict]] = None,
    segments: Optional[list[dict]] = None,
    offset: float = 0.0,
    use_pycaps: bool = False,
    srt_path: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Burn styled, word-synced captions onto a clip.

    Primary path: an ASS file built here (karaoke ``\\k`` timing, pop-in
    animations) burned with FFmpeg's libass filter — no browser, no extra
    services. When ``use_pycaps`` is True the pycaps-ai template pipeline is
    tried first and this renderer is used as an automatic fallback.

    Args:
        clip_path: source clip.
        output_path: destination file.
        style: "Viral Bold" | "Minimal Clean" | "Karaoke Highlight".
        words: word timings in *source video* time (offset is subtracted).
        segments: full transcript; used when ``words`` is None.
        offset: clip start inside the source video.
        use_pycaps: try pycaps-ai templates first.
        srt_path: optional sidecar SRT to reuse / create.
        progress_cb: optional progress callback.

    Returns:
        Path to the captioned clip (or ``clip_path`` if captions were skipped).
    """
    if not os.path.isfile(clip_path):
        raise ClipperError(f"Clip not found: {clip_path}")
    width, height = ffprobe_size(clip_path)
    if width <= 0 or height <= 0:
        width, height = 1080, 1920

    clip_words = _words_relative(words, offset) if words else _words_from_segments(segments or [], offset)

    # Never caption past the end of the clip (the segment fallback can return the
    # whole transcript), and drop words that start after it.
    clip_duration = ffprobe_duration(clip_path)
    if clip_duration > 0 and clip_words:
        clip_words = [
            {"start": w["start"], "end": min(w["end"], clip_duration), "word": w["word"]}
            for w in clip_words
            if w["start"] < clip_duration
        ]
        clip_words = [w for w in clip_words if w["end"] > w["start"]]

    if progress_cb:
        progress_cb(10, "Building caption timeline…")

    work_dir = tempfile.mkdtemp(prefix="autoclip_caps_")
    try:
        local_srt = srt_path
        if not local_srt and clip_words:
            # Write a clip-relative SRT (times start at 0 for this clip).
            local_srt = os.path.join(work_dir, "captions.srt")
            _write_srt_from_words(clip_words, local_srt)

        if use_pycaps and clip_words:
            if progress_cb:
                progress_cb(30, "Rendering captions with pycaps-ai…")
            try:
                return _add_captions_pycaps(clip_path, output_path, style, local_srt)
            except Exception as exc:  # noqa: BLE001 - fall through to ASS
                if progress_cb:
                    progress_cb(35, f"pycaps unavailable ({type(exc).__name__}), using FFmpeg captions…")

        if not clip_words:
            if progress_cb:
                progress_cb(100, "No speech in clip — captions skipped")
            ensure_dir(output_path)
            shutil.copy2(clip_path, output_path)
            return output_path

        if progress_cb:
            progress_cb(45, f"Styling captions ({style})…")
        ass_path = os.path.join(work_dir, "captions.ass")
        build_ass(clip_words, ass_path, style=style, width=width, height=height)

        if progress_cb:
            progress_cb(60, "Burning captions…")
        _burn_ass(clip_path, output_path, ass_path)
        if progress_cb:
            progress_cb(100, "Captions burned")
        return output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def create_zip(clip_paths: list[str], zip_path: str) -> str:
    """Bundle clips (and their sibling .srt files) into a ZIP archive."""
    valid = [p for p in clip_paths if p and os.path.isfile(p)]
    if not valid:
        raise ClipperError("No clips available to zip.")
    ensure_dir(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in valid:
            zf.write(path, arcname=os.path.basename(path))
            sidecar = os.path.splitext(path)[0] + ".srt"
            if os.path.isfile(sidecar):
                zf.write(sidecar, arcname=os.path.basename(sidecar))
    return zip_path


# ---------------------------------------------------------------------------
# 6. Demo mode + end-to-end pipeline
# ---------------------------------------------------------------------------
DEMO_LINES = [
    (0.0, 3.4, "Nobody tells you this, but the first version of anything is always bad."),
    (3.4, 7.2, "I rebuilt my entire workflow in one weekend and it changed everything."),
    (7.2, 11.0, "Here is the part that actually matters: you do not need more tools."),
    (11.0, 14.6, "You need one boring habit that you repeat for ninety days straight."),
    (14.6, 18.4, "That is the secret nobody wants to hear, because it is not exciting."),
    (18.4, 22.2, "So pick the smallest action, and do it today, not on Monday."),
    (22.2, 26.0, "If you do that, in three months you will be unrecognizable."),
    (26.0, 30.0, "That is how you actually get ahead. Save this one."),
]


def generate_demo_video(output_path: str, duration: float = 30.0) -> str:
    """Render a synthetic test video (no download, no API key).

    Uses FFmpeg's built-in ``testsrc2`` pattern with a gentle hue rotation plus a
    silent AAC track, so every downstream step behaves like a real download while
    relying only on filters every FFmpeg build ships with. The "DEMO" text
    overlay is added when libfreetype is available and skipped otherwise.
    """
    if not which("ffmpeg"):
        raise ClipperError("FFmpeg is required for demo mode. " + check_ffmpeg()[1])
    ensure_dir(output_path)

    base_filters = (
        "hue=h='t*24':s=0.9,"
        "drawbox=x='(w/2)+(w/4)*sin(t*0.9)-150':y='h/2-150':w=300:h=300:"
        "color=0xFF3B7B@0.85:t=fill"
    )
    text_filter = (
        ",drawtext=text='AutoClip AI - DEMO':fontcolor=white:fontsize=54:"
        "x=(w-text_w)/2:y=h*0.14:box=1:boxcolor=black@0.45:boxborderw=18"
    )

    def build(filters: str) -> list[str]:
        return [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=30:d={duration:.2f}",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration:.2f}",
            "-vf", filters,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            "-movflags", "+faststart", output_path,
        ]

    try:
        _run(build(base_filters + text_filter), timeout=600)
    except ClipperError:
        # Some minimal builds lack drawtext (libfreetype): retry without it.
        _run(build(base_filters), timeout=600)

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise ClipperError("Demo video generation failed.")
    return output_path


def demo_transcript(duration: float = 30.0) -> list[dict]:
    """Word-level transcript matching :func:`generate_demo_video`."""
    segments: list[dict] = []
    for start, end, text in DEMO_LINES:
        tokens = text.split()
        step = (end - start) / max(len(tokens), 1)
        words = [
            {"start": round(start + i * step, 3),
             "end": round(start + (i + 1) * step * 0.94, 3),
             "word": token}
            for i, token in enumerate(tokens)
        ]
        segments.append({
            "start": start, "end": end, "text": text, "words": words, "language": "en",
        })
    return segments


def demo_highlights(num_clips: int = 3) -> list[dict]:
    """Deterministic highlight picks for demo mode."""
    pool = [
        {"start": 0.0, "end": 14.6, "reason": "Strong hook plus a contrarian claim — classic short-form opener.", "viral_score": 92},
        {"start": 11.0, "end": 26.0, "reason": "The core insight lands here: one habit, ninety days.", "viral_score": 88},
        {"start": 18.4, "end": 30.0, "reason": "Actionable payoff and a closing call to save the clip.", "viral_score": 81},
    ]
    return sorted(pool[:max(1, min(10, num_clips))], key=lambda c: c["start"])


def probe_video(url: str, timeout: int = 45) -> dict:
    """Read metadata (title, duration) without downloading the video."""
    if yt_dlp is None:
        raise ClipperError("yt-dlp is not installed. Run: pip install yt-dlp")
    if not is_valid_youtube_url(url):
        raise ClipperError(
            "That does not look like a YouTube URL. "
            "Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        raise ClipperError(f"Could not read the video info: {exc}") from exc
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        info = entries[0] if entries else info
    return {
        "title": info.get("title") or "Untitled video",
        "duration": float(info.get("duration") or 0.0),
        "uploader": info.get("uploader") or "",
        "thumbnail": info.get("thumbnail") or "",
    }


@dataclass
class PipelineConfig:
    """Everything the pipeline needs, assembled by the UI."""

    url: str = ""
    num_clips: int = 3
    aspect_ratio: str = "9:16 (Shorts/Reels)"
    caption_style: str = "Viral Bold"
    api_key: str = ""
    vision_mode: bool = False
    auto_framing: bool = True
    whisper_model: str = "base"
    demo_mode: bool = False
    use_pycaps: bool = True
    output_dir: str = DEFAULT_OUTPUT_DIR


def run_pipeline(
    config: PipelineConfig,
    step_cb: Optional[Callable[[str, str, str], None]] = None,
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> dict:
    """Run the full AutoClip AI pipeline and return the results payload.

    Steps: download → transcribe → detect highlights → (vision) → cut →
    auto-frame → captions → SRT → ZIP. Works with or without an API key: with
    no key it skips Whisper entirely, picks moments with the offline heuristic,
    and produces uncaptioned clips (there is no word timing to sync to).

    Args:
        config: :class:`PipelineConfig`.
        step_cb: ``step_cb(step_id, state, message)`` with state in
            {"running", "done", "error", "skipped"}.
        progress_cb: ``progress_cb(step_id, percent, message)``.

    Returns:
        {"clips": [ClipResult dicts], "zip_path": str, "elapsed": float,
         "video_title": str, "output_dir": str}
    """
    import time

    started = time.time()
    out_dir = os.path.abspath(config.output_dir or "output")
    os.makedirs(out_dir, exist_ok=True)

    def step(step_id: str, state: str, message: str = "") -> None:
        if step_cb:
            step_cb(step_id, state, message)

    def progress(step_id: str, pct: float, message: str = "") -> None:
        if progress_cb:
            progress_cb(step_id, max(0.0, min(100.0, float(pct))), message)

    ok, ffmpeg_msg = check_ffmpeg()
    if not ok:
        step("setup", "error", ffmpeg_msg)
        raise ClipperError(ffmpeg_msg)

    api_key = (config.api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    num_clips = max(1, min(10, int(config.num_clips)))
    source_video = os.path.join(out_dir, "_source.mp4")
    title = "Demo clip"
    ar_target = _parse_aspect(config.aspect_ratio)
    framing_enabled = bool(config.auto_framing) and ar_target[1] > ar_target[0]

    # Remove clips from previous runs so the folder and ZIP never mix results.
    for name in os.listdir(out_dir):
        if re.fullmatch(r"clip_\d+\.(mp4|srt)", name) or name == "autoclip_clips.zip":
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass

    # --- Step 1: source video -------------------------------------------------
    try:
        if config.demo_mode:
            step("download", "running", "Rendering a synthetic 30s demo video…")
            progress("download", 10, "Preparing demo footage…")
            generate_demo_video(source_video)
            segments = demo_transcript()
            progress("download", 100, "Demo video ready")
            step("download", "done", "Demo video generated (no download needed)")
            step("transcribe", "done", "Demo transcript loaded")
            title = "AutoClip AI demo (synthetic footage)"
            duration = ffprobe_duration(source_video) or 30.0
        else:
            if not config.url:
                raise ClipperError("Please paste a YouTube URL, or enable Demo Mode.")
            step("download", "running", "Downloading video with yt-dlp…")
            progress("download", 2, "Contacting YouTube…")
            download_video(
                config.url, source_video, max_height=DEFAULT_MAX_HEIGHT,
                progress_cb=lambda pct, msg: progress("download", pct, msg),
            )
            duration = ffprobe_duration(source_video)
            if duration <= 0:
                raise ClipperError("Could not read the downloaded video's duration.")
            step("download", "done", f"Downloaded {_fmt_ts(duration)} of footage")
            title = config.url

            # --- Step 2: transcription --------------------------------------
            if not api_key:
                step("transcribe", "skipped", "No API key — using the offline highlight heuristic")
                segments = []
            else:
                step("transcribe", "running", f"Transcribing with Whisper ({config.whisper_model})…")
                segments = transcribe_video(
                    source_video, config.whisper_model, api_key=api_key,
                    progress_cb=lambda pct, msg: progress("transcribe", pct, msg),
                )
                step("transcribe", "done", f"{len(segments)} segments transcribed")
    except ClipperError as exc:
        step("download", "error", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        step("download", "error", str(exc))
        raise ClipperError(f"Video preparation failed: {exc}") from exc

    # --- Step 3: highlight detection -----------------------------------------
    vision_results: list[dict] = []
    try:
        if config.demo_mode:
            step("detect", "done", "Demo highlights loaded")
            highlights = demo_highlights(num_clips)
        elif api_key and segments:
            step("detect", "running", "Asking GPT-4o-mini for the most viral moments…")
            progress("detect", 15, "Analysing transcript…")
            llm_results = detect_highlights_llm(
                segments, num_clips, api_key, video_duration=duration,
            )
            progress("detect", 50, f"{len(llm_results)} candidate clips")

            if config.vision_mode:
                step("vision", "running", "Scoring frames with GPT-4o vision…")
                try:
                    vision_results = detect_highlights_vision(
                        source_video, api_key,
                        progress_cb=lambda pct, msg: progress("vision", pct, msg),
                    )
                    step("vision", "done", f"{len(vision_results)} frames scored")
                except ClipperError as exc:
                    step("vision", "skipped", f"Vision analysis skipped: {exc}")
                    vision_results = []
            else:
                step("vision", "skipped", "AI Vision Mode is off")

            highlights = merge_highlights(llm_results, vision_results, num_clips, duration)
            progress("detect", 100, f"{len(highlights)} clips selected")
            step("detect", "done", f"{len(highlights)} clips selected")
        else:
            reason = "Demo" if not config.url else "No API key"
            step("detect", "running", f"{reason}: ranking moments with the offline heuristic…")
            highlights = fallback_highlights(segments, num_clips, duration) if segments else []
            if not highlights:
                # No transcript at all: cut evenly spaced windows.
                span = min(MAX_CLIP_SECONDS, max(MIN_CLIP_SECONDS, duration / max(num_clips, 1)))
                highlights = [
                    {"start": round(i * span, 2),
                     "end": round(min(duration, i * span + span), 2),
                     "reason": "Evenly sampled segment (no transcript available).",
                     "viral_score": 60}
                    for i in range(num_clips)
                    if i * span + 3 < duration
                ]
            step("detect", "done", f"{len(highlights)} clips selected")
        if not highlights:
            raise ClipperError("No highlight candidates were found for this video.")
    except ClipperError as exc:
        step("detect", "error", str(exc))
        raise

    # --- Steps 4-6: cut, frame, caption --------------------------------------
    clips: list[ClipResult] = []
    total = len(highlights)
    try:
        for i, hl in enumerate(highlights, start=1):
            step("cut", "running", f"Processing clip {i}/{total}…")
            raw_clip = os.path.join(out_dir, f"_raw_clip_{i}.mp4")
            final_clip = os.path.join(out_dir, f"clip_{i}.mp4")
            srt_path = os.path.join(out_dir, f"clip_{i}.srt")

            progress("cut", (i - 1) / total * 100, f"Cutting clip {i}/{total}…")
            # Vertical + auto-framing needs the uncropped frame so faces can move
            # the crop window; otherwise cut straight to the final aspect ratio.
            cut_clip(
                source_video, hl["start"], hl["end"], raw_clip,
                aspect_ratio="source" if framing_enabled else config.aspect_ratio,
            )
            framed = raw_clip
            if framing_enabled:
                progress("frame", (i - 1) / total * 100, f"Face-tracking clip {i}/{total}…")
                framed_path = os.path.join(out_dir, f"_framed_clip_{i}.mp4")
                result = apply_auto_framing(raw_clip, framed_path, config.aspect_ratio)
                if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                    framed = result

            clip_words = [
                w for seg in segments
                for w in (seg.get("words") or [])
                if w["end"] > hl["start"] and w["start"] < hl["end"]
            ]
            if segments and clip_words:
                # clip_words are in source time; save_srt shifts them by -offset.
                save_srt(
                    [{"start": w["start"], "end": w["end"], "text": w["word"],
                      "words": [w]} for w in clip_words],
                    srt_path,
                    offset=hl["start"],
                )

            progress("caption", (i - 1) / total * 100, f"Captioning clip {i}/{total}…")
            add_captions(
                framed, final_clip,
                style=config.caption_style,
                words=clip_words or None,
                segments=segments or None,
                offset=hl["start"],
                use_pycaps=config.use_pycaps,
                srt_path=srt_path if os.path.isfile(srt_path) else None,
            )

            for tmp in (raw_clip, framed):
                if tmp != final_clip and os.path.isfile(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

            clips.append(ClipResult(
                index=i, path=final_clip,
                start=float(hl["start"]), end=float(hl["end"]),
                reason=str(hl.get("reason", "")),
                viral_score=int(hl.get("final_score", hl.get("viral_score", 70))),
                srt_path=srt_path if os.path.isfile(srt_path) else None,
            ))
            progress("cut", i / total * 100, f"Clip {i}/{total} done")
        step("cut", "done", f"{len(clips)} clips cut")
        step("frame", "done" if framing_enabled else "skipped",
             "Face-tracked vertical crop" if framing_enabled else "Auto-framing off")
        step("caption", "done", f"{config.caption_style} captions burned")
    except ClipperError as exc:
        step("cut", "error", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        step("cut", "error", str(exc))
        raise ClipperError(f"Clip rendering failed: {exc}") from exc

    # --- Packaging ------------------------------------------------------------
    zip_path = ""
    try:
        step("pack", "running", "Packaging clips…")
        zip_path = create_zip([c.path for c in clips], os.path.join(out_dir, "autoclip_clips.zip"))
        step("pack", "done", "ZIP ready")
    except ClipperError as exc:
        step("pack", "skipped", f"ZIP skipped: {exc}")

    if os.path.isfile(source_video):
        try:
            os.remove(source_video)
        except OSError:
            pass

    return {
        "clips": [c.to_dict() for c in clips],
        "zip_path": zip_path,
        "elapsed": round(time.time() - started, 1),
        "video_title": title,
        "output_dir": out_dir,
    }



