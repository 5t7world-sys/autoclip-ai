from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Callable

try:
    import yt_dlp
except Exception:
    yt_dlp = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import mediapipe as mp
except Exception:
    mp = None

class ClipperError(RuntimeError):
    pass

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

MIN_CLIP_SECONDS = 20.0
MAX_CLIP_SECONDS = 60.0
MAX_VIDEO_SECONDS = 30 * 60

def which(binary: str) -> Optional[str]:
    return shutil.which(binary)

def check_ffmpeg() -> tuple[bool, str]:
    if which("ffmpeg") and which("ffprobe"):
        return True, "FFmpeg found"
    return False, "FFmpeg is not installed."

def _run(cmd: list, timeout: Optional[int] = 3600) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore").strip()
        raise ClipperError(f"Command failed: {err[-200:]}")
    return proc

def ffprobe_duration(video_path: str) -> float:
    if not which("ffprobe"): return 0.0
    try:
        out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], timeout=120).stdout.decode().strip()
        return float(out)
    except:
        return 0.0

def ensure_dir(path: str) -> str:
    directory = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if directory: os.makedirs(directory, exist_ok=True)
    return path

def is_valid_youtube_url(url: str) -> bool:
    if not url: return False
    return "youtube.com" in url or "youtu.be" in url

def download_video(url: str, output_path: str, progress_cb: Optional[Callable] = None) -> str:
    if yt_dlp is None: raise ClipperError("yt-dlp is not installed.")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    def hook(data):
        if progress_cb and data.get("status") == "downloading":
            total = data.get("total_bytes") or 1
            done = data.get("downloaded_bytes") or 0
            progress_cb((done/total)*100, "Downloading...")

    opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "progress_hooks": [hook],
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
        "http_headers": {"User-Agent": "Mozilla/5.0"},
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise ClipperError(f"Download failed: {exc}")
    return output_path

def transcribe_video(video_path: str, model_size: str = "base", api_key: str = "", progress_cb: Optional[Callable] = None) -> list:
    if not OpenAI: raise ClipperError("openai package missing.")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key: raise ClipperError("OpenAI API key required.")
    
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    
    _run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_path])
    
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), fh.read()),
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    
    segments = []
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    for seg in data.get("segments") or []:
        segments.append({
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "text": str(seg.get("text", "")).strip(),
        })
    return segments

def detect_highlights(segments: list, num_clips: int, api_key: str) -> list:
    """AI को ट्रांसक्रिप्ट भेजकर बेस्ट पार्ट्स ढूंढता है।"""
    if not OpenAI or not api_key:
        # Fallback: just take the first few chunks
        clips = []
        for i in range(num_clips):
            start = i * 30
            clips.append({"start": float(start), "end": float(start+30), "reason": "Auto selected based on time."})
        return clips

    client = OpenAI(api_key=api_key)
    text = "\n".join([f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments])
    
    prompt = f"Here is a video transcript. Pick {num_clips} most engaging moments. Return JSON: [{{\"start\": float, \"end\": float, \"reason\": str}}]. Transcript:\n{text[:5000]}"
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("clips", result) if isinstance(result, dict) else result
    except Exception as e:
        print("LLM failed:", e)
        return [{"start": 0, "end": 30, "reason": "Fallback clip"}]

def cut_clip(video_path: str, start: float, end: float, output_path: str) -> str:
    """FFmpeg से वीडियो काटता है।"""
    _run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path, "-c", "copy", output_path])
    return output_path     "outtmpl": template,
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
