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
    if not OpenAI or not api_key:
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
    _run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path, "-c", "copy", output_path])
    return output_path