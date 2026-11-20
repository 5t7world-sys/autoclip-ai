"""AutoClip AI - Streamlit front end."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import streamlit as st

import clipper
from clipper import (
    ASPECT_RATIOS,
    CAPTION_STYLES,
    ClipperError,
    check_ffmpeg,
    is_valid_youtube_url,
    ffprobe_duration as probe_video,
    download_video,
    transcribe_video,
    ensure_dir
)

APP_DIR = Path(__file__).resolve().parent
MAX_FREE_SECONDS = clipper.MAX_VIDEO_SECONDS

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(APP_DIR / ".env")
except Exception:
    pass

OUTPUT_DIR = Path(os.environ.get("AUTOCLIP_OUTPUT_DIR", "") or (APP_DIR / "output"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (APP_DIR / OUTPUT_DIR).resolve()

st.set_page_config(
    page_title="AutoClip AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --ac-bg: #0b0d17;
    --ac-panel: #141827;
    --ac-panel-2: #1b2033;
    --ac-border: #262c42;
    --ac-text: #e8ecf7;
    --ac-muted: #9aa4bf;
    --ac-accent: #ff3b7b;
    --ac-accent-2: #7b5cff;
    --ac-accent-3: #24d3ee;
}
.stApp {
    background: radial-gradient(1200px 600px at 12% -10%, rgba(123,92,255,.20), transparent 60%),
        radial-gradient(900px 500px at 92% 0%, rgba(255,59,123,.16), transparent 55%),
        var(--ac-bg);
    color: var(--ac-text);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10131f 0%, #0d1018 100%);
    border-right: 1px solid var(--ac-border);
}
section[data-testid="stSidebar"] * { color: var(--ac-text); }
h1, h2, h3, h4 { color: var(--ac-text) !important; letter-spacing: -0.01em; }
p, span, label, li, div { color: var(--ac-text); }
small, .ac-muted { color: var(--ac-muted) !important; }
.ac-hero {
    background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 52%, #24d3ee 100%);
    border-radius: 22px; padding: 30px 34px 26px 34px; margin: 4px 0 22px 0;
    box-shadow: 0 18px 48px rgba(123, 92, 255, .28); position: relative; overflow: hidden;
}
.ac-hero::after {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(600px 220px at 85% 10%, rgba(255,255,255,.28), transparent 60%); pointer-events: none;
}
.ac-hero h1 {
    margin: 0; font-size: 2.55rem; line-height: 1.1; color: #fff !important; font-weight: 800;
    text-shadow: 0 3px 18px rgba(0,0,0,.30);
}
.ac-hero p { margin: .55rem 0 0 0; color: rgba(255,255,255,.94) !important; font-size: 1.02rem; max-width: 62ch; }
.ac-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.ac-badge {
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.34);
    color: #fff; font-size: .78rem; font-weight: 600; padding: 5px 12px; border-radius: 999px; backdrop-filter: blur(6px);
}
.ac-card {
    background: linear-gradient(180deg, var(--ac-panel) 0%, var(--ac-panel-2) 100%);
    border: 1px solid var(--ac-border); border-radius: 18px; padding: 16px 18px;
    margin-bottom: 14px; box-shadow: 0 10px 28px rgba(0,0,0,.28);
}
.ac-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
.ac-card .ac-reason { color: var(--ac-muted) !important; font-size: .9rem; line-height: 1.45; }
.ac-score {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,59,123,.16); border: 1px solid rgba(255,59,123,.42);
    color: #ff8fb1 !important; font-weight: 700; font-size: .78rem;
    padding: 4px 11px; border-radius: 999px;
}
.ac-score.high { background: rgba(36,211,238,.15); border-color: rgba(36,211,238,.45); color: #7fe6f7 !important; }
.ac-time { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; color: var(--ac-muted) !important; }
div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px !important; border: 1px solid rgba(255,255,255,.10) !important;
    font-weight: 650 !important; padding: .58rem 1.05rem !important; transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 10px 26px rgba(123,92,255,.32); filter: brightness(1.05);
}
div.stButton > button[kind="primary"], div.stDownloadButton > button {
    background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 100%) !important; color: #fff !important; border: none !important;
}
div.stButton > button[kind="secondary"] { background: rgba(255,255,255,.05) !important; color: var(--ac-text) !important; }
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.045) !important; border: 1px solid var(--ac-border) !important; border-radius: 12px !important; color: var(--ac-text) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--ac-accent-2) !important; box-shadow: 0 0 0 3px rgba(123,92,255,.22) !important;
}
div[data-testid="stExpander"] { background: rgba(255,255,255,.03); border: 1px solid var(--ac-border); border-radius: 14px; }
div[data-testid="stVideo"] video { border-radius: 14px; border: 1px solid var(--ac-border); background: #000; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #ff3b7b, #7b5cff, #24d3ee); border-radius: 999px; }
div[data-testid="stProgressBar"] { border-radius: 999px; }
div[data-testid="stStatusWidget"] { border-radius: 14px; border: 1px solid var(--ac-border); background: rgba(255,255,255,.03); }
.ac-steps { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 16px 0; }
.ac-step {
    font-size: .78rem; font-weight: 600; padding: 6px 12px; border-radius: 999px;
    border: 1px solid var(--ac-border); background: rgba(255,255,255,.04); color: var(--ac-muted) !important;
}
.ac-step.running { border-color: rgba(123,92,255,.65); color: #c9b8ff !important; background: rgba(123,92,255,.14); }
.ac-step.done { border-color: rgba(36,211,238,.5); color: #7fe6f7 !important; background: rgba(36,211,238,.12); }
.ac-step.error { border-color: rgba(255,59,123,.6); color: #ff9dbb !important; background: rgba(255,59,123,.14); }
.ac-step.skipped { opacity: .55; }
.ac-footer { margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--ac-border); color: var(--ac-muted) !important; font-size: .82rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def ffmpeg_status() -> tuple[bool, str]:
    return check_ffmpeg()


@st.cache_resource(show_spinner=False)
def library_status() -> dict[str, bool]:
    return {
        "yt-dlp": clipper.yt_dlp is not None,
        "openai": clipper.OpenAI is not None,
        "mediapipe": clipper.mp is not None,
        "pycaps-ai": clipper.TemplateLoader is not None,
        "pillow": clipper.Image is not None,
    }


STEPS = [
    ("download", "1 · Download"),
    ("transcribe", "2 · Transcribe"),
    ("detect", "3 · Detect highlights"),
    ("vision", "4 · Vision"),
    ("render", "5 · Cut, frame & caption"),
    ("pack", "6 · Package"),
]
STEP_ALIAS = {"setup": "download", "cut": "render", "frame": "render", "caption": "render"}

def _canonical_step(step_id: str) -> str:
    return STEP_ALIAS.get(step_id, step_id)

if "step_state" not in st.session_state: st.session_state.step_state = {}
if "results" not in st.session_state: st.session_state.results = None
if "step_message" not in st.session_state: st.session_state.step_message = {}
if "long_video_ok" not in st.session_state: st.session_state.long_video_ok = False
if "long_video_url" not in st.session_state: st.session_state.long_video_url = ""
if "running" not in st.session_state: st.session_state.running = False


def render_steps() -> None:
    chips = []
    for key, label in STEPS:
        state = st.session_state.step_state.get(key, "")
        message = st.session_state.step_message.get(key, "")
        title = f"{label} — {message}" if message else label
        chips.append(f'<span class="ac-step {state}" title="{_html_escape(title)}">{label}</span>')
    st.markdown(f'<div class="ac-steps">{"".join(chips)}</div>', unsafe_allow_html=True)


def _html_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as fh: return fh.read()
    except OSError: return None


st.markdown(
    """
    <div class="ac-hero">
      <h1>🎬 AutoClip AI</h1>
      <p>Paste a YouTube link and get ready-to-post highlight clips with AI-picked
      moments and word-synced, styled captions. Built for Shorts, Reels and TikTok.</p>
      <div class="ac-badges">
        <span class="ac-badge">Whisper transcription</span>
        <span class="ac-badge">GPT-4o highlight scoring</span>
        <span class="ac-badge">Face-tracked vertical crop</span>
        <span class="ac-badge">Karaoke captions</span>
      </div>
    </div>
    """, unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    demo_mode = st.checkbox("🧪 Demo Mode", value=False)
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", disabled=demo_mode)
    num_clips = st.slider("Number of clips", min_value=1, max_value=10, value=3)
    aspect_ratio = st.selectbox("Aspect ratio", list(ASPECT_RATIOS.keys()), index=1)
    caption_style = st.selectbox("Caption style", list(CAPTION_STYLES.keys()), index=0)

    with st.expander("🧠 AI options", expanded=True):
        vision_mode = st.checkbox("Enable AI Vision Mode", value=False)
        auto_framing = st.checkbox("Auto-framing for vertical", value=True)
        high_accuracy = st.checkbox("High accuracy transcription", value=False)
        use_pycaps = st.checkbox("Use pycaps-ai templates", value=True)

    api_key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-…")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = (api_key_input or env_key).strip()
    if api_key: st.caption(f"🔑 Key loaded (…{api_key[-4:]})" if len(api_key) > 4 else "🔑 Key loaded")
    else: st.caption("🔑 No key — offline heuristic mode")

    st.divider()
    generate = st.button("🚀 Generate Clips", type="primary", use_container_width=True)

    with st.expander("🩺 System check"):
        ok_ffmpeg, ffmpeg_msg = ffmpeg_status()
        st.write(("✅ " if ok_ffmpeg else "❌ ") + "FFmpeg")
        for name, present in library_status().items():
            st.write(("✅ " if present else "⚠️ ") + name)

def validate_inputs() -> tuple[bool, str]:
    if demo_mode: return True, ""
    if not url.strip(): return False, "Please paste a YouTube URL."
    if not is_valid_youtube_url(url.strip()): return False, "That doesn't look like a valid YouTube URL."
    return True, ""

def confirm_long_video() -> bool:
    if demo_mode: return True
    if st.session_state.long_video_ok and st.session_state.long_video_url == url.strip(): return True
    try:
        duration = float(probe_video(url.strip()) or 0)
    except Exception:
        return True
    if duration <= MAX_FREE_SECONDS: return True

    st.warning(f"⏳ This video is **{format_ts(duration)}** long. Processing will take time.")
    if st.button("✅ I understand — process it anyway", type="secondary"):
        st.session_state.long_video_ok = True
        st.session_state.long_video_url = url.strip()
        st.rerun()
    return False

def render_results(clips: list) -> None:
    st.markdown("### ✨ Your clips")
    if not clips:
        st.warning("No clips were generated.")
        return
    
    columns = st.columns(2, gap="medium")
    for i, clip in enumerate(clips):
        with columns[i % 2]:
            score = int(clip.get("viral_score", 0) or 0)
            score_class = "ac-score high" if score >= 80 else "ac-score"
            st.markdown(
                f"""
                <div class="ac-card">
                  <h4>Clip {clip.get('index', i + 1)}</h4>
                  <div>
                    <span class="{score_class}">🔥 Viral score {score}</span>
                    <span class="ac-time">
                      {format_ts(clip.get('start', 0))} → {format_ts(clip.get('end', 0))}
                    </span>
                  </div>
                  <p class="ac-reason">{_html_escape(clip.get('reason', ''))}</p>
                </div>
                """, unsafe_allow_html=True,
            )
            path = clip.get("path", "")
            if os.path.isfile(path):
                st.video(path)
                video_bytes = read_bytes(path)
                if video_bytes:
                    st.download_button("⬇️ Download MP4", data=video_bytes, file_name=os.path.basename(path), mime="video/mp4", key=f"dl_video_{i}", use_container_width=True)

render_steps()

if st.session_state.results:
    render_results(st.session_state.results)

if generate and not st.session_state.running:
    ok, error_msg = validate_inputs()
    if not ok:
        st.error(error_msg)
    elif not confirm_long_video():
        pass
    else:
        st.session_state.running = True
        st.session_state.results = None
        st.session_state.step_state = {}
        st.session_state.step_message = {}
        
        # Simulated pipeline for now, since clipper no longer has run_pipeline
        try:
            with st.status("🎬 Working on your clips...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                
                # Step 1: Download
                st.session_state.step_state["download"] = "running"
                render_steps()
                if demo_mode:
                    st.write("🧪 Demo mode: skipping download.")
                    time.sleep(1)
                else:
                    st.write("⬇️ Downloading video...")
                    video_path = str(OUTPUT_DIR / "source.mp4")
                    ensure_dir(video_path)
                    download_video(url.strip(), video_path)
                
                st.session_state.step_state["download"] = "done"
                progress_bar.progress(20)
                render_steps()
                
                # Step 2: Transcribe
                st.session_state.step_state["transcribe"] = "running"
                render_steps()
                if demo_mode:
                    st.write("🧪 Demo mode: skipping transcription.")
                    time.sleep(1)
                else:
                    st.write("🎤 Transcribing audio...")
                    transcribe_video(video_path, model_size="base", api_key=api_key)
                
                st.session_state.step_state["transcribe"] = "done"
                progress_bar.progress(50)
                render_steps()
                
                # Step 3-5: Simulated
                st.write("✂️ Cutting clips... (this is a placeholder, logic needed)")
                time.sleep(2)
                
                st.session_state.step_state["detect"] = "done"
                st.session_state.step_state["render"] = "done"
                st.session_state.step_state["pack"] = "done"
                progress_bar.progress(100)
                render_steps()
                
                # Create dummy result just to show it finished
                dummy_clip_path = str(OUTPUT_DIR / "clip_1.mp4")
                ensure_dir(dummy_clip_path)
                # Create a tiny empty file just so download button appears
                with open(dummy_clip_path, "wb") as f:
                    f.write(b"dummy")
                    
                st.session_state.results = [
                    {
                        "index": 1,
                        "path": dummy_clip_path,
                        "start": 0,
                        "end": 30,
                        "reason": "This is a placeholder clip. Real cut logic is missing in the simplified clipper.py.",
                        "viral_score": 85
                    }
                ]
                
                status.update(label="✅ Clips ready!", state="complete")
                
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.session_state.step_state["download"] = "error"
            render_steps()
        finally:
            st.session_state.running = False
            st.rerun()

if st.session_state.results:
    render_results(st.session_state.results)
