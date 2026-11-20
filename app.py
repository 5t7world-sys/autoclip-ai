"""AutoClip AI - Streamlit front end.

Paste a YouTube URL, pick a few options and get ready-to-post vertical clips
with word-synced, styled captions.

Run with:
    streamlit run app.py
"""

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
    PipelineConfig,
    check_ffmpeg,
    is_valid_youtube_url,
    probe_video,
    run_pipeline,
)

APP_DIR = Path(__file__).resolve().parent
MAX_FREE_SECONDS = clipper.MAX_VIDEO_SECONDS

# Load .env (optional dependency) so OPENAI_API_KEY can live in a file.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(APP_DIR / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

# Resolved after .env is loaded, so AUTOCLIP_OUTPUT_DIR is honoured.
OUTPUT_DIR = Path(os.environ.get("AUTOCLIP_OUTPUT_DIR", "") or (APP_DIR / "output"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (APP_DIR / OUTPUT_DIR).resolve()

st.set_page_config(
    page_title="AutoClip AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
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
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(123,92,255,.20), transparent 60%),
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
    border-radius: 22px;
    padding: 30px 34px 26px 34px;
    margin: 4px 0 22px 0;
    box-shadow: 0 18px 48px rgba(123, 92, 255, .28);
    position: relative;
    overflow: hidden;
}
.ac-hero::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(600px 220px at 85% 10%, rgba(255,255,255,.28), transparent 60%);
    pointer-events: none;
}
.ac-hero h1 {
    margin: 0; font-size: 2.55rem; line-height: 1.1;
    color: #fff !important; font-weight: 800;
    text-shadow: 0 3px 18px rgba(0,0,0,.30);
}
.ac-hero p {
    margin: .55rem 0 0 0; color: rgba(255,255,255,.94) !important;
    font-size: 1.02rem; max-width: 62ch;
}
.ac-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.ac-badge {
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.34);
    color: #fff; font-size: .78rem; font-weight: 600;
    padding: 5px 12px; border-radius: 999px; backdrop-filter: blur(6px);
}

.ac-card {
    background: linear-gradient(180deg, var(--ac-panel) 0%, var(--ac-panel-2) 100%);
    border: 1px solid var(--ac-border);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,.28);
}
.ac-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
.ac-card .ac-reason { color: var(--ac-muted) !important; font-size: .9rem; line-height: 1.45; }

.ac-score {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,59,123,.16);
    border: 1px solid rgba(255,59,123,.42);
    color: #ff8fb1 !important;
    font-weight: 700; font-size: .78rem;
    padding: 4px 11px; border-radius: 999px;
}
.ac-score.high { background: rgba(36,211,238,.15); border-color: rgba(36,211,238,.45); color: #7fe6f7 !important; }
.ac-time {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .78rem; color: var(--ac-muted) !important;
}

div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    font-weight: 650 !important;
    padding: .58rem 1.05rem !important;
    transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(123,92,255,.32);
    filter: brightness(1.05);
}
div.stButton > button[kind="primary"], div.stDownloadButton > button {
    background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 100%) !important;
    color: #fff !important;
    border: none !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,.05) !important;
    color: var(--ac-text) !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.045) !important;
    border: 1px solid var(--ac-border) !important;
    border-radius: 12px !important;
    color: var(--ac-text) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--ac-accent-2) !important;
    box-shadow: 0 0 0 3px rgba(123,92,255,.22) !important;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--ac-border);
    border-radius: 14px;
}

div[data-testid="stVideo"] video {
    border-radius: 14px;
    border: 1px solid var(--ac-border);
    background: #000;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff3b7b, #7b5cff, #24d3ee);
    border-radius: 999px;
}
div[data-testid="stProgressBar"] { border-radius: 999px; }

div[data-testid="stStatusWidget"] {
    border-radius: 14px;
    border: 1px solid var(--ac-border);
    background: rgba(255,255,255,.03);
}

.ac-steps { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 16px 0; }
.ac-step {
    font-size: .78rem; font-weight: 600; padding: 6px 12px;
    border-radius: 999px; border: 1px solid var(--ac-border);
    background: rgba(255,255,255,.04); color: var(--ac-muted) !important;
}
.ac-step.running { border-color: rgba(123,92,255,.65); color: #c9b8ff !important; background: rgba(123,92,255,.14); }
.ac-step.done { border-color: rgba(36,211,238,.5); color: #7fe6f7 !important; background: rgba(36,211,238,.12); }
.ac-step.error { border-color: rgba(255,59,123,.6); color: #ff9dbb !important; background: rgba(255,59,123,.14); }
.ac-step.skipped { opacity: .55; }

.ac-footer {
    margin-top: 26px; padding-top: 14px;
    border-top: 1px solid var(--ac-border);
    color: var(--ac-muted) !important; font-size: .82rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ffmpeg_status() -> tuple[bool, str]:
    """Cache the FFmpeg availability probe (it shells out)."""
    return check_ffmpeg()


@st.cache_resource(show_spinner=False)
def library_status() -> dict[str, bool]:
    """Which optional libraries are importable in this environment."""
    return {
        "yt-dlp": clipper.yt_dlp is not None,
        "openai": clipper.OpenAI is not None,
        "mediapipe": clipper.mp is not None,
        "pycaps-ai": clipper.TemplateLoader is not None,
        "pillow": clipper.Image is not None,
    }


@st.cache_data(show_spinner=False, ttl=900)
def cached_probe(url: str) -> dict:
    """Cache YouTube metadata lookups for 15 minutes."""
    return probe_video(url)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
STEPS = [
    ("download", "1 · Download"),
    ("transcribe", "2 · Transcribe"),
    ("detect", "3 · Detect highlights"),
    ("vision", "4 · Vision"),
    ("render", "5 · Cut, frame & caption"),
    ("pack", "6 · Package"),
]

# The pipeline reports the render sub-steps separately; the UI shows one chip.
STEP_ALIAS = {"setup": "download", "cut": "render", "frame": "render", "caption": "render"}


def _canonical_step(step_id: str) -> str:
    """Map a pipeline step id onto the chip that displays it."""
    return STEP_ALIAS.get(step_id, step_id)

if "step_state" not in st.session_state:
    st.session_state.step_state = {}
if "results" not in st.session_state:
    st.session_state.results = None
if "step_message" not in st.session_state:
    st.session_state.step_message = {}
if "long_video_ok" not in st.session_state:
    st.session_state.long_video_ok = False
if "long_video_url" not in st.session_state:
    st.session_state.long_video_url = ""
if "running" not in st.session_state:
    st.session_state.running = False


def render_steps() -> None:
    """Paint the pipeline step chips from session state."""
    chips = []
    for key, label in STEPS:
        state = st.session_state.step_state.get(key, "")
        message = st.session_state.step_message.get(key, "")
        title = f"{label} — {message}" if message else label
        chips.append(f'<span class="ac-step {state}" title="{_html_escape(title)}">{label}</span>')
    st.markdown(f'<div class="ac-steps">{"".join(chips)}</div>', unsafe_allow_html=True)


def _html_escape(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def format_ts(seconds: float) -> str:
    """0:07 / 12:34 style timestamp."""
    seconds = max(0.0, float(seconds or 0))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
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
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    demo_mode = st.checkbox(
        "🧪 Demo Mode (no download / no API key)",
        value=False,
        help="Renders a synthetic 30-second clip so you can test the whole pipeline offline.",
    )

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=…",
        disabled=demo_mode,
        help="Any youtube.com/watch, /shorts or youtu.be link.",
    )

    num_clips = st.slider("Number of clips", min_value=1, max_value=10, value=3)

    aspect_ratio = st.selectbox(
        "Aspect ratio", list(ASPECT_RATIOS.keys()), index=1,
        help="9:16 for Shorts/Reels, 16:9 for YouTube, 1:1 for Instagram feed.",
    )

    caption_style = st.selectbox("Caption style", list(CAPTION_STYLES.keys()), index=0)

    with st.expander("🧠 AI options", expanded=True):
        vision_mode = st.checkbox(
            "Enable AI Vision Mode",
            value=False,
            help="Samples 1 frame every 5s and scores visual interest with GPT-4o. Slower, smarter.",
        )
        auto_framing = st.checkbox(
            "Auto-framing for vertical",
            value=True,
            help="Detects faces with mediapipe and keeps the speaker centered in 9:16.",
        )
        high_accuracy = st.checkbox(
            "High accuracy transcription",
            value=False,
            help="Uses the 'small' Whisper profile: slower, better with accents and noisy audio.",
        )
        use_pycaps = st.checkbox(
            "Use pycaps-ai templates",
            value=True,
            help="Try pycaps-ai first for captions; falls back to the built-in FFmpeg renderer.",
        )

    api_key_input = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-…",
        help="Never stored on disk. You can also set OPENAI_API_KEY in .env.",
    )
    env_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = (api_key_input or env_key).strip()
    if api_key:
        st.caption(f"🔑 Key loaded (…{api_key[-4:]})" if len(api_key) > 4 else "🔑 Key loaded")
    else:
        st.caption("🔑 No key — offline heuristic mode")

    st.divider()
    generate = st.button("🚀 Generate Clips", type="primary", use_container_width=True)

    with st.expander("🩺 System check"):
        ok_ffmpeg, ffmpeg_msg = ffmpeg_status()
        st.write(("✅ " if ok_ffmpeg else "❌ ") + "FFmpeg")
        if not ok_ffmpeg:
            st.code(ffmpeg_msg, language="text")
        for name, present in library_status().items():
            st.write(("✅ " if present else "⚠️ ") + name)

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def validate_inputs() -> tuple[bool, str]:
    """Return (ok, error_message) for the current sidebar values."""
    if demo_mode:
        return True, ""
    if not url.strip():
        return False, "Please paste a YouTube URL, or turn on 🧪 Demo Mode to try it without one."
    if not is_valid_youtube_url(url.strip()):
        return False, (
            "That doesn't look like a YouTube URL. Examples:\n"
            "• https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
            "• https://youtu.be/dQw4w9WgXcQ\n"
            "• https://www.youtube.com/shorts/abc123"
        )
    if not api_key:
        st.info(
            "No OpenAI API key — running in offline mode: highlights come from a "
            "speech-energy heuristic and captions are timed from the transcript if available."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Long-video confirmation
# ---------------------------------------------------------------------------
def confirm_long_video() -> bool:
    """Warn and ask for confirmation when the video exceeds 30 minutes."""
    if demo_mode:
        return True
    # A confirmation only applies to the URL it was given for.
    if st.session_state.long_video_ok and st.session_state.long_video_url == url.strip():
        return True
    try:
        info = cached_probe(url.strip())
    except ClipperError:
        return True          # can't probe: let the download step report the real error
    duration = float(info.get("duration") or 0)
    if duration <= MAX_FREE_SECONDS:
        return True

    st.warning(
        f"⏳ This video is **{format_ts(duration)}** long "
        f"(over the {MAX_FREE_SECONDS // 60}-minute guideline). Transcription and "
        "vision analysis will take a while and use more API credits."
    )
    st.caption(f"🎞️ {info.get('title', '')}" + (f" — {info['uploader']}" if info.get("uploader") else ""))
    if st.button("✅ I understand — process it anyway", type="secondary"):
        st.session_state.long_video_ok = True
        st.session_state.long_video_url = url.strip()
        st.session_state.confirmed_run = True
        st.rerun()
    return False


# ---------------------------------------------------------------------------
# Results view
# ---------------------------------------------------------------------------
def render_results(results: dict) -> None:
    """Render clip cards, per-clip downloads and the ZIP download."""
    clips = results.get("clips") or []
    st.markdown("### ✨ Your clips")
    st.caption(
        f"🎞️ {results.get('video_title', '')} · ⏱️ processed in "
        f"{results.get('elapsed', 0)}s · 📁 saved to `{results.get('output_dir', 'output')}`"
    )

    zip_path = results.get("zip_path") or ""
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(
            f'<span class="ac-muted">Generated {len(clips)} clip(s) with '
            f'{len([c for c in clips if c.get("srt_path")])} subtitle file(s).</span>',
            unsafe_allow_html=True,
        )
    with top_right:
        if zip_path and os.path.isfile(zip_path):
            data = read_bytes(zip_path)
            if data:
                st.download_button(
                    "⬇️ Download all (ZIP)",
                    data=data,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip",
                    use_container_width=True,
                    key="zip_all",
                )

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
                      ({clip.get('duration', 0):.1f}s)
                    </span>
                  </div>
                  <p class="ac-reason">{_html_escape(clip.get('reason', ''))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            path = clip.get("path", "")
            if os.path.isfile(path):
                try:
                    st.video(path)
                except Exception:
                    st.caption("Preview unavailable — use the download button.")
                video_bytes = read_bytes(path)
                if video_bytes:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=video_bytes,
                        file_name=os.path.basename(path),
                        mime="video/mp4",
                        key=f"dl_video_{i}",
                        use_container_width=True,
                    )
                srt = clip.get("srt_path")
                if srt and os.path.isfile(srt):
                    srt_bytes = read_bytes(srt)
                    if srt_bytes:
                        st.download_button(
                            "📝 Download SRT",
                            data=srt_bytes,
                            file_name=os.path.basename(srt),
                            mime="text/plain",
                            key=f"dl_srt_{i}",
                            use_container_width=True,
                        )
            else:
                st.error(f"Clip file is missing: {path}")


# ---------------------------------------------------------------------------
# Main action
# ---------------------------------------------------------------------------
render_steps()

# A long-video confirmation click sets confirmed_run, then reruns the script.
try:
    confirmed_run = bool(st.session_state.get("confirmed_run", False))
    if confirmed_run:
        st.session_state.confirmed_run = False
except Exception:
    confirmed_run = False
run_requested = bool(generate or confirmed_run)

if run_requested:
    if st.session_state.running:
        st.warning("A run is already in progress — please wait for it to finish.")
    else:
        ok, error = validate_inputs()
        if not ok:
            st.error(error)
        elif not ffmpeg_status()[0]:
            st.error("FFmpeg is required but was not found.")
            st.code(ffmpeg_status()[1], language="text")
        elif confirm_long_video():
            st.session_state.running = True
            st.session_state.step_state = {}
            st.session_state.step_message = {}
            st.session_state.results = None

            progress_state = {"index": 0, "sub": "Starting…"}
            bar = st.progress(0.0, text="Starting…")

            def step_cb(step_id: str, state: str, message: str = "") -> None:
                key = _canonical_step(step_id)
                st.session_state.step_state[key] = state
                st.session_state.step_message[key] = message
                keys = [k for k, _ in STEPS]
                if key in keys:
                    index = keys.index(key)
                    # Never let an aliased sub-step (cut → frame → caption)
                    # move the outer progress bar backwards.
                    progress_state["index"] = max(progress_state["index"], index)
                    progress_state["sub"] = message
                if state == "error" and key not in {"vision"}:
                    progress_state["failed"] = True

            def progress_cb(step_id: str, pct: float, message: str = "") -> None:
                total_steps = max(1, len(STEPS))
                index = progress_state["index"]
                overall = (index + max(0.0, min(100.0, pct)) / 100.0) / total_steps
                bar.progress(min(0.999, overall), text=message or step_id)

            config = PipelineConfig(
                url=url.strip(),
                num_clips=num_clips,
                aspect_ratio=aspect_ratio,
                caption_style=caption_style,
                api_key=api_key,
                vision_mode=vision_mode,
                auto_framing=auto_framing,
                whisper_model="small" if high_accuracy else "base",
                demo_mode=demo_mode,
                use_pycaps=use_pycaps,
                output_dir=str(OUTPUT_DIR),
            )

            started = time.time()
            try:
                with st.status("🎬 Working on your clips…", expanded=True) as status:
                    st.write("Preparing pipeline…")
                    results = run_pipeline(
                        config,
                        step_cb=step_cb,
                        progress_cb=progress_cb,
                    )
                    status.update(
                        label=f"✅ Done in {results['elapsed']}s",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.results = results
                bar.progress(1.0, text=f"Finished in {time.time() - started:.1f}s")
                st.toast("Clips are ready 🎉", icon="🎉")
            except ClipperError as exc:
                bar.empty()
                st.error(f"❌ {exc}")
                with st.expander("What can I do?"):
                    st.markdown(
                        "- Check that the URL is public (not private or age-restricted).\n"
                        "- Verify your OpenAI key has credit and `whisper-1` / `gpt-4o-mini` access.\n"
                        "- Try **🧪 Demo Mode** to confirm the pipeline itself works.\n"
                        "- Re-run the system check in the sidebar."
                    )
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(f"❌ Unexpected error: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc(), language="python")
            finally:
                st.session_state.running = False
            render_steps()

# Persisted results across reruns
if st.session_state.results:
    st.divider()
    render_results(st.session_state.results)
else:
    if not run_requested:
        st.markdown(
            """
            <div class="ac-card">
              <h4>How it works</h4>
              <p class="ac-reason">
                1. <b>Download</b> — yt-dlp grabs up to 1080p MP4.<br>
                2. <b>Transcribe</b> — Whisper returns word-level timestamps.<br>
                3. <b>Detect</b> — GPT-4o-mini ranks the most viral 20–60s moments
                   (optionally blended with GPT-4o vision frame scores).<br>
                4. <b>Cut &amp; frame</b> — FFmpeg cuts each clip; faces are tracked
                   for vertical crops.<br>
                5. <b>Caption</b> — word-synced, styled captions are burned in.<br>
                6. <b>Package</b> — MP4 + SRT files, plus a ZIP of everything.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Set your options in the sidebar, then hit **🚀 Generate Clips**. "
                "No API key? Enable 🧪 Demo Mode.")

st.markdown(
    '<div class="ac-footer">AutoClip AI · clips are generated locally in ./output · '
    'respect copyright and YouTube\'s Terms of Service when republishing.</div>',
    unsafe_allow_html=True,
)
