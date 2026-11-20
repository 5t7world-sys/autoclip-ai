import os
import time
from pathlib import Path
import streamlit as st

import clipper
from clipper import (
    ASPECT_RATIOS,
    CAPTION_STYLES,
    ClipperError,
    check_ffmpeg,
    is_valid_youtube_url,
    download_video,
    transcribe_video,
    detect_highlights,
    cut_clip,
    ensure_dir
)

st.set_page_config(page_title="AutoClip AI", page_icon="🎬", layout="wide")

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"

# Custom Dark UI Fix
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0b0d17;
        color: #e8ecf7;
    }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #141827;
        border-right: 1px solid #262c42;
    }
    
    /* Text Colors */
    h1, h2, h3, h4, p, span, label, .stMarkdown {
        color: #e8ecf7 !important;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #1b2033;
        color: #e8ecf7;
        border: 1px solid #7b5cff;
    }
    
    /* Hero Banner */
    .ac-hero {
        background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 52%, #24d3ee 100%);
        padding: 30px;
        border-radius: 22px;
        margin-bottom: 20px;
        box-shadow: 0 18px 48px rgba(123, 92, 255, .28);
    }
    .ac-hero h1 { color: white !important; margin: 0; }
    .ac-hero p { color: rgba(255,255,255,0.9) !important; }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="ac-hero"><h1>🎬 AutoClip AI</h1><p>Paste a YouTube link and get ready-to-post vertical clips (9:16) with AI-picked moments.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    
    # Default 9:16 (Shorts/Reels) set to index 1
    aspect_ratio = st.selectbox("Aspect ratio", list(ASPECT_RATIOS.keys()), index=1)
    num_clips = st.slider("Number of clips", 1, 10, 2)
    caption_style = st.selectbox("Caption style", list(CAPTION_STYLES.keys()), index=0)
    
    st.divider()
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-…").strip()
    if api_key:
        st.caption(f"🔑 Key loaded (...{api_key[-4:]})")
    
    generate = st.button("🚀 Generate Clips", type="primary", use_container_width=True)

def render_results(clips):
    st.markdown("### ✨ Your Clips")
    if not clips:
        st.warning("No clips generated.")
        return
    cols = st.columns(2)
    for i, clip in enumerate(clips):
        with cols[i % 2]:
            st.markdown(f"**Clip {i+1}**")
            path = clip.get("path")
            if path and os.path.isfile(path):
                st.video(path)
                with open(path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download MP4", 
                        data=f, 
                        file_name=os.path.basename(path), 
                        mime="video/mp4", 
                        key=f"dl_{i}"
                    )
            else:
                st.error("Clip file is missing.")

# Error Handling and Generate Logic
if generate:
    if not url or not is_valid_youtube_url(url):
        st.error("Please enter a valid YouTube URL.")
    elif not api_key and not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter an OpenAI API key.")
    elif not check_ffmpeg()[0]:
        st.error("FFmpeg is not installed on this server.")
    else:
        try:
            with st.status("🎬 Processing your video...", expanded=True) as status:
                st.write("1. Downloading video...")
                video_path = str(OUTPUT_DIR / "source.mp4")
                ensure_dir(video_path)
                download_video(url, video_path)
                
                st.write("2. Transcribing audio (Whisper)...")
                segments = transcribe_video(video_path, api_key=api_key)
                
                st.write("3. Finding best highlights (AI)...")
                highlights = detect_highlights(segments, num_clips, api_key)
                
                st.write("4. Cutting clips...")
                results = []
                for i, h in enumerate(highlights):
                    start = max(0, float(h.get("start", 0)))
                    end = min(float(h.get("end", start+30)), start+60)
                    out_path = str(OUTPUT_DIR / f"clip_{i+1}.mp4")
                    cut_clip(video_path, start, end, out_path)
                    results.append({"path": out_path, "start": start, "end": end})
                
                status.update(label="✅ Clips are ready!", state="complete")
                st.session_state.results = results
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if "results" in st.session_state and st.session_state.results:
    render_results(st.session_state.results)
