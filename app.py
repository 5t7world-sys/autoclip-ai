import os
import time
import json
from pathlib import Path
import streamlit as st
import clipper
from clipper import ASPECT_RATIOS, CAPTION_STYLES, ClipperError, check_ffmpeg, is_valid_youtube_url, download_video, transcribe_video, detect_highlights, cut_clip, ensure_dir

st.set_page_config(page_title="AutoClip AI", page_icon="🎬", layout="wide")

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"

st.markdown("""
    <style>
    .stApp { background: #0b0d17; color: white; }
    h1, h2, h3 { color: #e8ecf7 !important; }
    .ac-hero { background: linear-gradient(120deg, #ff3b7b 0%, #7b5cff 52%, #24d3ee 100%); padding: 30px; border-radius: 22px; margin-bottom: 20px; }
    .ac-hero h1 { color: white !important; margin: 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="ac-hero"><h1>🎬 AutoClip AI</h1><p>Paste a YouTube link and get ready-to-post highlight clips.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    num_clips = st.slider("Number of clips", 1, 10, 3)
    aspect_ratio = st.selectbox("Aspect ratio", list(ASPECT_RATIOS.keys()), index=1)
    caption_style = st.selectbox("Caption style", list(CAPTION_STYLES.keys()), index=0)
    api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-…").strip()
    
    st.divider()
    generate = st.button("🚀 Generate Clips", type="primary", use_container_width=True)

def render_results(clips):
    st.markdown("### ✨ Your clips")
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
                    st.download_button("⬇️ Download MP4", f, file_name=os.path.basename(path), mime="video/mp4", key=f"dl_{i}")
            else:
                st.error("File missing.")

if generate:
    if not url or not is_valid_youtube_url(url):
        st.error("Please enter a valid YouTube URL.")
    elif not api_key and not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter an OpenAI API key.")
    elif not check_ffmpeg()[0]:
        st.error("FFmpeg is missing on the server.")
    else:
        try:
            with st.status("🎬 Processing...", expanded=True) as status:
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
                    results.append({"path": out_path, "start": start, "end": end, "reason": h.get("reason", "")})
                
                status.update(label="✅ Done!", state="complete")
                st.session_state.results = results
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if "results" in st.session_state:
    render_results(st.session_state.results), value=True)

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
