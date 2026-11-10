# Smart Video Highlight & GIF Generator
# Main entry point (Streamlit app)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

from core import (
    get_video_metadata,
    sample_frames_chunked,
    extract_audio,
    compute_visual_scores,
    compute_audio_scores,
    fuse_and_detect_highlights,
    create_highlight_reel
)
from utils import (
    create_temp_dir,
    cleanup_temp_dir,
    format_timestamp
)

st.set_page_config(
    page_title="Smart Video Highlight Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* main styling */

    /* header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
    }

    /* segment cards */
    .segment-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3d);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .segment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }

    /* stat boxes */
    .stat-box {
        background: linear-gradient(135deg, #1e2235, #282e44);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-box .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.25rem;
    }

    /* footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.85rem;
    }

    /* hide default streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #151b28 100%);
    }

    /* upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 12px;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎬 Smart Video Highlight & GIF Generator</h1>
    <p>Upload a video and automatically extract the most exciting moments using visual action 
    detection and audio climax analysis — no GPU needed, everything runs locally.</p>
</div>
""", unsafe_allow_html=True)

# session state defaults
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'fused_timeline' not in st.session_state:
    st.session_state.fused_timeline = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None

# sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Processing Settings")
    st.markdown("---")

    st.markdown("### 🎯 Score Weights")
    st.caption("Control the relative importance of visual vs. audio signals.")

    visual_weight = st.slider(
        "👁️ Visual Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How much visual action (motion, scene changes) influences highlight selection."
    )

    audio_weight = st.slider(
        "🔊 Audio Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How much audio excitement (loudness, brightness, transients) influences highlight selection."
    )

    st.markdown("---")

    st.markdown("### 🔍 Detection Parameters")

    threshold_percentile = st.slider(
        "📊 Threshold Percentile",
        min_value=50,
        max_value=99,
        value=75,
        step=1,
        help="Higher = more selective (fewer, more intense highlights). "
             "75 means the top 25% of scored moments qualify."
    )

    min_clip_duration = st.slider(
        "⏱️ Min Clip Duration (s)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Minimum duration of each highlight clip in seconds. "
             "Short clips are padded to at least this length."
    )

    max_total_duration = st.slider(
        "📏 Max Total Duration (s)",
        min_value=5,
        max_value=300,
        value=30,
        step=5,
        help="Maximum total duration of the highlight reel. "
             "Only the highest-scoring segments are kept."
    )

    st.markdown("---")

    st.markdown("### 📤 Output Settings")

    project_preset = st.selectbox(
        "🎬 Project Type (Destination)",
        options=[
            "Standard Trailer (16:9 Landscape)",
            "TikTok / Reel (9:16 Vertical)",
            "WebP Animation (GIF)"
        ],
        index=0,
        help="Choose the platform or format you are exporting for."
    )

    if "TikTok" in project_preset:
        output_format = "mp4"
        vertical_crop = True
        extract_single_peak = False
    elif "WebP" in project_preset:
        output_format = "mp4_loop"
        vertical_crop = False
        extract_single_peak = True
    else:
        output_format = "mp4"
        vertical_crop = False
        extract_single_peak = False

    st.markdown("---")

    with st.expander("🔧 Advanced Settings"):

        analysis_fps = st.slider(
            "Analysis FPS",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
            help="Frames per second to sample for visual analysis. "
                 "Lower = faster processing, higher = more precise detection. "
                 "10 fps is recommended for most videos."
        )

        transition_duration = st.slider(
            "Transition Duration (s)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Duration of fade-in/fade-out transitions between highlight clips."
        )

st.markdown("### 📁 Upload Your Video")

uploaded_file = st.file_uploader(
    "Drag and drop a video file or click to browse",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Supported formats: MP4, AVI, MOV, MKV, WebM. Max size depends on Streamlit config."
)

if uploaded_file is not None:

    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = create_temp_dir()

    temp_video_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

    # only re-save if it's a new file
    if st.session_state.video_path != temp_video_path:
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_path = temp_video_path

        # clear old results
        st.session_state.processing_done = False
        st.session_state.segments = None
        st.session_state.fused_timeline = None
        st.session_state.output_path = None
        st.rerun()

    metadata = get_video_metadata(temp_video_path)
    st.session_state.metadata = metadata

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{metadata['duration']:.1f}s</div>
            <div class="stat-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{metadata['width']}×{metadata['height']}</div>
            <div class="stat-label">Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{metadata['fps']:.1f}</div>
            <div class="stat-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{metadata['frame_count']}</div>
            <div class="stat-label">Total Frames</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    with st.expander("🎥 Preview Original Video", expanded=False):
        st.video(temp_video_path)

    st.markdown("---")

    process_col, info_col = st.columns([1, 2])
    with process_col:
        process_button = st.button(
            "🚀 Generate Highlights",
            type="primary",
            use_container_width=True
        )
    with info_col:
        st.info(
            f"Will analyze at **{analysis_fps} fps** "
            f"(~{int(metadata['duration'] * analysis_fps)} frames to process). "
            f"Output: **{output_format.upper()}** "
            f"{'+ vertical crop 📱' if vertical_crop else ''}"
        )

    if process_button:

        st.session_state.processing_done = False
        st.session_state.output_path = None

        with st.status("🔄 Processing video...", expanded=True) as status:

            import concurrent.futures
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

            st.write("🎞️ & 🔊 **Step 1 & 2:** Computing visual and audio scores in parallel...")
            progress_bar = st.progress(0, text="Reading video frames...")

            ctx = get_script_run_ctx()

            def update_frame_progress(current, total):
                if total > 0:
                    progress_bar.progress(
                        min(current / total, 1.0),
                        text=f"Analyzing video chunks: {current}/{total} frames"
                    )

            def process_visual_pipeline():

                add_script_run_ctx(ctx=ctx)
                
                all_vt = []
                all_vs = []
                for chunk_t, chunk_gray, chunk_color in sample_frames_chunked(
                    temp_video_path,
                    analysis_fps=analysis_fps,
                    chunk_size=500,
                    progress_callback=update_frame_progress
                ):
                    vt, vs = compute_visual_scores(chunk_t, chunk_gray, chunk_color)
                    all_vt.append(vt)
                    all_vs.append(vs)
                
                if all_vt:
                    return np.concatenate(all_vt), np.concatenate(all_vs)
                return np.array([]), np.array([])

            def process_audio_pipeline():

                add_script_run_ctx(ctx=ctx)
                
                audio_path = extract_audio(temp_video_path, st.session_state.temp_dir)
                if audio_path:
                    audio_t, audio_s = compute_audio_scores(audio_path, metadata['duration'])
                    return audio_t, audio_s, audio_path
                return np.array([]), np.array([]), None

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_visual = executor.submit(process_visual_pipeline)
                future_audio = executor.submit(process_audio_pipeline)

                visual_timestamps, visual_scores = future_visual.result()
                audio_timestamps, audio_scores, audio_path = future_audio.result()

            progress_bar.progress(1.0, text="✅ Visual analysis complete")
            st.write(f"✅ Visual scores computed ({len(visual_scores)} data points)")
            
            if audio_path:
                st.write(f"✅ Audio scores computed ({len(audio_scores)} data points)")
            else:
                st.write("⚠️ No audio track found — using visual scores only")
                audio_timestamps = visual_timestamps
                audio_scores = np.zeros_like(visual_scores)

            # fuse scores + detect highlights
            st.write("🔀 **Step 5/5:** Fusing scores and detecting highlights...")
            st.write(f"    ↳ Visual weight: {visual_weight:.2f}, "
                     f"Audio weight: {audio_weight:.2f}, "
                     f"Threshold: {threshold_percentile}th percentile")

            segments, fused_timeline = fuse_and_detect_highlights(
                visual_timestamps, visual_scores,
                audio_timestamps, audio_scores,
                visual_weight=visual_weight,
                audio_weight=audio_weight,
                threshold_percentile=threshold_percentile,
                min_clip_duration=min_clip_duration,
                max_total_duration=max_total_duration,
                video_duration=metadata['duration'],
                extract_single_peak=extract_single_peak
            )

            st.write(f"✅ Detected **{len(segments)}** highlight segments")

            st.session_state.segments = segments
            st.session_state.fused_timeline = fused_timeline

            # generate output
            if segments:
                st.write(f"✂️ **Generating {output_format.upper()} highlight reel...**")

                output_path = create_highlight_reel(
                    temp_video_path,
                    segments,
                    output_format=output_format,
                    vertical_crop=vertical_crop,
                    output_dir=st.session_state.temp_dir,
                    transition_duration=transition_duration
                )

                st.session_state.output_path = output_path
                st.write(f"✅ Highlight reel saved!")
            else:
                st.warning("No highlight segments detected. Try lowering the threshold.")
                st.session_state.output_path = None

            st.session_state.processing_done = True
            status.update(label="✅ Processing complete!", state="complete")

    if st.session_state.processing_done and st.session_state.fused_timeline:

        st.markdown("---")
        st.markdown("### 📊 Interest Score Timeline")

        fused_data = st.session_state.fused_timeline
        common_t = fused_data['common_timestamps']
        visual_aligned = fused_data['visual_aligned']
        audio_aligned = fused_data['audio_aligned']
        fused_scores = fused_data['fused_scores']
        threshold_val = fused_data['threshold_value']

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(common_t, visual_aligned,
                color='#4ECDC4', alpha=0.5, linewidth=1.2,
                label='Visual Score')
        ax.plot(common_t, audio_aligned,
                color='#FF6B6B', alpha=0.5, linewidth=1.2,
                label='Audio Score')
        ax.plot(common_t, fused_scores,
                color='#FFE66D', linewidth=2.0, alpha=0.9,
                label='Fused Score')

        ax.axhline(
            y=threshold_val,
            color='#888888',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            label=f'Threshold ({threshold_val:.2f})'
        )

        if st.session_state.segments:
            for i, (seg_start, seg_end) in enumerate(st.session_state.segments):
                ax.axvspan(
                    seg_start, seg_end,
                    alpha=0.15,
                    color='#95E855',
                    label='Highlight' if i == 0 else None
                )

        ax.set_xlabel('Time (seconds)', fontsize=11, color='#cccccc')
        ax.set_ylabel('Score (0–1)', fontsize=11, color='#cccccc')
        ax.set_title('Visual + Audio Interest Score Timeline',
                     fontsize=14, fontweight='bold', color='#ffffff', pad=15)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### ✂️ Detected Highlight Segments")

        if st.session_state.segments:
            segments = st.session_state.segments

            total_hl_duration = sum(e - s for s, e in segments)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Segments Found", len(segments))
            with m_col2:
                st.metric("Total Highlight Duration", f"{total_hl_duration:.1f}s")
            with m_col3:
                compression_ratio = (1 - total_hl_duration / metadata['duration']) * 100
                st.metric("Content Reduction", f"{compression_ratio:.0f}%")

            for i, (seg_start, seg_end) in enumerate(segments):
                duration = seg_end - seg_start
                st.markdown(
                    f"""<div class="segment-card">
                        <strong>Segment {i+1}</strong> &nbsp;|&nbsp;
                        🕐 {format_timestamp(seg_start)} → {format_timestamp(seg_end)}
                        &nbsp;|&nbsp; ⏱️ Duration: {duration:.1f}s
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.warning(
                "No highlight segments were detected. "
                "Try lowering the **Threshold Percentile** in the sidebar, "
                "or adjusting the score weights."
            )

        st.markdown("---")
        st.markdown("### 📹 Highlight Reel Output")

        if st.session_state.output_path and os.path.exists(st.session_state.output_path):
            output_path = st.session_state.output_path

            if output_format in ['mp4', 'webm', 'mp4_loop']:

                if vertical_crop:
                    col1, col2, col3 = st.columns([1.5, 1, 1.5])
                else:
                    col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if output_format == 'mp4_loop':
                        st.video(output_path, loop=True, autoplay=True, muted=True)
                    else:
                        st.video(output_path)
            elif output_format in ['gif', 'webp']:
                st.image(output_path, caption=f"Generated {output_format.upper()} Highlight Reel")

            with open(output_path, 'rb') as f:
                file_bytes = f.read()

            file_size_mb = len(file_bytes) / (1024 * 1024)
            mime_map = {
                'mp4': 'video/mp4',
                'mp4_loop': 'video/mp4',
                'webm': 'video/webm',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }

            st.download_button(
                label=f"⬇️ Download Highlight Reel ({file_size_mb:.1f} MB)",
                data=file_bytes,
                file_name=f"highlights.{output_format}",
                mime=mime_map.get(output_format, 'video/mp4'),
                use_container_width=True
            )
        elif st.session_state.processing_done:
            st.error("Output file was not generated. Check the processing logs above.")

st.markdown("---")
st.markdown("""
<div class="app-footer">
    <p>
        🎬 Smart Video Highlight & GIF Generator &nbsp;|&nbsp;
        Powered by OpenCV Optical Flow · librosa Audio Analysis · MoviePy &nbsp;|&nbsp;
        100% Local Processing, No APIs, No GPU Required
    </p>
</div>
""", unsafe_allow_html=True)
