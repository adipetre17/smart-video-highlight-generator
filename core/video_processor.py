import cv2
import numpy as np
import tempfile
import os

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

def get_video_metadata(video_path):
    """Read basic video info (fps, resolution, duration) without decoding frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def sample_frames_chunked(video_path, analysis_fps=10.0, chunk_size=500,
                          progress_callback=None, max_analysis_height=360):
    """Yield frames in chunks to keep RAM usage bounded."""
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_interval = max(1, int(round(native_fps / analysis_fps)))
    need_resize = native_height > max_analysis_height

    if need_resize:
        scale = max_analysis_height / native_height
        target_w = int(native_width * scale)
        target_w = target_w - (target_w % 2)
        target_h = max_analysis_height - (max_analysis_height % 2)
        analysis_size = (target_w, target_h)

    frame_idx = 0
    prev_timestamp, prev_gray, prev_color = None, None, None

    while True:
        timestamps = []
        frames_gray = []
        frames_color = []

        if prev_timestamp is not None:
            timestamps.append(prev_timestamp)
            frames_gray.append(prev_gray)
            frames_color.append(prev_color)

        frames_sampled_in_chunk = 0

        while frames_sampled_in_chunk < chunk_size:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / native_fps

                if need_resize:
                    frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                timestamps.append(timestamp)
                frames_gray.append(gray)
                frames_color.append(frame)

                frames_sampled_in_chunk += 1

            if progress_callback and frame_idx % (frame_interval * 10) == 0:
                progress_callback(frame_idx, total_frames)

            frame_idx += 1

        if len(timestamps) < 2:
            break

        prev_timestamp = timestamps[-1]
        prev_gray = frames_gray[-1]
        prev_color = frames_color[-1]

        yield np.array(timestamps), frames_gray, frames_color

        if not success:
            break

    cap.release()

def extract_audio(video_path, output_dir=None, sr=22050):
    """Extract audio track from video as WAV for analysis."""
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    audio_path = os.path.join(output_dir, "extracted_audio.wav")
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        return None

    clip.audio.write_audiofile(
        audio_path, fps=sr, nbytes=2, codec='pcm_s16le', logger=None
    )

    clip.close()
    return audio_path
