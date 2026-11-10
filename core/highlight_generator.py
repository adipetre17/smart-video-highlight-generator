import os
import numpy as np

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips


def apply_vertical_crop(clip):
    """Crop clip to 9:16 portrait for social media."""
    target_ratio = 9.0 / 16.0
    current_ratio = clip.w / clip.h

    if current_ratio > target_ratio:
        new_width = int(clip.h * target_ratio)
        new_width = new_width - (new_width % 2)
        x_center = clip.w // 2
        x1 = x_center - new_width // 2
        x2 = x_center + new_width // 2
        return clip.crop(x1=x1, x2=x2)

    elif current_ratio < target_ratio:
        new_height = int(clip.w / target_ratio)
        new_height = new_height - (new_height % 2)
        y_center = clip.h // 2
        y1 = y_center - new_height // 2
        y2 = y_center + new_height // 2
        return clip.crop(y1=y1, y2=y2)

    else:
        return clip


def _make_fade_filter(clip_duration, fade_duration, fade_type='in'):
    """Create a uint8 fade effect (avoids moviepy's default float64 which eats RAM)."""
    def fade_filter(get_frame, t):
        frame = get_frame(t)

        if fade_type == 'in':
            if t < fade_duration:
                factor = t / fade_duration
            else:
                return frame
        else:
            time_remaining = clip_duration - t
            if time_remaining < fade_duration:
                factor = time_remaining / fade_duration
            else:
                return frame

        faded = np.clip(frame.astype(np.float32) * factor, 0, 255)
        return faded.astype(np.uint8)

    return fade_filter


def create_highlight_reel(video_path, segments, output_format='mp4',
                          vertical_crop=False, output_dir=None,
                          transition_duration=0.3, gif_fps=10,
                          gif_max_width=480, max_resolution=720):
    """Cut highlight segments from the video, apply fades, concatenate and export."""
    if not segments:
        return None

    if output_dir is None:
        import tempfile
        output_dir = tempfile.gettempdir()

    extension_map = {
        'mp4': '.mp4', 'mp4_loop': '.mp4', 'webm': '.webm',
        'gif': '.gif', 'webp': '.webp'
    }
    ext = extension_map.get(output_format.lower(), '.mp4')
    output_path = os.path.join(output_dir, f"highlight_reel{ext}")

    full_clip = VideoFileClip(video_path)

    # downscale if needed
    if full_clip.h > max_resolution:
        scale_factor = max_resolution / full_clip.h
        new_width = int(full_clip.w * scale_factor)
        new_width = new_width - (new_width % 2)
        new_height = max_resolution - (max_resolution % 2)
        full_clip = full_clip.resize((new_width, new_height))

    subclips = []
    for start, end in segments:
        start = max(0, start)
        end = min(end, full_clip.duration)

        if end - start < 0.1:
            continue

        subclip = full_clip.subclip(start, end)
        clip_duration = end - start

        if transition_duration > 0 and clip_duration > transition_duration * 2:
            try:
                fade_in_fn = _make_fade_filter(clip_duration, transition_duration, 'in')
                subclip = subclip.fl(fade_in_fn)
                fade_out_fn = _make_fade_filter(clip_duration, transition_duration, 'out')
                subclip = subclip.fl(fade_out_fn)
            except Exception:
                pass

        subclips.append(subclip)

    if not subclips:
        full_clip.close()
        return None

    final_clip = concatenate_videoclips(subclips)

    if vertical_crop:
        final_clip = apply_vertical_crop(final_clip)

    try:
        if output_format.lower() == 'mp4':
            final_clip.write_videofile(
                output_path, codec='libx264', audio_codec='aac',
                preset='ultrafast', threads=2, logger=None
            )
        elif output_format.lower() == 'mp4_loop':
            final_clip.write_videofile(
                output_path, codec='libx264', audio=False,
                preset='ultrafast', threads=2, logger=None
            )
        elif output_format.lower() == 'webm':
            final_clip.write_videofile(
                output_path, codec='libvpx', audio_codec='libvorbis',
                threads=2, logger=None
            )
        elif output_format.lower() in ['gif', 'webp']:
            anim_clip = final_clip
            if anim_clip.w > gif_max_width:
                anim_clip = anim_clip.resize(width=gif_max_width)
            anim_clip.write_gif(output_path, fps=gif_fps, opt={'loop': 0}, logger=None)

    except Exception as e:
        full_clip.close()
        raise RuntimeError(f"Failed to export highlight reel: {str(e)}")

    full_clip.close()
    return output_path
