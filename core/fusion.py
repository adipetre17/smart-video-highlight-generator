import numpy as np
from scipy.signal import find_peaks

def normalize_min_max(scores):
    """Min-max normalize an array to [0, 1]."""
    if len(scores) == 0:
        return scores
    s_min = scores.min()
    s_max = scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-10)


def align_to_common_timeline(visual_timestamps, visual_scores,
                             audio_timestamps, audio_scores, common_fps=10.0):
    """Interpolate visual and audio scores onto a shared time axis."""
    all_timestamps = []
    if len(visual_timestamps) > 0:
        all_timestamps.extend([visual_timestamps[0], visual_timestamps[-1]])
    if len(audio_timestamps) > 0:
        all_timestamps.extend([audio_timestamps[0], audio_timestamps[-1]])

    if len(all_timestamps) == 0:
        return np.array([]), np.array([]), np.array([])

    t_start = min(all_timestamps)
    t_end = max(all_timestamps)

    common_timestamps = np.arange(t_start, t_end, 1.0 / common_fps)

    if len(visual_timestamps) > 0 and len(visual_scores) > 0:
        visual_interp = np.interp(common_timestamps, visual_timestamps, visual_scores)
    else:
        visual_interp = np.zeros(len(common_timestamps))

    if len(audio_timestamps) > 0 and len(audio_scores) > 0:
        audio_interp = np.interp(common_timestamps, audio_timestamps, audio_scores)
    else:
        audio_interp = np.zeros(len(common_timestamps))

    return common_timestamps, visual_interp, audio_interp


def merge_overlapping_segments(segments):
    """Merge adjacent or overlapping time segments."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0]]

    for current_start, current_end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]

        if current_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def fuse_and_detect_highlights(visual_timestamps, visual_scores,
                               audio_timestamps, audio_scores,
                               visual_weight=0.5, audio_weight=0.5,
                               threshold_percentile=75.0,
                               min_clip_duration=2.0,
                               max_total_duration=30.0,
                               peak_padding=1.5,
                               video_duration=None,
                               extract_single_peak=False):
    """Fuse visual + audio scores and extract highlight segments above the threshold."""
    visual_norm = normalize_min_max(visual_scores)
    audio_norm = normalize_min_max(audio_scores)

    common_timestamps, visual_aligned, audio_aligned = align_to_common_timeline(
        visual_timestamps, visual_norm,
        audio_timestamps, audio_norm,
        common_fps=10.0
    )

    if len(common_timestamps) == 0:
        return [], {
            'common_timestamps': np.array([]),
            'visual_aligned': np.array([]),
            'audio_aligned': np.array([]),
            'fused_scores': np.array([]),
            'threshold_value': 0.0
        }

    weight_sum = visual_weight + audio_weight
    if weight_sum > 0:
        w_v = visual_weight / weight_sum
        w_a = audio_weight / weight_sum
    else:
        w_v, w_a = 0.5, 0.5

    fused_scores = w_v * visual_aligned + w_a * audio_aligned

    # single peak mode (for looping GIF/WebP exports)
    if extract_single_peak:
        if len(fused_scores) > 0:
            peak_idx = np.argmax(fused_scores)
            peak_time = common_timestamps[peak_idx]

            half_duration = min_clip_duration / 2.0
            seg_start = max(0.0, peak_time - half_duration)
            max_time = video_duration if video_duration else common_timestamps[-1]
            seg_end = min(max_time, peak_time + half_duration)

            final_segments = [(seg_start, seg_end)]
            threshold_value = fused_scores[peak_idx]
        else:
            final_segments = []
            threshold_value = 0.0

        fused_timeline = {
            'common_timestamps': common_timestamps,
            'visual_aligned': visual_aligned,
            'audio_aligned': audio_aligned,
            'fused_scores': fused_scores,
            'threshold_value': threshold_value
        }
        return final_segments, fused_timeline

    threshold_value = np.percentile(fused_scores, threshold_percentile)
    above_threshold = fused_scores >= threshold_value

    diff = np.diff(above_threshold.astype(int))
    rising_edges = np.where(diff == 1)[0] + 1
    falling_edges = np.where(diff == -1)[0] + 1

    if above_threshold[0]:
        rising_edges = np.concatenate(([0], rising_edges))
    if above_threshold[-1]:
        falling_edges = np.concatenate((falling_edges, [len(above_threshold)]))

    raw_segments = []
    for start_idx, end_idx in zip(rising_edges, falling_edges):
        seg_start = common_timestamps[start_idx]
        seg_end = common_timestamps[min(end_idx, len(common_timestamps) - 1)]
        raw_segments.append((seg_start, seg_end))

    max_time = video_duration if video_duration else (
        common_timestamps[-1] if len(common_timestamps) > 0 else 0
    )

    padded_segments = []
    for start, end in raw_segments:
        start = max(0.0, start - peak_padding)
        end = min(max_time, end + peak_padding)

        duration = end - start
        if duration < min_clip_duration:
            pad = (min_clip_duration - duration) / 2.0
            start = max(0.0, start - pad)
            end = min(max_time, end + pad)
        padded_segments.append((start, end))

    merged_segments = merge_overlapping_segments(padded_segments)

    # keep only the best segments that fit within max_total_duration
    if max_total_duration > 0 and len(merged_segments) > 0:
        scored_segments = []
        for start, end in merged_segments:
            mask = (common_timestamps >= start) & (common_timestamps <= end)
            if np.any(mask):
                avg_score = np.mean(fused_scores[mask])
            else:
                avg_score = 0.0
            scored_segments.append((start, end, avg_score))

        scored_segments.sort(key=lambda s: s[2], reverse=True)

        selected = []
        total_duration = 0.0
        for start, end, score in scored_segments:
            seg_duration = end - start
            if total_duration + seg_duration <= max_total_duration:
                selected.append((start, end))
                total_duration += seg_duration

        selected.sort(key=lambda s: s[0])
        final_segments = selected
    else:
        final_segments = merged_segments

    fused_timeline = {
        'common_timestamps': common_timestamps,
        'visual_aligned': visual_aligned,
        'audio_aligned': audio_aligned,
        'fused_scores': fused_scores,
        'threshold_value': threshold_value
    }

    return final_segments, fused_timeline
