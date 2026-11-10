import cv2
import numpy as np

def compute_optical_flow_scores(frames_gray):
    """Compute motion scores using Farneback dense optical flow between consecutive frames."""
    if len(frames_gray) < 2:
        return np.array([])

    h, w = frames_gray[0].shape[:2]
    frame_diagonal = np.sqrt(w**2 + h**2)

    flow_scores = []

    for i in range(1, len(frames_gray)):
        prev_gray = frames_gray[i - 1]
        curr_gray = frames_gray[i]

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_magnitude = np.mean(magnitude) / frame_diagonal

        flow_scores.append(mean_magnitude)

    return np.array(flow_scores)


def compute_scene_change_scores(frames_color):
    """Detect scene changes by comparing HSV color histograms between frames."""
    if len(frames_color) < 2:
        return np.array([])

    def compute_hs_histogram(bgr_frame):
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist

    prev_hist = compute_hs_histogram(frames_color[0])
    scene_scores = []

    for i in range(1, len(frames_color)):
        curr_hist = compute_hs_histogram(frames_color[i])
        correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        change_score = max(0.0, 1.0 - correlation)
        scene_scores.append(change_score)
        prev_hist = curr_hist

    return np.array(scene_scores)


def compute_visual_scores(timestamps, frames_gray, frames_color,
                          flow_weight=0.7, scene_weight=0.3):
    """Combine optical flow and scene change scores into a single visual interest score."""
    flow_scores = compute_optical_flow_scores(frames_gray)
    scene_scores = compute_scene_change_scores(frames_color)

    if len(flow_scores) == 0 or len(scene_scores) == 0:
        return np.array([]), np.array([])

    flow_min, flow_max = flow_scores.min(), flow_scores.max()
    flow_normalized = (flow_scores - flow_min) / (flow_max - flow_min + 1e-10)

    scene_min, scene_max = scene_scores.min(), scene_scores.max()
    scene_normalized = (scene_scores - scene_min) / (scene_max - scene_min + 1e-10)

    combined_scores = (
        flow_weight * flow_normalized +
        scene_weight * scene_normalized
    )

    visual_timestamps = (timestamps[:-1] + timestamps[1:]) / 2.0

    return visual_timestamps, combined_scores
