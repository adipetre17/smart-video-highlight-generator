import numpy as np
import librosa

def compute_audio_scores(audio_path, video_duration, sr=22050, hop_length=512,
                         rms_weight=0.5, centroid_weight=0.2, onset_weight=0.3):
    """Compute combined audio interest score from a WAV file.
    Uses RMS energy, spectral centroid, and onset strength."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True, dtype=np.float32)

    if len(y) < hop_length * 2:
        return np.array([]), np.array([])

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    min_length = min(len(rms), len(spectral_centroid), len(onset_env))
    rms = rms[:min_length]
    spectral_centroid = spectral_centroid[:min_length]
    onset_env = onset_env[:min_length]

    def normalize_min_max(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min + 1e-10)

    rms_norm = normalize_min_max(rms)
    centroid_norm = normalize_min_max(spectral_centroid)
    onset_norm = normalize_min_max(onset_env)

    combined_scores = (
        rms_weight * rms_norm +
        centroid_weight * centroid_norm +
        onset_weight * onset_norm
    )

    audio_timestamps = librosa.frames_to_time(
        np.arange(min_length), sr=sr, hop_length=hop_length
    )

    return audio_timestamps, combined_scores
