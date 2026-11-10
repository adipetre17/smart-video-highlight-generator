# Smart Video Highlight & GIF Generator

An AI-powered local web app that automatically extracts the most **interesting moments** from any video based on **visual action detection** and **audio climax analysis**, producing a highlight reel (MP4/WebM) or animated GIF/WebP.

**100% local processing** — no APIs, no GPU, no cloud required.

---

## Features

- **Visual Interest Scoring**: Uses Farneback dense optical flow to detect motion intensity, and HSV histogram correlation to detect scene changes.
- **Audio Interest Scoring**: Analyzes RMS energy (loudness), spectral centroid (brightness), and onset strength (transients) using librosa.
- **Intelligent Fusion**: Combines visual and audio scores with user-adjustable weights and percentile-based thresholding.
- **Memory Optimized**: Employs chunked processing (batches of 500 frames) to keep RAM usage constant, allowing long or HD videos to be processed smoothly on modest hardware (e.g., 8GB RAM).
- **Multiple Output Formats**: Export as MP4 (H.264), WebM (VP8), WebP animation, or GIF.
- **Social Media Ready**: Optional centered 9:16 vertical crop tailored for Instagram Reels, TikTok, and YouTube Shorts.
- **Interactive Timeline**: Visualize the interest score timeline with highlight regions highlighted.
- **Fully Configurable**: Adjust weights, thresholds, clip durations, and analysis precision via intuitive sliders.

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| UI | Streamlit | Web interface |
| Video I/O | OpenCV + MoviePy | Frame reading, editing, export |
| Visual Analysis | OpenCV | Optical flow, histogram comparison |
| Audio Analysis | librosa | RMS, spectral centroid, onset strength |
| Signal Processing & Math | NumPy, SciPy | Matrix operations, normalization, peak detection |
| Visualization | Matplotlib | Score timeline charts |

---

## Setup & Installation

### Prerequisites

- **Python 3.9+** (tested with 3.10, 3.11)
- **FFmpeg** must be installed and available in your system PATH
  - Windows: `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Install Dependencies

```bash
# Navigate to the project directory
cd path/to/project

# (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## How It Works

### Processing Pipeline

```text
Upload Video → Sample Frames → Extract Audio
                    │                  │
            Optical Flow +      RMS Energy +
            Histogram Diff      Spectral Centroid +
                    │           Onset Strength
                    │                  │
                    ▼                  ▼
             Visual Scores       Audio Scores
                    │                  │
                    └──────┬───────────┘
                           ▼
                    Score Fusion (weighted sum)
                           │
                    Percentile Thresholding
                           │
                    Segment Detection & Merging
                           │
                    Highlight Reel Export
                 (MP4 / WebM / WebP / GIF)
```

### Key Algorithms

1. **Farneback Dense Optical Flow** — Measures per-pixel motion between consecutive frames using polynomial expansion on image pyramids. The mean flow magnitude quantifies overall motion intensity.

2. **HSV Histogram Correlation** — Compares the 2D Hue-Saturation color distribution between consecutive frames. A drop in Pearson correlation indicates a scene change or dramatic visual shift.

3. **Audio Feature Fusion** — Combines three complementary signals:
   - RMS Energy: raw loudness envelope (~50% weight)
   - Spectral Centroid: frequency center-of-mass indicating brightness (~20%)
   - Onset Strength: spectral flux indicating transients and attacks (~30%)

4. **Percentile-Based Thresholding** — Adapts to each video's score distribution rather than using a fixed threshold. The 75th percentile selects the top 25% of moments.

5. **Memory Optimization (Chunked Processing)** — To prevent Out of Memory errors without relying on a GPU, the video processor yields frames in sequential chunks of 500. Each chunk is processed and immediately released from memory via Python's Garbage Collection, ensuring constant RAM usage regardless of the video's total length.

---

## Configuration Tips

| Scenario | Recommended Settings |
|----------|---------------------|
| Sports highlights | Visual weight: 0.7, Audio weight: 0.3, Threshold: 80 |
| Music video | Visual weight: 0.3, Audio weight: 0.7, Threshold: 70 |
| Vlog / talking head | Visual weight: 0.5, Audio weight: 0.5, Threshold: 75 |
| Action movie | Visual weight: 0.6, Audio weight: 0.4, Threshold: 80 |
| Long video (>10 min) | Analysis FPS: 5, Threshold: 85, Max duration: 60s |

---

## Limitations

- **Memory**: Very long videos (>30 minutes) at high analysis FPS may use significant RAM. Lower the analysis FPS for long videos.
- **Animation Size**: WebP and GIF outputs are inherently large. The app limits resolution to 480px width and 10 FPS for these formats.
- **No GPU Acceleration**: All processing runs on CPU. A 5-minute video takes ~30-60 seconds to process depending on your CPU.
- **Single Audio Channel**: Stereo audio is mixed to mono for analysis.

---

## License

This project is provided as a Proof of Concept for educational purposes.
All dependencies are open-source and free to use.