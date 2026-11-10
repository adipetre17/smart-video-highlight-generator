# re-export main functions so app.py can do "from core import ..."
from .video_processor import get_video_metadata, sample_frames_chunked, extract_audio
from .visual_scorer import compute_visual_scores
from .audio_scorer import compute_audio_scores
from .fusion import fuse_and_detect_highlights
from .highlight_generator import create_highlight_reel
