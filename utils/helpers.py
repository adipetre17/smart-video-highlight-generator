import os
import shutil
import tempfile

def create_temp_dir(prefix="video_highlight_"):
    return tempfile.mkdtemp(prefix=prefix)

def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:04.1f}"

def get_output_extension(output_format):
    extensions = {
        'mp4': '.mp4',
        'gif': '.gif',
        'webm': '.webm'
    }
    return extensions.get(output_format.lower(), '.mp4')
