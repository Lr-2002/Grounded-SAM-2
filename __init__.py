import sys 
import os

print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .video_processor import VideoProcessor
from .online_process_two_videos import MaskGenerator