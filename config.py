# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Video
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30
LIP_PADDING = 20          # pixels around lip bbox

# Audio
SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 3  # seconds per inference chunk
AUDIO_CHANNELS = 1
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Model
WHISPER_MODEL = "openai/whisper-tiny"   # CPU-friendly
DEVICE = "cpu"

# Mode: "audio", "video", "fusion"
MODE = "fusion"

# Display
FONT_SCALE = 0.7
TEXT_COLOR = (0, 255, 0)
BOX_COLOR = (0, 255, 255)