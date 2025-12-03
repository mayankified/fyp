# config.py
import os

# Set to True ONLY when running on your powerful PC
USE_AI = False  

# Server settings
BROADCAST_ROOM = 'live_stream'
ASYNC_MODE = 'threading'  # 'threading' is safer for testing than 'eventlet'

# AI Settings
MODEL_PATH = "best.pt"
CONF_THRESH = 0.25
PROCESS_FPS = 30
JPEG_QUALITY = 85