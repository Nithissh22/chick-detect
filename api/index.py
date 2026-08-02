import os
import sys

# Add the parent directory to the path so it can find video_app and yolov5
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_app import app

# Vercel needs a specific variable named 'app' which is a WSGI application.
