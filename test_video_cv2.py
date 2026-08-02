import os
import cv2
import torch
import numpy as np
import supervision as sv

MODEL_PATH = './models/chick_best.pt'
model = torch.hub.load('./yolov5', 'custom', path=MODEL_PATH, source='local', force_reload=True)
model.conf = 0.15
model.iou = 0.45

input_path = "C:\\Users\\Nithissh\\Downloads\\Well_trained_chickens_tricks_smart_720p.mp4"
output_path = "detected_test.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Could not open video file")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video size: {width}x{height} @ {fps}fps")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

ret, frame = cap.read()
if ret:
    print(f"Successfully read first frame of shape {frame.shape}")
else:
    print("Failed to read first frame")

cap.release()
out.release()
print("Done")
