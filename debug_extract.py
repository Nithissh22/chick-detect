import cv2
import torch
import os

model = torch.hub.load('./yolov5', 'custom', path='./models/chick_best.pt', source='local')
model.conf = 0.1 # Very low for debugging

cap = cv2.VideoCapture("chick_1.mp4")
ret, frame = cap.read()
if ret:
    print("Successfully read a frame from chick_1.mp4")
    results = model(frame)
    print(f"Detections: {results.pandas().xyxy[0]}")
    cv2.imwrite("debug_test.jpg", frame)
else:
    print("Failed to read a frame from chick_1.mp4")
cap.release()
