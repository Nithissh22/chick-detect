import cv2
import torch
import os

# Try different models
models = [
    './models/best.pt',
    './models/chick_best.pt',
    './models/yolov5s.pt'
]

cap = cv2.VideoCapture("chick_1.mp4")
ret, frame = cap.read()
if ret:
    for m_path in models:
        print(f"Testing model: {m_path}")
        try:
            if m_path == './models/yolov5s.pt' or 'yolov5s' in m_path:
                # Load from hub if it's the base model
                model = torch.hub.load('./yolov5', 'yolov5s', source='local')
            else:
                model = torch.hub.load('./yolov5', 'custom', path=m_path, source='local')
            
            model.conf = 0.05
            results = model(frame)
            print(f"Detections for {m_path}: {len(results.pandas().xyxy[0])}")
            if len(results.pandas().xyxy[0]) > 0:
                print(results.pandas().xyxy[0])
        except Exception as e:
            print(f"Failed to load/run {m_path}: {e}")
else:
    print("Failed to read a frame from chick_1.mp4")
cap.release()
