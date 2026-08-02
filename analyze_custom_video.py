import os
import sys
import cv2
import torch
import time
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

# Settings
INPUT_VIDEO = r"E:\Downloads\vidssave.com Chicken clucking loudly after laying her egg #chicken 1080P.mp4"
OUTPUT_VIDEO = "detected_custom_video.mp4"
MODEL_PATH = './models/chick_best.pt'
FRAME_STRIDE = 1

# Load Model
print(f"Loading model from {MODEL_PATH}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5'), 'custom', path=MODEL_PATH, source='local')
model.to(device)
if device.type != 'cpu':
    model.half()

model.conf = 0.20
model.iou = 0.45
if isinstance(model.names, list):
    model.names = {i: n for i, n in enumerate(model.names)}

print("Model loaded successfully.")

COLOR_MAP = {
    'Healthy': (0, 255, 0),
    'Sick': (0, 0, 255),
    'Coccidiosis': (0, 165, 255),
    'New Castle Disease': (0, 0, 255),
    'Salmonella': (255, 0, 255),
    'chicken': (255, 255, 0)
}

tracker = sv.ByteTrack()

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"❌ Could not open video file: {INPUT_VIDEO}")
    sys.exit(1)

cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

max_dim = 1280
if cap_width > max_dim or cap_height > max_dim:
    scale = max_dim / max(cap_width, cap_height)
    width, height = int(cap_width * scale), int(cap_height * scale)
else:
    width, height = cap_width, cap_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % FRAME_STRIDE != 0:
        out.write(cv2.resize(frame, (width, height)))
        continue

    # Convert to RGB
    full_res_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(full_res_rgb, size=1280)
    detections = sv.Detections.from_yolov5(results)
    
    if len(detections) == 0:
        out.write(cv2.resize(frame, (width, height)))
        continue

    detections = detections[detections.confidence > model.conf]
    tracks = tracker.update_with_detections(detections)
    
    annotated_frame = cv2.resize(frame, (width, height)) if width != cap_width else frame.copy()
    sx = width / float(cap_width)
    sy = height / float(cap_height)

    for box, conf, track_id, class_id in zip(tracks.xyxy, tracks.confidence, tracks.tracker_id, tracks.class_id):
        x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
        label_name = model.names[int(class_id)]
        color = COLOR_MAP.get(label_name, (200, 200, 200))
        label = f"{label_name[:1]}{track_id} {conf:.2f}"
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out.write(annotated_frame)

cap.release()
out.release()
print(f"✅ Video Processing Complete. Saved to {OUTPUT_VIDEO}.")
