import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageOps

MODEL_PATH = './models/chick_best.pt'
VIDEO_PATH = 'C:\\Users\\Nithissh\\Downloads\\Well_trained_chickens_tricks_smart_720p.mp4'

def preprocess_like_roboflow(cv_img, size=(640, 640)):
    if cv_img is None:
        return None
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    img_pil = ImageOps.exif_transpose(img_pil)
    img_pil = img_pil.resize(size, Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
    return np.array(img_pil)

print("Loading model...")
model = torch.hub.load('./yolov5', 'custom', path=MODEL_PATH, source='local', force_reload=True)

# Test configs
configs = [
    {"conf": 0.15, "iou": 0.45, "max_det": 300}, # Original
    {"conf": 0.10, "iou": 0.60, "max_det": 1000}, # More sensitive, allows overlapping boxes, up to 1000 dets
    {"conf": 0.05, "iou": 0.70, "max_det": 1000}, # Extremely sensitive
]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Read frame 10 (giving video a moment to start)
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame")
    exit(1)

processed_rgb = preprocess_like_roboflow(frame)

print(f"\n=== Testing on frame 10 ===")
for cfg in configs:
    model.conf = cfg["conf"]
    model.iou = cfg["iou"]
    model.max_det = cfg["max_det"]
    
    results = model(processed_rgb)
    df = results.pandas().xyxy[0]
    
    # Run through supervision NMS like video_app does
    if df.empty:
        count = 0
    else:
        detections = sv.Detections(
            xyxy=df[['xmin', 'ymin', 'xmax', 'ymax']].values,
            confidence=df['confidence'].values,
            class_id=df['class'].values
        )
        detections = detections[detections.confidence > model.conf]
        detections = detections.with_nms(threshold=model.iou)
        count = len(detections)
        
    print(f"Config: {cfg} -> Total Detections after NMS: {count}")
    
    if not df.empty:
        print("  Raw detections from model (before supervision NMS):", len(df))
        
print("Done.")
