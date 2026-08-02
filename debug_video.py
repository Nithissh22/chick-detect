"""
Deep debug - tests different confidence thresholds and BGR vs RGB input.
"""
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = './models/chick_best.pt'
VIDEO_PATH = 'C:\\Users\\Nithissh\\Downloads\\Well_trained_chickens_tricks_smart_720p.mp4'

print("Loading model...")
model = torch.hub.load('./yolov5', 'custom', path=MODEL_PATH, source='local', force_reload=True)
model.conf = 0.01   # Very low confidence to see ANY detections
model.iou = 0.45
print(f"Model names: {model.names}")

cap = cv2.VideoCapture(VIDEO_PATH)
print(f"Video opened: {cap.isOpened()}")
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Cannot read frame from video")
    exit(1)

print(f"Frame shape (BGR from cv2): {frame.shape}")

# Test 1: BGR frame as-is (what our code does)
print("\n--- Test 1: BGR frame (cv2 default) at conf=0.01 ---")
r1 = model(frame)
df1 = r1.pandas().xyxy[0]
print(f"  Detections: {len(df1)}")
if not df1.empty:
    print(df1[['confidence','name']].to_string())

# Test 2: RGB frame (PIL-style)
print("\n--- Test 2: RGB frame (correct for YOLOv5) at conf=0.01 ---")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
r2 = model(frame_rgb)
df2 = r2.pandas().xyxy[0]
print(f"  Detections: {len(df2)}")
if not df2.empty:
    print(df2[['confidence','name']].to_string())

# Test 3: PIL image directly
print("\n--- Test 3: PIL Image at conf=0.01 ---")
pil_img = Image.fromarray(frame_rgb)
r3 = model(pil_img)
df3 = r3.pandas().xyxy[0]
print(f"  Detections: {len(df3)}")
if not df3.empty:
    print(df3[['confidence','name']].to_string())

# Test 4: Resize to 640x640 with PIL, pass as np array (RGB)
print("\n--- Test 4: Resized 640x640 RGB array at conf=0.01 ---")
resized = pil_img.resize((640,640), Image.BILINEAR)
arr = np.array(resized)
r4 = model(arr)
df4 = r4.pandas().xyxy[0]
print(f"  Detections: {len(df4)}")
if not df4.empty:
    print(df4[['confidence','name']].to_string())

# Test 5: what if we use the output.jpg that we know works?
print("\n--- Test 5: output.jpg (known-working image) at conf=0.01 ---")
img = cv2.imread('output.jpg')
if img is not None:
    r5 = model(img)
    df5 = r5.pandas().xyxy[0]
    print(f"  Detections: {len(df5)}")
    if not df5.empty:
        print(df5[['confidence','name']].head(5).to_string())
else:
    print("  output.jpg not found")

print("\n--- Done ---")
