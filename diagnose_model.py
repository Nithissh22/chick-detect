"""
Diagnose why the model always predicts 'Sick'.
Extracts a frame from the video and shows raw prediction details.
"""
import os, sys, cv2, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

MODEL_PATH = './models/chick_best.pt'
YOLOV5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
VIDEO = r"E:\Downloads\vidssave.com Million Colors Cute Tweeties 🥹💕 Colored Hen Babies #shorts #viralvideo #chicks 720P.mp4"

model = torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.01   # VERY LOW to see ALL predictions
model.iou  = 0.45
print(f"Model classes: {model.names}")

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video has {total} frames")

# Sample 5 frames evenly spread across the video
sample_indices = [int(total * i / 6) for i in range(1, 6)]

for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=640)
    df = results.pandas().xyxy[0]
    
    print(f"\n{'='*60}")
    print(f"FRAME {idx}/{total}  |  Shape: {frame.shape}")
    print(f"Total detections (conf>0.01): {len(df)}")
    
    if df.empty:
        print("  No detections at all.")
        continue
    
    # Show class distribution at different thresholds
    for thresh in [0.01, 0.10, 0.15, 0.20, 0.30, 0.50]:
        sub = df[df['confidence'] >= thresh]
        if sub.empty:
            print(f"  conf >= {thresh:.2f}: 0 detections")
            continue
        class_counts = sub['name'].value_counts().to_dict()
        print(f"  conf >= {thresh:.2f}: {len(sub)} detections  →  {class_counts}")
    
    # Show top 10 individual predictions
    print(f"\n  Top 10 predictions (sorted by confidence):")
    top = df.nlargest(10, 'confidence')
    for _, row in top.iterrows():
        print(f"    class={row['name']}  conf={row['confidence']:.4f}  "
              f"box=[{row['xmin']:.0f},{row['ymin']:.0f},{row['xmax']:.0f},{row['ymax']:.0f}]")

cap.release()
print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
