"""
Detailed Chick Analysis Script
- Processes EVERY frame (no skipping)
- Uses SAHI-style tiled inference for small/dense chick detection
- Full labels with individual tracking IDs, class names, confidence
- Rich stats HUD overlay per frame
"""
import os
import sys
import cv2
import torch
import time
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

# ── Settings ──────────────────────────────────────────────────────────
INPUT_VIDEO  = r"E:\Downloads\vidssave.com Million Colors Cute Tweeties 🥹💕 Colored Hen Babies #shorts #viralvideo #chicks 720P.mp4"
OUTPUT_VIDEO = "detected_chicks_detailed.mp4"
MODEL_PATH   = './models/chick_best.pt'
YOLOV5_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')

# ── Load Model ────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')
model.to(device)
if device.type != 'cpu':
    model.half()

model.conf    = 0.15          # Lower threshold to catch every chick
model.iou     = 0.45
model.max_det = 1000          # Support large numbers of chicks
if isinstance(model.names, list):
    model.names = {i: n for i, n in enumerate(model.names)}

print(f"✅ Model loaded on {device}  |  Classes: {model.names}")

COLOR_MAP = {
    'Healthy':             (0, 255, 0),
    'Sick':                (0, 0, 255),
    'Coccidiosis':         (0, 165, 255),
    'New Castle Disease':  (0, 0, 255),
    'Salmonella':          (255, 0, 255),
    'chicken':             (255, 255, 0),
}


# ── SAHI-style Tiled Inference ────────────────────────────────────────
def get_tiled_detections(img_rgb, model, tile_size=(640, 640), overlap=0.25):
    """
    Slicing Aided Hyper Inference – splits the image into overlapping
    tiles, runs batch inference, then merges + NMS.
    """
    h, w, _ = img_rgb.shape
    th, tw  = tile_size
    stride_h = int(th * (1 - overlap))
    stride_w = int(tw * (1 - overlap))

    tiles, tile_coords = [], []

    # Global view (resized to tile_size for context)
    tiles.append(cv2.resize(img_rgb, tile_size))
    tile_coords.append((0, 0, w / tw, h / th))

    # Local tiles
    for y in range(0, h - stride_h + 1, stride_h):
        for x in range(0, w - stride_w + 1, stride_w):
            y2 = min(y + th, h); x2 = min(x + tw, w)
            y1 = max(0, y2 - th); x1 = max(0, x2 - tw)
            tiles.append(img_rgb[y1:y2, x1:x2])
            tile_coords.append((x1, y1, 1.0, 1.0))

    if not tiles:
        return sv.Detections.empty()

    results = model(tiles)

    all_xyxy, all_conf, all_cls = [], [], []
    for i, df in enumerate(results.pandas().xyxy):
        if df.empty:
            continue
        ox, oy, sx, sy = tile_coords[i]
        df['xmin'] = df['xmin'] * sx + ox
        df['xmax'] = df['xmax'] * sx + ox
        df['ymin'] = df['ymin'] * sy + oy
        df['ymax'] = df['ymax'] * sy + oy
        all_xyxy.append(df[['xmin', 'ymin', 'xmax', 'ymax']].values)
        all_conf.append(df['confidence'].values)
        all_cls.append(df['class'].values)

    if not all_xyxy:
        return sv.Detections.empty()

    dets = sv.Detections(
        xyxy       = np.concatenate(all_xyxy),
        confidence = np.concatenate(all_conf),
        class_id   = np.concatenate(all_cls),
    )
    return dets.with_nms(threshold=model.iou)


# ── Open Video ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"❌ Cannot open: {INPUT_VIDEO}")
    sys.exit(1)

cap_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Keep original resolution for detail
width, height = cap_w, cap_h
print(f"📹 Video: {cap_w}x{cap_h} @ {fps:.1f} fps  |  ~{total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    print("❌ VideoWriter failed"); sys.exit(1)

tracker = sv.ByteTrack()

# Cumulative tracking sets
all_ids      = {}          # track_id → last seen class name
frame_count  = 0
start_time   = time.time()

print("🔍 Processing every frame with tiled inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Convert BGR → RGB for inference
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tiled inference for dense small-chick detection
    detections = get_tiled_detections(rgb, model)

    if len(detections) == 0:
        # No detections – write original frame with progress bar only
        out.write(frame)
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames}  |  0 detections")
        continue

    detections = detections[detections.confidence > model.conf]
    tracks     = tracker.update_with_detections(detections)

    # ── Annotate ──────────────────────────────────────────────────
    annotated = frame.copy()
    class_names = model.names
    frame_counts = {}   # per-class count this frame

    for box, conf, tid, cid in zip(
        tracks.xyxy, tracks.confidence, tracks.tracker_id, tracks.class_id
    ):
        x1, y1, x2, y2 = map(int, box)
        label_name = class_names[int(cid)]
        color      = COLOR_MAP.get(label_name, (200, 200, 200))

        frame_counts[label_name] = frame_counts.get(label_name, 0) + 1
        all_ids[tid] = label_name

        # Full label: ClassName #ID  Conf%
        label_text = f"{label_name} #{tid} {conf*100:.0f}%"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw_lbl, th_lbl), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1 - 4, th_lbl + 6)
        cv2.rectangle(annotated,
                      (x1, ly - th_lbl - baseline - 2),
                      (x1 + tw_lbl + 4, ly + baseline),
                      color, -1)
        text_color = (0, 0, 0) if label_name == 'Healthy' else (255, 255, 255)
        cv2.putText(annotated, label_text, (x1 + 2, ly - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # ── Stats HUD (semi-transparent) ─────────────────────────────
    n_this_frame = sum(frame_counts.values())
    n_unique     = len(all_ids)

    hud_lines = [f"Frame {frame_count}/{total_frames}",
                 f"This frame: {n_this_frame}",
                 f"Unique tracked: {n_unique}",
                 "─" * 22]
    for cls, cnt in sorted(frame_counts.items()):
        hud_lines.append(f"  {cls}: {cnt}")

    line_h   = 22
    hud_w    = 280
    hud_h    = 12 + line_h * len(hud_lines)
    ox, oy   = width - hud_w - 12, 12

    overlay = annotated.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + hud_w, oy + hud_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)

    for i, line in enumerate(hud_lines):
        col = (220, 220, 220)
        # Colour-code class lines
        for cls_name, cls_col in COLOR_MAP.items():
            if cls_name in line:
                col = cls_col
                break
        cv2.putText(annotated, line,
                    (ox + 8, oy + 18 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    out.write(annotated)

    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_now = frame_count / elapsed if elapsed > 0 else 0
        print(f"  Frame {frame_count}/{total_frames}  |  "
              f"{n_this_frame} this frame  |  "
              f"{n_unique} unique  |  {fps_now:.1f} FPS")

cap.release()
out.release()

elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"✅ Done!  Processed {frame_count} frames in {elapsed:.1f}s  "
      f"({frame_count/elapsed:.1f} FPS)")
print(f"   Unique chicks tracked: {len(all_ids)}")
print(f"   Output saved to: {OUTPUT_VIDEO}")

# Summary breakdown
summary = {}
for tid, cls in all_ids.items():
    summary[cls] = summary.get(cls, 0) + 1
print(f"\n📊 Tracking Summary:")
for cls, cnt in sorted(summary.items()):
    print(f"   {cls}: {cnt} unique individuals")
print(f"   TOTAL: {len(all_ids)}")
