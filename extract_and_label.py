import cv2
import torch
import os
import shutil
from pathlib import Path

# Paths
VIDEO_PATHS = [
    "chick_1.mp4",
    "chick_2.mp4"
]
OUTPUT_DIR = r"e:\Chickdetect\Chickdetect\data_training"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Load models
# best.pt is good at finding 'chicken' (class 0)
# chick_best.pt is good at classifying 'Healthy' (0) vs 'Sick' (1)
model_find = torch.hub.load('./yolov5', 'custom', path='./models/best.pt', source='local')
model_class = torch.hub.load('./yolov5', 'custom', path='./models/chick_best.pt', source='local')

model_find.conf = 0.3
model_class.conf = 0.2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_find.to(DEVICE)
model_class.to(DEVICE)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def extract_and_label():
    frame_idx = 0
    for v_path in VIDEO_PATHS:
        print(f"Processing {v_path}...")
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"Failed to open {v_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps * 0.5))  # Extract every 0.5 seconds
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % interval == 0:
                frame_name = f"frame_{frame_idx:05d}.jpg"
                frame_path = os.path.join(IMAGES_DIR, frame_name)
                
                # 1. Find all chickens
                res_find = model_find(frame).pandas().xyxy[0]
                # 2. Try to classify them
                res_class = model_class(frame).pandas().xyxy[0]
                
                if not res_find.empty:
                    # Save image
                    cv2.imwrite(frame_path, frame)
                    label_path = os.path.join(LABELS_DIR, f"frame_{frame_idx:05d}.txt")
                    h, w, _ = frame.shape
                    
                    with open(label_path, 'w') as f:
                        for _, row in res_find.iterrows():
                            # Default to Healthy (0)
                            final_class = 0 
                            
                            # Check for classification overlap
                            for _, crow in res_class.iterrows():
                                if iou([row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                                       [crow['xmin'], crow['ymin'], crow['xmax'], crow['ymax']]) > 0.4:
                                    final_class = int(crow['class'])
                                    break
                            
                            x_center = (row['xmin'] + row['xmax']) / 2 / w
                            y_center = (row['ymin'] + row['ymax']) / 2 / h
                            width = (row['xmax'] - row['xmin']) / w
                            height = (row['ymax'] - row['ymin']) / h
                            
                            f.write(f"{final_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    frame_idx += 1
                    print(f"Saved {frame_name}")
                
            count += 1
            
        cap.release()
    print(f"Done! Extracted {frame_idx} labeled frames.")

if __name__ == "__main__":
    extract_and_label()
