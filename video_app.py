import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageOps
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import subprocess
import time 
import pandas as pd
from supervision.tracker.byte_tracker.core import ByteTrack
from yolo_cam import generate_cam_heatmap
from flask import Response
import json
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = 'scan_history.json'

def log_scan(filename, is_video, total_count, unhealthy_count, primary_diagnosis, final_url):
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try:
                    history = json.load(f)
                except:
                    pass
        
        entry = {
            'id': str(time.time()),
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'type': 'video' if is_video else 'image',
            'total_count': total_count,
            'unhealthy_count': unhealthy_count,
            'primary_diagnosis': primary_diagnosis,
            'url': final_url
        }
        history.insert(0, entry) # prepend
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history[:50], f) # Keep last 50
    except Exception as e:
        print(f"Error logging scan: {e}")

# Color mapping for diseases
COLOR_MAP = {
    'Healthy': (0, 255, 0),
    'Sick': (0, 0, 255),            # Blue/Red
    'Coccidiosis': (0, 165, 255),    # Orange
    'New Castle Disease': (0, 0, 255), # Red
    'Salmonella': (255, 0, 255),      # Magenta
    'chicken': (255, 255, 0)          # Yellow/Cyan
}

# DISEASE KNOWLEDGE BASE
DISEASE_INFO = {
    'Coccidiosis': {
        'symptoms': 'Bloody droppings, ruffled feathers, huddling, and weight loss.',
        'prevention': 'Maintain dry litter, use coccidiostats in feed, and ensure clean water.',
        'urgency': 'High'
    },
    'New Castle Disease': {
        'symptoms': 'Twisted necks, gasping, greenish diarrhea, and sudden drop in egg production.',
        'prevention': 'Strict biosecurity and systematic vaccination programs.',
        'urgency': 'Critical'
    },
    'Salmonella': {
        'symptoms': 'Weakness, loss of appetite, and chalky white diarrhea.',
        'prevention': 'Rodent control, clean hatchery conditions, and antibiotic treatment if prescribed.',
        'urgency': 'Medium'
    },
    'Sick': {
        'symptoms': 'General symptoms of illness detected. Potential diagnoses include Coccidiosis, New Castle Disease, or Salmonella. Monitor for specific signs like Bloody droppings or twisted necks.',
        'prevention': 'Isolate affected birds immediately. Ensure strict biosecurity and consult a veterinarian for specific treatment.',
        'urgency': 'Action Required'
    },
    'Healthy': {
        'symptoms': 'Normal active behavior, bright eyes, and healthy appetite.',
        'prevention': 'Continue standard biosecurity and balanced nutrition.',
        'urgency': 'None'
    }
}

UPLOAD_FOLDER = 'uploads'
DETECTION_OUTPUT_FOLDER =  'static/detections'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_OUTPUT_FOLDER'] = DETECTION_OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_FOLDER, exist_ok=True)

# OPTIMIZATION SETTINGS
FRAME_STRIDE = 2        # Process every 2nd frame (boosts FPS by 2x)
USE_TILING_VIDEO = False # Disable tiling for video to boost FPS (3-5x speedup)

try:
    MODEL_PATH = './models/chick_best.pt'
    YOLOV5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    model = torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')
    # ACCURACY & SPEED: Optimize model settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if device.type != 'cpu':
        model.half()  # Speed: Use FP16 if on GPU
    
    model.conf = 0.35      # Increased for better accuracy and fewer false positives
    model.iou = 0.50       # Tuned IOU for better bounding box accuracy
    model.max_det = 1500   # Support large numbers of chicks
    # Normalize model.names to a dict for consistent access
    if isinstance(model.names, list):
        model.names = {i: n for i, n in enumerate(model.names)}
    print(f"Model loaded successfully on {device}")
    print(f"Model Names: {model.names}")
    
    # SYSTEM CHECK: Ensure ffmpeg is installed for video processing
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg found: Video re-encoding enabled.")
    except Exception:
        print("⚠️ FFmpeg NOT FOUND. Video previews may not play in some browsers.")
        print("   To fix this, install FFmpeg and add it to your PATH.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error loading model: {e}")
    model = None


def fix_video_metadata(input_path, final_path):
    """Re-encode using hardware acceleration if available, otherwise fast CPU."""
    try:
        # Try NVENC first for NVIDIA GPUs, fallback to libx264
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '24',
            '-c:a', 'aac', '-movflags', '+faststart', '-pix_fmt', 'yuv420p', final_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            # Fallback to libx264
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
                '-c:a', 'aac', '-movflags', '+faststart', '-pix_fmt', 'yuv420p', final_path
            ]
            subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg re-encode failed: {e}")
        return False


def fast_preprocess(cv_img, size=(640, 640)):
    """Fast OpenCV based transformation for real-time video."""
    if cv_img is None: return None
    resized = cv2.resize(cv_img, size)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def fast_to_rgb(cv_img):
    """Converts BGR to RGB without resizing."""
    if cv_img is None: return None
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_tiled_detections(img, model, tile_size=(640, 640), overlap=0.25, augment=False):
    """
    Slicing Aided Hyper Inference (SAHI) style tiling for high density.
    Optimized with Batch Inference for 3-5x speedup.
    """
    h, w, _ = img.shape
    th, tw = tile_size
    
    # Grid calculation
    stride_h = int(th * (1 - overlap))
    stride_w = int(tw * (1 - overlap))
    
    all_xyxy = []
    all_conf = []
    all_class = []
    
    # Prepare batch
    tiles = []
    tile_coords = []
    
    # 1. Add Global image for context (resized)
    tiles.append(cv2.resize(img, tile_size))
    tile_coords.append((0, 0, w/tw, h/th)) # Scale factors for global
    
    # 2. Collect tiles
    for y in range(0, h - stride_h + 1, stride_h):
        for x in range(0, w - stride_w + 1, stride_w):
            y2 = min(y + th, h)
            x2 = min(x + tw, w)
            y1 = max(0, y2 - th)
            x1 = max(0, x2 - tw)
            
            tiles.append(img[y1:y2, x1:x2])
            tile_coords.append((x1, y1, 1.0, 1.0)) # No scaling for tiles, just offset
                
    if not tiles:
        return sv.Detections.empty()

    # BATCH INFERENCE: The most critical optimization
    # YOLOv5 handles a list of images efficiently
    results = model(tiles, augment=augment)
    
    # Process batch results
    # results.pandas().xyxy is a list of DataFrames for each image in batch
    for i, df in enumerate(results.pandas().xyxy):
        if df.empty: continue
        
        ox, oy, sx, sy = tile_coords[i]
        
        # Scale and Offset
        df['xmin'] = (df['xmin'] * sx) + ox
        df['xmax'] = (df['xmax'] * sx) + ox
        df['ymin'] = (df['ymin'] * sy) + oy
        df['ymax'] = (df['ymax'] * sy) + oy
        
        all_xyxy.append(df[['xmin', 'ymin', 'xmax', 'ymax']].values)
        all_conf.append(df['confidence'].values)
        all_class.append(df['class'].values)

    if not all_xyxy:
        return sv.Detections.empty()
        
    detections = sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_conf),
        class_id=np.concatenate(all_class)
    )
    
    # Deduplicate overlapping detections from different tiles
    return detections.with_nms(threshold=model.iou)


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/detect', methods=['POST'])
def detect_image():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    file = request.files.get('file')
    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            return jsonify({'error': 'Could not read image file'}), 400

        # Use Full-Res RGB for Tiling to capture details
        full_res_rgb = fast_to_rgb(img_cv)
        if full_res_rgb is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        # Get target class for CAM (default to first detected sick class if not provided)
        target_cam_class_id = request.form.get('target_class_id')
        if target_cam_class_id is not None:
            target_cam_class_id = int(target_cam_class_id)
        
        # Use Tiled Inference for Image too to improve accuracy on small/close subjects
        # Increased overlap to 0.5 and enabled augment=True for maximum accuracy
        detections = get_tiled_detections(full_res_rgb, model, overlap=0.5, augment=True)
        print(f"DEBUG: Tiled Detections count: {len(detections)}")

        # The 'detections' object is already created by get_tiled_detections
        detections = detections[detections.confidence > model.conf]
        detections = detections.with_nms(threshold=model.iou)

        # Dynamic labels from model names
        class_names = model.names
        detection_labels = [class_names[int(class_id)] for class_id in detections.class_id]

        detections.label = [
            f"{label} {conf:.2f}" for label, conf in zip(detection_labels, detections.confidence)
        ]
        
        # Consistent color mapping for diseases
        # 0: Coccidiosis, 1: Healthy, 2: New Castle Disease, 3: Salmonella, 4: Sick
        color_map = {
            'Healthy': (0, 255, 0),
            'Sick': (0, 0, 255),             # Blue/Red for sick
            'Coccidiosis': (0, 165, 255),    # Orange
            'New Castle Disease': (0, 0, 255), # Red
            'Salmonella': (255, 0, 255)       # Magenta
        }
        
        detections.colors = [
            color_map.get(label, (255, 255, 255)) for label in detection_labels
        ]

        box_annotator = sv.BoxAnnotator(thickness=2)
        # full_res_rgb is already RGB, pass it directly
        annotated_frame = box_annotator.annotate(
            scene=full_res_rgb.copy(),
            detections=detections
        )

        # Convert RGB → BGR for OpenCV drawing & saving
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        for box, label, label_name in zip(detections.xyxy, detections.label, detection_labels):
            x1, y1, x2, y2 = map(int, box)
            color = color_map.get(label_name, (255, 255, 255))
            cv2.rectangle(annotated_frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame_bgr,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        counts = {}
        for label in detection_labels:
            counts[label] = counts.get(label, 0) + 1
        
        total_count = len(detections)

        img_h, img_w = annotated_frame_bgr.shape[:2]
        box_w, box_h = 280, 40 + (len(counts) * 25)
        x, y = img_w - box_w - 20, 20
        cv2.rectangle(annotated_frame_bgr, (x, y), (x + box_w, y + box_h), (0, 255, 255), -1)

        stats = [f"{label}: {count}" for label, count in counts.items()]
        stats.append(f"Total: {total_count}")

        for i, stat in enumerate(stats):
            cv2.putText(
                annotated_frame_bgr, stat,
                (x + 10, y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2, cv2.LINE_AA
            )

        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['DETECTION_OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_frame_bgr)
        
        # Restore CAM visualization
        sick_count = total_count - counts.get('Healthy', 0)
        target_cam_class = target_cam_class_id if target_cam_class_id is not None else 0
        
        if target_cam_class_id is None and sick_count > 0:
            # Pick the first non-healthy class for CAM as fallback
            for label, count in counts.items():
                if label != 'Healthy' and count > 0:
                    name_list = list(model.names.values()) if isinstance(model.names, dict) else model.names
                    target_cam_class = name_list.index(label)
                    break
        
        cam_success, cam_img = generate_cam_heatmap(model, img_cv, target_category_id=target_cam_class)
        cam_url = None
        if cam_success:
            cam_output_filename = f"cam_{filename}"
            cam_output_path = os.path.join(app.config['DETECTION_OUTPUT_FOLDER'], cam_output_filename)
            cv2.imwrite(cam_output_path, cam_img)
            cam_url = f'/static/detections/{cam_output_filename}'

        # Build comprehensive disease info
        disease_info_to_return = {}
        for k in counts.keys():
            if k in DISEASE_INFO:
                disease_info_to_return[k] = DISEASE_INFO[k]
        
        # If any sick chicks, add the potential specific diseases to info for user education
        if sick_count > 0:
            for d in ['Coccidiosis', 'New Castle Disease', 'Salmonella']:
                if d in DISEASE_INFO:
                    disease_info_to_return[d] = DISEASE_INFO[d]

        # Improved Primary Diagnosis: Prioritize any disease over "Healthy"
        if sick_count > 0:
            sick_labels = [k for k in counts.keys() if k != 'Healthy']
            primary_diag = max(sick_labels, key=counts.get) if sick_labels else 'Suspected Illness'
            if primary_diag == 'Sick':
                primary_diag = 'Potential: Coccidiosis / NCD / Salmonella'
        else:
            primary_diag = 'Healthy'
            
        final_img_url = f'/static/detections/{output_filename}'
        log_scan(filename, False, total_count, sick_count, primary_diag, final_img_url)

        return jsonify({
            'success': True,
            'detected_image_url': f'/static/detections/{output_filename}',
            'cam_image_url': cam_url,
            'counts': counts,
            'total_count': total_count,
            'unhealthy_count': sick_count,
            'disease_info': disease_info_to_return,
            'primary_diagnosis': primary_diag
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during image detection: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/detect_video', methods=['POST'])
def detect_video():
    if model is None:
        print("❌ Model not loaded.")
        return jsonify({'error': 'Model not loaded'}), 500

    file = request.files.get('file')
    if not file or file.filename == '' or not allowed_file(file.filename):
        print(f"❌ Invalid file: {file.filename if file else 'None'}")
        return jsonify({'error': 'Invalid video file'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file.save(input_path)

    output_filename = f"detected_{filename.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.abspath(os.path.join(app.config['DETECTION_OUTPUT_FOLDER'], output_filename))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open video file.")
        return jsonify({'error': 'Could not open video file'}), 500

    tracker = sv.ByteTrack()
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Optimization: If video is > 720p, downscale output for 3x speedup in encoding/drawing
    max_dim = 1280
    if cap_width > max_dim or cap_height > max_dim:
        scale = max_dim / max(cap_width, cap_height)
        width, height = int(cap_width * scale), int(cap_height * scale)
    else:
        width, height = cap_width, cap_height

    # Codec Selection: avc1/H264 is more browser-compatible than mp4v
    # Try avc1 (H.264) first, fallback if it fails
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("⚠️ avc1 (H.264) codec failed. Falling back to mp4v (requires FFmpeg for browser playback).")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ VideoWriter failed to open. Check permissions or codec.")
        return jsonify({'error': 'Failed to initialize video writer'}), 500

    healthy_count = 0
    sick_count = 0
    total_count = 0
    all_healthy_ids = set()
    all_sick_ids = set()

    try:
        frame_count = 0
        start_time = time.time()
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % FRAME_STRIDE != 0:
                # If skipping, just write the previous frame or high-res resized
                out.write(cv2.resize(frame, (width, height)))
                continue

            # Use Full-Res RGB for Tiling to capture details
            full_res_rgb = fast_to_rgb(frame)
            if full_res_rgb is None: break
            
            # High-Density Inference Toggle
            if USE_TILING_VIDEO:
                detections = get_tiled_detections(full_res_rgb, model)
            else:
                # Standard Inference (High-Res 1280 for accuracy)
                results = model(full_res_rgb, size=1280)
                detections = sv.Detections.from_yolov5(results)

            if len(detections) == 0:
                out.write(cv2.resize(frame, (width, height)))
                continue

            detections = detections[detections.confidence > model.conf]
            tracks = tracker.update_with_detections(detections)

            # Per-frame illness classification
            class_names = model.names
            frame_counts = {}

            for track_id, class_id in zip(tracks.tracker_id, tracks.class_id):
                label_name = class_names[int(class_id)]
                frame_counts[label_name] = frame_counts.get(label_name, 0) + 1
                
                # Accumulate unique IDs for each state
                if label_name != 'Healthy':
                    all_sick_ids.add(track_id)
                else:
                    all_healthy_ids.add(track_id)

            # ANNOTATION: Optimize by using resized frame if output is downscaled
            if width != cap_width:
                annotated_frame = cv2.resize(frame, (width, height))
            else:
                annotated_frame = frame.copy()

            # Resolution scaling Corrected:
            # Map detections from full-res (cap_width) to output-res (width)
            sx = width / float(cap_width)
            sy = height / float(cap_height)

            # Dynamic font & thickness based on detection count (handle dense frames)
            n_det = len(tracks.xyxy)
            if n_det > 30:
                font_scale = 0.30
                box_thick  = 1
                txt_thick  = 1
            elif n_det > 15:
                font_scale = 0.40
                box_thick  = 1
                txt_thick  = 1
            else:
                font_scale = 0.55
                box_thick  = 2
                txt_thick  = 2

            # Draw bounding boxes + labels for EVERY detected chick
            for box, conf, track_id, class_id in zip(
                tracks.xyxy, tracks.confidence, tracks.tracker_id, tracks.class_id
            ):
                x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
                label_name = class_names[int(class_id)]
                # Short labels for HUD, full for single boxes
                short_label = label_name[:1] if n_det > 10 else label_name
                color = COLOR_MAP.get(label_name, (200, 200, 200))
                
                label = f"{short_label}{track_id} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thick)
                
                (lw, lh), b = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_thick)
                ly = max(y1 - 4, lh + 4)
                cv2.rectangle(annotated_frame, (x1, ly - lh - b), (x1 + lw, ly + b), color, -1)
                cv2.putText(annotated_frame, label, (x1, ly - b), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0) if label_name == 'Healthy' else (255,255,255), txt_thick, cv2.LINE_AA)

            # Stats overlay (HUD)
            box_w, box_h = 240, 30 + (len(frame_counts) * 22)
            ox, oy = width - box_w - 16, 16
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (ox, oy), (ox + box_w, oy + box_h), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            for i, (txt_label, count) in enumerate(frame_counts.items()):
                col = COLOR_MAP.get(txt_label, (255,255,255))
                cv2.putText(annotated_frame, f"{txt_label}: {count}", (ox + 10, oy + 25 + i * 22), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

            out.write(annotated_frame)
        cap.release()
        out.release()
        total_time = time.time() - start_time
        print(f"✅ Video Processing Complete. Avg Speed: {frame_count/total_time:.1f} FPS")
        time.sleep(0.1)

        # Final counts
        sick_count = len(all_sick_ids)
        healthy_count = len(all_healthy_ids - all_sick_ids)
        total_count = len(all_healthy_ids | all_sick_ids)

        fixed_path = output_path.replace(".mp4", "_fixed.mp4")
        if fix_video_metadata(output_path, fixed_path):
            final_video_url = f'/static/detections/{os.path.basename(fixed_path)}'
        else:
            final_video_url = f'/static/detections/{output_filename}'

        # Build comprehensive disease info for video summary
        disease_info_to_return = {k: DISEASE_INFO.get(k, {}) for k in DISEASE_INFO.keys()}
        primary_diag = 'Suspected: Coccidiosis / NCD / Salmonella' if sick_count > 0 else 'Healthy'
        
        log_scan(filename, True, total_count, sick_count, primary_diag, final_video_url)

        return jsonify({
            'success': True,
            'detected_video_url': final_video_url,
            'total_count': total_count,
            'unhealthy_count': sick_count,
            'healthy_count': healthy_count,
            'disease_info': disease_info_to_return,
            'primary_diagnosis': 'Suspected: Coccidiosis / NCD / Salmonella' if sick_count > 0 else 'Healthy'
        })

    except Exception as e:
        import traceback
        print(f"❌ Error during video detection:")
        traceback.print_exc()
        return jsonify({'error': f'Video detection failed: {str(e)}'}), 500

    finally:
        cap.release()
        out.release()
        time.sleep(0.1)
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as e:
                print(f"⚠️ Failed to remove uploaded file: {e}")

import threading

# ── LIVE MONITOR PERFORMANCE SETTINGS ──
LIVE_INFERENCE_STRIDE = 3      # Run YOLO every Nth frame (1=every frame, 3=skip 2)
LIVE_INPUT_SIZE = 320           # Inference resolution (320 = fastest, 416 = balanced, 640 = accurate)
LIVE_JPEG_QUALITY = 60          # JPEG encoding quality (60 = fast streaming, 85 = high quality)
LIVE_CAM_WIDTH = 640            # Cap camera width
LIVE_CAM_HEIGHT = 480           # Cap camera height


class CameraStream:
    """Threaded camera capture — eliminates I/O blocking on the inference thread."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # DirectShow for faster init on Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize internal buffer lag
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()


def generate_frames():
    """High-performance video streaming generator.
    
    Optimizations applied:
    1. Threaded camera capture (no I/O blocking)
    2. Camera resolution capped at 640×480
    3. YOLO inference at 320px (2-4× faster than 640)
    4. Inference runs every Nth frame; cached detections reused in between
    5. Low JPEG quality for faster encoding + network transfer
    6. Lightweight OpenCV drawing (no supervision overhead)
    """
    camera = CameraStream(0)
    if not camera.isOpened():
        print("❌ Cannot open camera")
        return

    # Cached detections from last inference
    cached_boxes = []       # list of (x1,y1,x2,y2, label_name, conf)
    frame_idx = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, LIVE_JPEG_QUALITY]

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame_idx += 1

            # ── RUN INFERENCE only every Nth frame ──
            if frame_idx % LIVE_INFERENCE_STRIDE == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb, size=LIVE_INPUT_SIZE)
                preds = results.xyxy[0].cpu().numpy()  # Direct tensor access (fastest)

                cached_boxes = []
                h_frame, w_frame = frame.shape[:2]
                # Scale detections from inference size back to frame size
                for *xyxy, conf, cls_id in preds:
                    if conf < model.conf:
                        continue
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    label_name = model.names[int(cls_id)]
                    cached_boxes.append((x1, y1, x2, y2, label_name, float(conf)))

            # ── DRAW cached detections on current frame ──
            for (x1, y1, x2, y2, label_name, conf) in cached_boxes:
                color = COLOR_MAP.get(label_name, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{label_name} {conf:.2f}"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)
                txt_color = (0, 0, 0) if label_name == 'Healthy' else (255, 255, 255)
                cv2.putText(frame, label, (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1, cv2.LINE_AA)

            # ── Overlay live FPS-style stats HUD ──
            counts = {}
            for (_, _, _, _, ln, _) in cached_boxes:
                counts[ln] = counts.get(ln, 0) + 1
            if counts:
                hud_y = 20
                for lbl, cnt in counts.items():
                    col = COLOR_MAP.get(lbl, (255, 255, 255))
                    cv2.putText(frame, f"{lbl}: {cnt}", (10, hud_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                    hud_y += 25

            # ── Encode and yield ──
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()
        print("📷 Camera released.")


@app.route('/api/reports')
def get_reports():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify({'success': True, 'reports': history})
    except Exception as e:
        pass
    return jsonify({'success': True, 'reports': []})

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    global model
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'conf': model.conf if model else 0.25,
            'iou': model.iou if model else 0.45,
            'device': str(model.device) if model else 'unknown'
        })
    else:
        if not model:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        data = request.json
        if 'conf' in data:
            model.conf = float(data['conf'])
        if 'iou' in data:
            model.iou = float(data['iou'])
        return jsonify({'success': True})

@app.route('/api/datasets')
def api_datasets():
    try:
        images_path = os.path.join(os.path.dirname(__file__), 'data_training', 'images')
        labels_path = os.path.join(os.path.dirname(__file__), 'data_training', 'labels')
        
        img_count = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
        lbl_count = len(os.listdir(labels_path)) if os.path.exists(labels_path) else 0
        
        return jsonify({
            'success': True,
            'images': img_count,
            'labels': lbl_count,
            'active_model': 'chick_best.pt'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
