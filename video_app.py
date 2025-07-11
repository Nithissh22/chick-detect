import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageOps
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import subprocess
import time 
from supervision.tracker.byte_tracker.core import ByteTrack

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DETECTION_OUTPUT_FOLDER = 'static/detections'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_OUTPUT_FOLDER'] = DETECTION_OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_FOLDER, exist_ok=True)

try:
    MODEL_PATH = './models/best.pt'
    model = torch.hub.load(
        './yolov5',
        'custom',
        path=MODEL_PATH,
        source='local',
        force_reload=True
    )
    model.conf = 0.30
    model.iou = 0.45
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Model names: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def fix_video_metadata(input_path, final_path):
    temp_path = final_path.replace(".mp4", "_fixed.mp4")

    try:
        subprocess.run([
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-movflags', '+faststart',
            '-c', 'copy',
            temp_path
        ], check=True)

        os.replace(temp_path, final_path)
        print(f"Metadata fixed and saved to: {final_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        return False


def preprocess_like_roboflow(cv_img, size=(640, 640)):
    if cv_img is None:
        return None
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    img_pil = ImageOps.exif_transpose(img_pil)
    img_pil = img_pil.resize(size, Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/detect', methods=['POST'])
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

        processed = preprocess_like_roboflow(img_cv)
        if processed is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        results = model(processed)
        detections_df = results.pandas().xyxy[0]

        detections = sv.Detections(
            xyxy=detections_df[['xmin', 'ymin', 'xmax', 'ymax']].values,
            confidence=detections_df['confidence'].values,
            class_id=detections_df['class'].values,
        )

        detections = detections[detections.confidence > model.conf]
        detections = detections.with_nms(threshold=model.iou)

        detection_status = ['Healthy' if conf > 0.40 else 'Sick' for conf in detections.confidence]

        detections.label = [
            f"{status} {conf:.2f}" for status, conf in zip(detection_status, detections.confidence)
        ]
        detections.colors = [
            (0, 255, 0) if status == "Healthy" else (0, 0, 255) for status in detection_status
        ]

        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_frame = box_annotator.annotate(
            scene=cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
            detections=detections
        )

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        for box, label, status in zip(detections.xyxy, detections.label, detection_status):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if status == "Healthy" else (0, 0, 255)
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

        healthy_count = detection_status.count("Healthy")
        unhealthy_count = detection_status.count("Sick")
        total_count = healthy_count + unhealthy_count

        img_h, img_w = annotated_frame_bgr.shape[:2]
        box_w, box_h = 260, 90
        x, y = img_w - box_w - 20, 20
        cv2.rectangle(annotated_frame_bgr, (x, y), (x + box_w, y + box_h), (0, 255, 255), -1)

        stats = [
            f"Healthy   : {healthy_count}",
            f"Unhealthy : {unhealthy_count}",
            f"Total     : {total_count}"
        ]

        for i, stat in enumerate(stats):
            cv2.putText(
                annotated_frame_bgr, stat,
                (x + 10, y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA
            )

        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['DETECTION_OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_frame_bgr)

        return jsonify({
            'success': True,
            'detected_image_url': f'/static/detections/{output_filename}'
        })

    except Exception as e:
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
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = f"detected_{filename.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join(app.config['DETECTION_OUTPUT_FOLDER'], output_filename)

    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return jsonify({'error': 'Could not open video file'}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return jsonify({'error': 'Failed to initialize video writer'}), 500

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = preprocess_like_roboflow(frame)
            results = model(processed_frame)
            detections_df = results.pandas().xyxy[0]

            if detections_df.empty:
                out.write(frame)
                continue

            detections = sv.Detections(
                xyxy=detections_df[['xmin', 'ymin', 'xmax', 'ymax']].values,
                confidence=detections_df['confidence'].values,
                class_id=detections_df['class'].values
            )

            detections = detections[detections.confidence > model.conf]
            detections = detections.with_nms(threshold=model.iou)

            tracks = tracker.update_with_detections(detections)

            healthy_ids = set()
            sick_ids = set()

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            for box, conf, track_id in zip(tracks.xyxy, tracks.confidence, tracks.tracker_id):
                x1, y1, x2, y2 = map(int, box)
                label_class = "Healthy" if conf > 0.40 else "Sick"
                color = (0, 255, 0) if label_class == "Healthy" else (0, 0, 255)
                label = f"ID:{track_id} {label_class} {conf:.2f}"

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if label_class == "Healthy":
                    healthy_ids.add(track_id)
                else:
                    sick_ids.add(track_id)

            common_ids = healthy_ids & sick_ids
            healthy_ids -= common_ids
            sick_ids -= common_ids

            healthy_count = len(healthy_ids)
            sick_count = len(sick_ids)
            total_count = healthy_count + sick_count

            annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            box_w, box_h = 260, 90
            x, y = width - box_w - 20, 20
            cv2.rectangle(annotated_frame, (x, y), (x + box_w, y + box_h), (0, 255, 255), -1)

            stats = [
                f"Healthy   : {healthy_count}",
                f"Unhealthy : {sick_count}",
                f"Total     : {total_count}"
            ]

            for i, stat in enumerate(stats):
                cv2.putText(annotated_frame, stat, (x + 10, y + 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            out.write(annotated_frame)

        cap.release()
        out.release()
        time.sleep(0.1)

        fixed_path = output_path.replace(".mp4", "_fixed.mp4")
        if fix_video_metadata(output_path, fixed_path):
            final_video_url = f'/static/detections/{os.path.basename(fixed_path)}'
        else:
            final_video_url = f'/static/detections/{output_filename}'

        return jsonify({
            'success': True,
            'detected_video_url': final_video_url
        })

    except Exception as e:
        print(f"❌ Error during video detection: {e}")
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
