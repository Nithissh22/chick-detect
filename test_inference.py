import torch
import cv2

MODEL_PATH = './models/chick_best.pt'
try:
    model = torch.hub.load(
        './yolov5',
        'custom',
        path=MODEL_PATH,
        source='local',
        force_reload=True
    )
    model.conf = 0.25
    print("Model loaded.")
    
    img = cv2.imread('output.jpg')
    if img is not None:
        results = model(img)
        df = results.pandas().xyxy[0]
        print(df)
    else:
        print("output.jpg not found.")
except Exception as e:
    print(e)
