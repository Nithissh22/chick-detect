import torch
import cv2
import os

model = torch.hub.load('./yolov5', 'custom', path='./models/chick_best.pt', source='local')
print(f"Model Names: {model.names}")
