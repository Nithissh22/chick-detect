import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

class YOLOv5ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOv5ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]

class YOLOv5Target:
    def __init__(self, category_id):
        self.category_id = category_id

    def __call__(self, model_outputs):
        if model_outputs.dim() == 3:
            preds = model_outputs[0]
            # 4 is conf, 5+category is class probability
            score = (preds[:, 4] * preds[:, 5 + self.category_id]).max()
            return score
        return model_outputs.sum()

def generate_cam_heatmap(model, img_bgr, target_category_id=0):
    try:
        # Resize to a consistent square size as YOLO expects, e.g., 640
        img_resized = cv2.resize(img_bgr, (640, 640))
        rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = np.float32(rgb_img) / 255.0

        device = next(model.parameters()).device
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

        # torch.hub model: AutoShape -> DetectionModel -> Sequential
        # model.model = DetectionModel, model.model.model = Sequential layers
        wrapped_model = YOLOv5ModelWrapper(model.model)
        # target_layers is usually one of the layers near the end before detection
        target_layers = [wrapped_model.model.model[-2]] 
        targets = [YOLOv5Target(category_id=target_category_id)]

        with EigenCAM(model=wrapped_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]

        # Resize back to original image dimensions
        grayscale_cam = cv2.resize(grayscale_cam, (img_bgr.shape[1], img_bgr.shape[0]))
        
        orig_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_img_float = np.float32(orig_img_rgb) / 255.0

        cam_image = show_cam_on_image(orig_img_float, grayscale_cam, use_rgb=False)
        return True, cam_image
    except Exception as e:
        print(f"CAM Generation Error: {e}")
        return False, img_bgr
