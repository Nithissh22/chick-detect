import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

class YOLOv5ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOv5ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # returns [batch_size, num_anchors, 85]
        return self.model(x)[0]

class YOLOv5Target:
    def __init__(self, category_id):
        self.category_id = category_id

    def __call__(self, model_outputs):
        # model_outputs is (batch, num_anchors, 85)
        # 4 is conf, 5+category_id is class probability
        # Let's maximize the sum of (confidence * class_prob) for the target class
        # to focus the heatmap on where this class exists
        if model_outputs.dim() == 3:
            preds = model_outputs[0]
            score = (preds[:, 4] * preds[:, 5 + self.category_id]).max()
            return score
        return model_outputs.sum()  # fallback

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('./yolov5', 'custom', path='./models/chick_best.pt', source='local', force_reload=True).to(device)
    model.eval()

    img_path = 'output.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image {img_path}")
        return

    img_resized = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = np.float32(rgb_img) / 255.0

    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

    # Wrap the model
    # model.model is the AutoShape wrapper
    # model.model.model is the actual YOLOv5 Sequential model
    wrapped_model = YOLOv5ModelWrapper(model.model.model)
    
    # Target the layer just before the Detection head. In YOLOv5 this is typically layer 23.
    target_layers = [wrapped_model.model.model[-2]] 

    # We will test class 0 (Healthy)
    targets = [YOLOv5Target(category_id=0)]

    try:
        # EigenCAM usually works best out of the box for YOLO because of complex backprop
        with EigenCAM(model=wrapped_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]

        # Resize the heatmap to original image size
        grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
        
        # Original float image for overlay
        orig_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img_float = np.float32(orig_img_rgb) / 255.0

        cam_image = show_cam_on_image(orig_img_float, grayscale_cam, use_rgb=True)

        # Save result
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_gradcam_out.jpg', cam_image_bgr)
        print("Successfully generated and saved test_gradcam_out.jpg")
    except Exception as e:
        print(f"Error during CAM generation: {e}")

if __name__ == "__main__":
    main()
