import torch
import cv2
import numpy as np
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Load model
model_path = os.path.join(os.getcwd(), 'Model', 'best_unet_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

kernel = np.ones((5, 5), np.uint8)

def segment_image(image_np):
    transformed = val_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    return pred_mask

def create_background(image_np, bg_type='purple'):
    height, width = image_np.shape[:2]
    if bg_type == 'blur':
        return cv2.GaussianBlur(image_np, (51, 51), 0)
    if bg_type == 'purple':
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = (179, 86, 255)
        return bg
    if bg_type == 'teal':
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = (64, 224, 208)
        return bg
    if bg_type == 'gradient':
        start = np.array([255, 179, 241], dtype=np.float32)
        stop = np.array([99, 102, 241], dtype=np.float32)
        gradient = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            bg[..., i] = (start[i] * (1 - gradient) + stop[i] * gradient).reshape(-1, 1)
        return bg
    return np.zeros((height, width, 3), dtype=np.uint8)

def replace_background(image_np, mask, bg_type='purple'):
    h, w = image_np.shape[:2]
    background = create_background(image_np, bg_type)
    # Resize mask to original frame size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Ensure mask is 0/1 float
    mask_resized = (mask_resized > 0).astype(np.float32)
    refined_mask = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    smooth_mask = cv2.GaussianBlur(refined_mask, (21, 21), 0)
    smooth_mask = np.clip(smooth_mask, 0, 1)
    smooth_mask = np.repeat(smooth_mask[:, :, None], 3, axis=2)
    # Resize background if needed
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    composite = (image_np.astype(np.float32) * smooth_mask + background.astype(np.float32) * (1.0 - smooth_mask)).clip(0, 255).astype(np.uint8)
    return composite

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return
    bg_types = ['purple', 'teal', 'gradient', 'blur']
    bg_idx = 0
    print('Press b to change background, q to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = segment_image(frame_rgb)
        result = replace_background(frame_rgb, mask, bg_type=bg_types[bg_idx])
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # Show the mask for debugging
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Segmentation Mask', mask_resized * 255)
        cv2.imshow('Live Background Removal', result_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            bg_idx = (bg_idx + 1) % len(bg_types)
            print('Background:', bg_types[bg_idx])
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
