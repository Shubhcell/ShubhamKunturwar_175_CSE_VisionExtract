from flask import Flask, request, render_template, jsonify
import base64
import torch
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join(os.getcwd(), 'Model', 'best_unet_model.pth')

try:
    model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
    model = None

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(img_tensor.device)
    return img_tensor * std + mean

def image_to_data_url(image_np):
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image_np).save(buffer, format='PNG')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

kernel = np.ones((5, 5), np.uint8)


def data_url_to_image(data_url):
    header, encoded = data_url.split(',', 1)
    image_bytes = base64.b64decode(encoded)
    image_np = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode uploaded frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def segment_image(image_np):
    transformed = val_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    return pred_mask


def create_background(image_np, bg_type):
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
    # Resize mask to frame size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.float32)
    refined_mask = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    smooth_mask = cv2.GaussianBlur(refined_mask, (21, 21), 0)
    smooth_mask = np.clip(smooth_mask, 0, 1)
    smooth_mask = np.repeat(smooth_mask[:, :, None], 3, axis=2)
    background = create_background(image_np, bg_type)
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    composite = (image_np.astype(np.float32) * smooth_mask + background.astype(np.float32) * (1.0 - smooth_mask)).clip(0, 255).astype(np.uint8)
    return composite


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename and model:
            try:
                image = Image.open(file.stream).convert('RGB')
                image_np = np.array(image)

                transformed = val_transform(image=image_np)
                input_tensor = transformed['image'].unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(input_tensor)
                    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

                refined_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

                denorm_img = denorm(input_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy().clip(0, 1)

                isolated = denorm_img.copy()
                isolated[refined_mask == 0] = 0

                original_url = image_to_data_url(image_np)
                isolated_url = image_to_data_url(isolated)

                return render_template('index.html', original_url=original_url, isolated_url=isolated_url)
            except Exception as e:
                return render_template('index.html', error=f"Processing failed: {str(e)}")
        elif not file or not file.filename:
            return render_template('index.html', error="Please upload an image.")
        elif not model:
            return render_template('index.html', error="Model not loaded. Please check model file.")

    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    if not model:
        return jsonify(error="Model not loaded"), 500

    data = request.get_json(force=True)
    frame_data = data.get('frame') if data else None
    background_type = data.get('background', 'purple') if data else 'purple'

    if not frame_data:
        return jsonify(error="No frame data provided"), 400

    try:
        image_np = data_url_to_image(frame_data)
        mask = segment_image(image_np)
        h, w = image_np.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        processed = replace_background(image_np, mask, background_type)
        # Convert mask to 3-channel for display
        mask_rgb = np.stack([mask_resized * 255]*3, axis=2).astype(np.uint8)
        return jsonify(
            frame=image_to_data_url(processed),
            mask=image_to_data_url(mask_rgb)
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)