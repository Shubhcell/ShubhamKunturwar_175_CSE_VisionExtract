# VisionExtract

VisionExtract is an AI-powered Flask app for subject extraction and virtual background replacement.

## Features
- Static image background removal
- Live webcam background replacement with OpenCV
- Multiple virtual background styles

## Setup
1. Create a Python virtual environment:
   - `python -m venv venv`
   - `venv\Scripts\activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `python app.py`
4. Open `http://127.0.0.1:5000/` in your browser.
5. Use `/live` for the webcam background replacement UI.

## Deploying on Render

1. Push this repository to GitHub.
2. In Render, create a new Web Service and connect to this repo.
3. Set the Start Command to:
   - `gunicorn app:app`
4. Render will install from `requirements.txt` and use the `PORT` environment variable automatically.
5. Ensure `Model/best_unet_model.pth` remains in the repo so the model loads correctly.

## Notes
- The live mode captures webcam frames and sends them to the Flask backend for segmentation and background replacement.
- Make sure `Model/best_unet_model.pth` exists and is compatible with the loaded model architecture.
