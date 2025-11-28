# AI Image Detector (Streamlit)

A Streamlit web app that predicts whether an image is likely **AI generated (FAKE)** or **real**, using a Hugging Face image classification model.

Use this as a **triage signal** (route to manual review), not as final proof.

## Features
- Upload images: JPG, JPEG, PNG, WEBP
- Camera capture (if supported by your browser)
- Batch upload (multiple images at once)
- Threshold slider to flag as FAKE
- Manual review zone for borderline scores

## Model
- dima806/ai_vs_real_image_detection

## Project Structure

    .
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Local Setup

### 1) Create and activate a virtual environment

Mac or Linux:

    python -m venv .venv
    source .venv/bin/activate

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### 2) Install dependencies

    pip install -r requirements.txt

### 3) Run the app

    streamlit run app.py

Then open the URL shown in your terminal (usually http://localhost:8501).

## requirements.txt

Make sure your requirements.txt contains:

    streamlit>=1.30
    transformers>=4.40
    torch
    pillow

## Deploy

### Option A: Streamlit Community Cloud
1. Push this repo to GitHub
2. Go to Streamlit Community Cloud and click "New app"
3. Select your repo and branch (main)
4. Main file path: app.py
5. Deploy

Note: First boot can be slower because the model downloads on first run.

### Option B: Hugging Face Spaces (Streamlit)
1. Create a new Space
2. Choose SDK: Streamlit
3. Upload files or connect your GitHub repo
4. Keep app.py and requirements.txt in the repo root

## Tips
- Tune the threshold using samples from your use case
- Keep a manual review step for high stakes decisions
- Detectors can drift over time as new generators improve

## Disclaimer
This tool provides a probabilistic prediction and may produce false positives or false negatives. Always include human review for important decisions.

## License
Add a LICENSE file (MIT is a common choice).
