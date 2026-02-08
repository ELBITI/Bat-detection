# Run Guide

## Prerequisites
- Python 3.10+ (3.11 recommended)
- pip
- `sahi` package

## Install
From the project root:

```
python -m venv venv
```

Windows PowerShell:

```
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```
pip install -r requirements.txt
```

## Run the app

```
streamlit run app.py
```

The app will open in your browser.

## Notes
- The app auto-selects GPU if available (you can override via the sidebar checkbox).
- The video workflow needs OpenCV. If you see a cv2 error, install:
  `pip install opencv-python-headless`
- If SAHI is enabled (see `settings.py`), make sure `sahi` is installed.
- Model path is configured in `settings.py` (`VIDEO_DETECTION_MODEL`).

## Troubleshooting
- If the app uses CPU only, check your CUDA setup and `Use GPU` checkbox.
- If you see missing package errors, re-run `pip install -r requirements.txt`.
