from pathlib import Path
import sys

# Get the absolute path of the current file (only works in .py files) - path to this file ./settings.py
file_path = Path(__file__).resolve()

# Get the parent directory of the current file (main file: /yolov8-streamlit)
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there : allows for things like helper.process_license_plate()
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the main folder (basically IMAGES_DIR = ../yolov8-streamlit/'images')
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / '1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / '2.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
CUSTOM_MODEL1 = MODEL_DIR / 'pot.pt'
CUSTOM_MODEL2 = MODEL_DIR / 'car.pt'
CUSTOM_MODEL3 = MODEL_DIR / 'ppe.pt'

# Modèle dédié à la détection vidéo (YOLO fine-tuné chauves-souris)
# Poids issus de l'entraînement : train2/weights/
VIDEO_DETECTION_MODEL = root_path / 'train2' / 'weights' / 'best.pt'

# SAHI slicing inference (optional)
USE_SAHI = True
SAHI_SLICE_HEIGHT = 416
SAHI_SLICE_WIDTH = 416
SAHI_OVERLAP_HEIGHT = 0.35
SAHI_OVERLAP_WIDTH = 0.35

