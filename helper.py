from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import tempfile
import easyocr 
import numpy as np
import time
import torch

# Module-level device (set by load_model)
DEVICE = 'cpu'

def load_model(model_path, use_gpu=None):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    global DEVICE
    # determine device: respect explicit use_gpu when provided, otherwise auto-detect
    if use_gpu is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    model = YOLO(model_path)
    try:
        model.to(DEVICE)
    except Exception:
        # some YOLO versions may not support .to() in same way; ignore if fails
        pass
    return model

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    
    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf, device=DEVICE)
    
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Vidéo annotée — Chauves-souris détectées',
                   channels="BGR",
                   use_container_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def infer_uploaded_video(conf, model):
    """
    Vidéo en direct + détection. La lecture continue tant que tu ne cliques pas sur Pause.
    Tu arrêtes quand tu veux avec le bouton Pause, puis Reprendre pour continuer.
    """
    VIDEO_PATH = "video_analysis_path"
    FRAME_INDEX = "video_analysis_frame"
    TOTAL_FRAMES = "video_analysis_total"
    FILE_ID = "video_analysis_file_id"
    AUTO_ADVANCE = "video_analysis_playing"  # True = lecture en cours, False = en pause

    source_video = st.sidebar.file_uploader(
        label="Choisir une vidéo...",
        type=["mp4", "avi", "mov", "mkv"],
        help="Vidéo à analyser pour la détection de chauves-souris"
    )

    if source_video:
        file_id = source_video.name + "_" + str(source_video.size)
        if FILE_ID in st.session_state and st.session_state.get(FILE_ID) != file_id:
            for k in [VIDEO_PATH, FRAME_INDEX, TOTAL_FRAMES, FILE_ID, AUTO_ADVANCE]:
                if k in st.session_state:
                    del st.session_state[k]
        st.markdown("**Aperçu de la vidéo**")
        st.video(source_video)

    if not source_video:
        return

    st.markdown("---")
    st.markdown("**Résultat de l'analyse**")

    # Single control in sidebar (no dynamic checkbox creation)
    autoplay = st.sidebar.checkbox("Lecture automatique", value=True, key="autoplay_state")
    stop_btn = st.sidebar.button("Arrêter l'analyse")

    if st.button("Lancer l'analyse", type="primary"):
        try:
            # Save uploaded video to a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(source_video.getvalue())
            tfile.flush()
            path = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            st_frame = st.empty()

            frame_idx = 0
            # Continuous loop: process frames in this run (no st.rerun())
            while cap.isOpened():
                if stop_btn:
                    break

                ret, image = cap.read()
                if not ret:
                    break

                # Run inference and show frame
                res = model.predict(image, conf=conf, device=DEVICE)
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption=f"Vidéo annotée — Chauves-souris détectées · Frame {frame_idx + 1} / {total}",
                               channels="BGR",
                               use_container_width=True)

                frame_idx += 1

                # Check latest autoplay state from sidebar checkbox
                autoplay_state = st.session_state.get("autoplay_state", True)
                if not autoplay_state:
                    st.sidebar.info("Lecture en pause — cochez 'Lecture automatique' pour reprendre.")
                    # Wait until user resumes
                    while not st.session_state.get("autoplay_state", True):
                        if stop_btn:
                            cap.release()
                            return
                        time.sleep(0.2)

                # small delay to avoid UI flooding
                time.sleep(0.02)

            cap.release()
            st.success("Analyse terminée.")

        except Exception as e:
            st.error(f"Erreur : {e}")

def upload_easyocr(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            detected_frames(conf,
                                            model,
                                            st_frame,
                                            image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

def detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    reader = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf, device=DEVICE)

    # Create a copy of the original frame to modify
    modified_frame = image.copy()

    for i, license_plate in enumerate(res[0].boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = license_plate

        # Process license plate
        license_plate_image_gray = cv2.cvtColor(modified_frame[int(y1):int(y2), int(x1):int(x2), :], cv2.COLOR_BGR2GRAY)
        _, license_plate_image_thresh = cv2.threshold(license_plate_image_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Read license plate image
        detections = reader.readtext(license_plate_image_thresh)

        if detections:
            detected_plate_text = detections[0][1]  # Extract the detected text

            # Replace the class name with the detected license plate text
            license_plate_text = detected_plate_text

            # Draw a bounding box around the detected car plate
            cv2.rectangle(modified_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calculate the text size
            text_size, _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Calculate the position to center the text
            text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
            text_y = int(y1 - 10)

            # Calculate the background size (adjust as needed)
            background_width = text_size[0] + 20  # Add some extra space for padding
            background_height = text_size[1] + 10  # Add some extra space for padding

            # Calculate the position for the background
            background_x1 = text_x - 10
            background_y1 = text_y - text_size[1] - 5  # Adjusted to be higher
            background_x2 = background_x1 + background_width
            background_y2 = text_y + 5  # Adjusted to be lower

            # Draw the enlarged filled background for the text
            background_color = (0, 0, 0)
            cv2.rectangle(modified_frame, (background_x1, background_y1), (background_x2, background_y2), background_color, -1)

            # Draw the centered text on the enlarged filled background
            text_color = (255, 255, 255)  # Text color
            cv2.putText(modified_frame, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display the modified frame with updated bounding boxes and license plate text
    st_frame.image(modified_frame, caption='Detected Video', channels="BGR", use_container_width=True)

# Function to apply flood fill and other processing to the image
def process_license_plate(license_plate_image, floodfill_threshold, threshold_block_size, brightness):
    # Convert to grayscale
    license_plate_image_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

    # Replace dark values (below floodfill_threshold) with floodfill_threshold
    license_plate_image_gray[license_plate_image_gray < floodfill_threshold] = floodfill_threshold

    # Adjust brightness
    license_plate_image_bright = cv2.convertScaleAbs(license_plate_image_gray, alpha=1, beta=brightness)

    # Apply adaptive thresholding to create a black-and-white image
    license_plate_image_thresh = cv2.adaptiveThreshold(license_plate_image_bright, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_block_size, 2)

    return license_plate_image_thresh

    
