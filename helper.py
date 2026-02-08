from ultralytics import YOLO
import streamlit as st
# Handle systems where OpenCV is not installable (avoid hard crash at import)
try:
    import cv2
    HAS_CV2 = True
except Exception as _ex:
    cv2 = None
    HAS_CV2 = False
    CV2_IMPORT_ERROR = str(_ex)
from pytube import YouTube
import tempfile
import easyocr 
import numpy as np
import time
import torch
import math
import os
import settings

try:
    from sahi.predict import get_sliced_prediction
    from sahi.auto_model import AutoDetectionModel
    HAS_SAHI = True
except Exception as _ex:
    HAS_SAHI = False
    SAHI_IMPORT_ERROR = str(_ex)

# Module-level device (set by load_model)
DEVICE = 'cpu'

def _format_mmss(seconds):
    seconds = max(0, int(round(seconds)))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:d}:{secs:02d}"

def _downsample_scores(scores, max_bins=400):
    if not scores:
        return [], 1
    if len(scores) <= max_bins:
        return scores, 1
    bin_size = int(math.ceil(len(scores) / max_bins))
    binned = []
    for i in range(0, len(scores), bin_size):
        binned.append(int(max(scores[i:i + bin_size])))
    return binned, bin_size

def _compute_top_segments(scores, max_per_frame, avg_conf, window_s=5, top_k=5):
    if not scores:
        return []
    window_s = max(1, int(window_s))
    segments = []
    last_start = max(0, len(scores) - window_s)
    for start in range(0, last_start + 1):
        end = start + window_s
        slice_scores = scores[start:end]
        slice_max_frame = max_per_frame[start:end]
        slice_conf = avg_conf[start:end]
        segments.append({
            'start': start,
            'end': end,
            'detections_sum': int(sum(slice_scores)),
            'detections_max_per_frame': int(max(slice_max_frame) if slice_max_frame else 0),
            'avg_conf': float(np.mean(slice_conf)) if slice_conf else 0.0,
        })
    segments.sort(key=lambda x: (x['detections_sum'], x['detections_max_per_frame']), reverse=True)
    return segments[:top_k]

def _export_highlight_clips(video_path, segments, fps, out_dir='outputs/highlights', padding_sec=5):
    if not segments:
        return []
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    for idx, seg in enumerate(segments, start=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        start_sec = max(0, seg['start'] - padding_sec)
        end_sec = seg['end'] + padding_sec
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        if total_frames > 0:
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_path = os.path.join(out_dir, f'highlight_{idx}_{start_sec}_{end_sec}.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cap.release()
        clips.append({'path': out_path, 'start': start_sec, 'end': end_sec})
    return clips

def _get_sahi_model(conf):
    if not HAS_SAHI:
        return None
    model_path = str(settings.VIDEO_DETECTION_MODEL)
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=DEVICE,
    )

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
                   width='stretch'
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
        if not HAS_CV2:
            st.sidebar.error("OpenCV (cv2) n'est pas disponible — cette fonctionnalité est désactivée sur ce serveur.")
            return
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
        if not HAS_CV2:
            st.sidebar.error("OpenCV (cv2) n'est pas disponible — cette fonctionnalité est désactivée sur ce serveur.")
            return
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
            use_sahi = bool(getattr(settings, "USE_SAHI", False))
            sahi_model = None
            if use_sahi:
                if not HAS_SAHI:
                    st.warning("SAHI n'est pas disponible dans l'environnement. Retour au mode YOLO standard.")
                    use_sahi = False
                else:
                    sahi_model = _get_sahi_model(conf)
            # Save uploaded video to a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(source_video.getvalue())
            tfile.flush()
            path = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            st_frame = st.empty()

            frame_idx = 0
            sec_scores = {}
            sec_conf_sum = {}
            sec_conf_count = {}
            sec_max_frame = {}
            # Continuous loop: process frames in this run (no st.rerun())
            while cap.isOpened():
                if stop_btn:
                    break

                ret, image = cap.read()
                if not ret:
                    break

                # Convert infrared video to grayscale for more stable detection
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                except Exception:
                    image_proc = image

                # Run inference and show frame
                if use_sahi and sahi_model is not None:
                    result = get_sliced_prediction(
                        image_proc,
                        sahi_model,
                        slice_height=getattr(settings, "SAHI_SLICE_HEIGHT", 416),
                        slice_width=getattr(settings, "SAHI_SLICE_WIDTH", 416),
                        overlap_height_ratio=getattr(settings, "SAHI_OVERLAP_HEIGHT", 0.35),
                        overlap_width_ratio=getattr(settings, "SAHI_OVERLAP_WIDTH", 0.35),
                    )
                    modified_frame = image_proc.copy()
                    scores = []
                    for obj in result.object_prediction_list:
                        x1, y1, x2, y2 = obj.bbox.to_xyxy()
                        score = float(obj.score.value)
                        scores.append(score)
                        cv2.rectangle(
                            modified_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                            lineType=cv2.LINE_AA
                        )
                    res_plotted = modified_frame
                    count = int(len(scores))
                    conf_vals = np.array(scores, dtype=float) if scores else np.array([], dtype=float)
                else:
                    res = model.predict(image_proc, conf=conf, device=DEVICE)
                    boxes = res[0].boxes
                    count = int(len(boxes))
                    conf_vals = boxes.conf.detach().cpu().numpy() if count > 0 and hasattr(boxes, "conf") else np.array([], dtype=float)
                    res_plotted = res[0].plot()
                sec = int(frame_idx / fps) if fps > 0 else 0
                sec_scores[sec] = sec_scores.get(sec, 0) + count
                sec_max_frame[sec] = max(sec_max_frame.get(sec, 0), count)
                if count > 0 and conf_vals.size > 0:
                    sec_conf_sum[sec] = sec_conf_sum.get(sec, 0.0) + float(conf_vals.sum())
                    sec_conf_count[sec] = sec_conf_count.get(sec, 0) + int(count)
                st_frame.image(res_plotted,
                               caption=f"Vidéo annotée — Chauves-souris détectées · Frame {frame_idx + 1} / {total}",
                               channels="BGR",
                               width='stretch')

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

            duration_sec = int(math.ceil(frame_idx / fps)) if fps > 0 else len(sec_scores)
            scores = [sec_scores.get(i, 0) for i in range(duration_sec)]
            max_per_frame = [sec_max_frame.get(i, 0) for i in range(duration_sec)]
            avg_conf = []
            for i in range(duration_sec):
                if sec_conf_count.get(i, 0) > 0:
                    avg_conf.append(sec_conf_sum.get(i, 0.0) / sec_conf_count.get(i, 1))
                else:
                    avg_conf.append(0.0)

            st.markdown("**Timeline des detections**")
            st.caption("Lecture : l'axe horizontal represente le temps (video) et l'intensite indique ou les detections sont les plus fortes.")
            binned, bin_size = _downsample_scores(scores, max_bins=400)
            if binned:
                st.bar_chart(binned)
                st.caption("Axes : X = temps (secondes, regroupees par bins), Y = nombre total de detections dans la portion.")
            else:
                st.info("Aucune detection a afficher dans la timeline.")

            st.markdown("**Courbe des detections (par seconde)**")
            if scores:
                st.line_chart(scores)
                st.caption("Axes : X = seconde de la video, Y = nombre de detections sur cette seconde.")
            else:
                st.info("Aucune detection a afficher dans la courbe.")

            st.markdown("**Top segments (fenetres de 5s)**")
            segments = _compute_top_segments(scores, max_per_frame, avg_conf, window_s=5, top_k=5)
            if segments:
                seg_rows = []
                for seg in segments:
                    seg_rows.append({
                        'start': _format_mmss(seg['start']),
                        'end': _format_mmss(seg['end']),
                        'detections_sum': seg['detections_sum'],
                        'detections_max_per_frame': seg['detections_max_per_frame'],
                        'avg_conf': round(seg['avg_conf'], 3),
                    })
                st.table(seg_rows)
            else:
                st.info("Aucun segment a afficher.")

            if segments:
                st.markdown("**Extraits des meilleurs segments**")
                clips = _export_highlight_clips(path, segments, fps, out_dir='outputs/highlights', padding_sec=5)
                if clips:
                    for clip in clips:
                        st.caption(f"Extrait {_format_mmss(clip['start'])} -> {_format_mmss(clip['end'])}")
                        try:
                            with open(clip['path'], 'rb') as fvid:
                                st.video(fvid.read())
                        except Exception:
                            st.video(clip['path'])
                        with open(clip['path'], 'rb') as f:
                            st.download_button(
                                label="Telecharger l'extrait",
                                data=f,
                                file_name=os.path.basename(clip['path']),
                                mime='video/mp4',
                            )

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
    # If OpenCV is not available in the deployment environment, show the raw frame and skip processing
    if not HAS_CV2:
        st_frame.image(image, caption='OpenCV non disponible — affichage brut', width='stretch')
        return

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
    st_frame.image(modified_frame, caption='Detected Video', channels="BGR", width='stretch')

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

    
