# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import cv2
import numpy as np
import torch

# Local Modules
import settings
import helper

# ============ CONFIGURATION DE LA PAGE ============
st.set_page_config(
    page_title="Détection de chauves-souris | Vision par ordinateur",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ STYLE PROFESSIONNEL + ANIMATIONS ============
st.markdown("""
<style>
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-12px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes subtlePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.92; }
    }

    /* Fond principal */
    .stApp {
        background: #f1f5f9;
    }
    [data-testid="stVerticalBlock"] > div {
        background: transparent;
        box-shadow: none;
        padding: 0.25rem 0;
    }

    /* Hero / En-tête */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.2);
        animation: fadeInUp 0.6s ease-out;
    }
    .hero h1 {
        color: #fff;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hero .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.75rem 0 0 0;
        line-height: 1.6;
    }

    /* Barre d’étapes */
    .steps-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-out 0.2s both;
    }
    .step-pill {
        background: #fff;
        color: #475569;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .step-pill.active { background: #1d4ed8; color: #fff; }

    /* Zone de contenu principale */
    .workspace {
        background: #fff;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        animation: fadeInUp 0.5s ease-out 0.15s both;
    }
    .workspace-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        animation: fadeInUp 0.5s ease-out 0.2s both;
    }

    /* Instructions */
    .info-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #1e3a5f;
        animation: fadeIn 0.5s ease-out 0.25s both;
    }

    /* Sidebar : structure claire, sans cartes blanches */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #cbd5e1 !important; font-weight: 500; }
    [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
    .sidebar-section {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1rem 0 0.5rem 0;
        padding-left: 0.5rem;
        border-left: 3px solid #3b82f6;
    }

    /* Boutons (couleurs conservées) */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        box-shadow: 0 4px 14px rgba(30, 64, 175, 0.45);
        transform: translateY(-1px);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] { color: #1e40af; }

    /* Pied de page */
    .footer {
        margin-top: 2rem;
        padding: 1.25rem;
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        animation: fadeIn 0.5s ease-out 0.4s both;
    }
</style>
""", unsafe_allow_html=True)

# ============ HERO ============
st.markdown("""
<div class="hero">
    <h1>🦇 Détection automatique de chauves-souris</h1>
    <p class="subtitle">Vision par ordinateur · YOLOv8 · Détection précise en faible luminosité et mouvement rapide</p>
</div>
""", unsafe_allow_html=True)

# ============ BARRE D’ÉTAPES ============
st.markdown("""
<div class="steps-bar">
    <span class="step-pill active">1. Paramètres</span>
    <span class="step-pill">2. Source</span>
    <span class="step-pill">3. Analyse</span>
</div>
""", unsafe_allow_html=True)

# ============ SIDEBAR : structure claire ============
st.sidebar.markdown('<p class="sidebar-section">Paramètres</p>', unsafe_allow_html=True)
confidence = float(st.sidebar.slider(
    "Seuil de confiance (%)",
    min_value=25,
    max_value=95,
    value=50,
    step=5
)) / 100

st.sidebar.markdown('<p class="sidebar-section">Source</p>', unsafe_allow_html=True)
source_radio = st.sidebar.radio(
    "Choisir la source",
    [settings.VIDEO, settings.IMAGE],
    format_func=lambda x: {"Video": "Vidéo", "Image": "Image"}.get(x, x)
)

# ============ MODÈLE YOLO (chauves-souris) ============
@st.cache_resource
def load_model(model_path, use_gpu):
    try:
        return helper.load_model(model_path, use_gpu=use_gpu)
    except Exception as ex:
        st.error(f"Impossible de charger le modèle. Vérifiez le chemin : {model_path}")
        st.error(ex)
        return None

# Modèle dédié à la détection de chauves-souris (vidéo et image)
model_path = Path(settings.VIDEO_DETECTION_MODEL)

# GPU option: active par défaut si CUDA est disponible
default_gpu = torch.cuda.is_available()

# Afficher le statut du GPU de manière claire
st.sidebar.markdown('<p class="sidebar-section">Performance</p>', unsafe_allow_html=True)
if default_gpu:
    st.sidebar.success("✅ GPU détecté — Utilisation activée par défaut")
else:
    st.sidebar.info("ℹ️ Pas de GPU — Utilisation du CPU")

# Checkbox pour activer/désactiver GPU
use_gpu = st.sidebar.checkbox("Utiliser GPU si disponible", value=default_gpu)
st.session_state["use_gpu_option"] = use_gpu

model = load_model(model_path, use_gpu)

if model is None:
    st.warning("Modèle introuvable. Vérifiez que **train2/weights/best.pt** existe (poids issus de l'entraînement).")
    st.stop()

# ============ FLUX PRINCIPAL ============
source_img = None

if source_radio == settings.IMAGE:
    # ---------- Mode Image ----------
    st.sidebar.markdown("---")
    source_img = st.sidebar.file_uploader(
        "Choisir une image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Image de test pour la détection"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Image d'entrée**")
        try:
            if source_img is None:
                default_path = str(settings.DEFAULT_IMAGE)
                st.image(default_path, caption="Image par défaut", use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Image chargée", use_container_width=True)
        except Exception as ex:
            st.error("Erreur lors du chargement de l'image.")
            st.error(ex)

    with col2:
        st.markdown("**Résultat de la détection**")
        if source_img is None:
            default_det_path = str(settings.DEFAULT_DETECT_IMAGE)
            try:
                st.image(default_det_path, caption="Exemple de détection", use_container_width=True)
            except Exception:
                st.info("Chargez une image puis cliquez sur **Détecter les chauves-souris**.")
        else:
            if st.sidebar.button("Détecter les chauves-souris"):
                with st.spinner("Analyse en cours..."):
                    res = model.predict(uploaded_image, conf=confidence, device=helper.DEVICE)
                    boxes = res[0].boxes
                    if len(boxes) == 0:
                        st.warning("Aucune chauve-souris détectée sur cette image.")
                    else:
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption=f"Détection — {len(boxes)} individu(s) détecté(s)", use_container_width=True)
                        with st.expander("Détails des détections"):
                            for i, box in enumerate(boxes):
                                st.write(f"**Détection {i+1}** : ", box.data)

elif source_radio == settings.VIDEO:
    # ---------- Mode Vidéo (flux principal) ----------
    st.markdown("""
    <div class="info-card">
        <strong>Comment utiliser :</strong> Uploadez une vidéo dans la barre latérale, puis cliquez sur <strong>Lancer l'analyse</strong>. 
        Les chauves-souris détectées s'affichent sur la vidéo ; vous pouvez récupérer une vidéo annotée en sortie.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="workspace-title">Espace d'analyse vidéo</p>""", unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sidebar-section">Vidéo</p>', unsafe_allow_html=True)
    helper.infer_uploaded_video(confidence, model)

else:
    st.info("Sélectionnez **Vidéo** ou **Image** dans la barre latérale.")

# ============ PIED DE PAGE ============
st.markdown("""
<div class="footer">
    Détection de chauves-souris
</div>
""", unsafe_allow_html=True)
