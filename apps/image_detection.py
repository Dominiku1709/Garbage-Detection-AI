
# app/run_image_app.py

import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import time
import os
import sys



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)



# IMPORT UTILITIES AND MODELS

from utils.class_remap import get_color_map
from models.loader import MODEL_NAMES,load_all_models
from inference.detector import run_inference


# CONFIGURATION

st.set_page_config(page_title="Image Detection & Comparison", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def _load_all_models_cached():
    return load_all_models()

def cached_load_all_models():
    st.toast("Loading all models...")
    models, errors = _load_all_models_cached()
    if errors:
        for name, err in errors.items():
            st.error(f"Failed to load {name}: {err}")
    st.toast("✅ Models loaded.")
    return models, errors



# PAGE 1: IMAGE UPLOAD & MODEL COMPARISON

def render_upload_page():
    st.header("📸 Image Upload & Model Comparison")
    st.info("Upload an image to run it through all 7 detection models.")

    uploaded_file = st.file_uploader("Choose an image to analyze", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Unified threshold control
        st.sidebar.header("⚙️ Configuration")
        global_conf_thresh = st.sidebar.slider(
            "Confidence Threshold (applies to all models)",
            0.0, 1.0, 0.25, 0.05
        )

        if st.button("Analyze with All Models", type="primary", use_container_width=True):
            st.subheader("🧩 Detection Results")

            all_models, errors = cached_load_all_models()

            if not all_models:
                st.error("No models were loaded. Cannot run analysis.")
                return

            num_cols = min(len(all_models), 3)
            cols = st.columns(num_cols)
            col_idx = 0

            # Convert PIL Image to BGR numpy array for OpenCV functions
            image_np_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            color_map = get_color_map() # Get the 6-class color map

            for model_name, model in sorted(all_models.items()):
                with cols[col_idx]:
                    st.markdown(f"##### {model_name}")
                    with st.spinner(f"Running {model_name}..."):
                        start_time = time.time()
                        
                        # --- Run Unified Inference ---
                        # This single function handles YOLO/RT-DETR,
                        # preprocessing, inference, remapping, and drawing.
                        annotated_image, detections = run_inference(
                            model,
                            image_np_bgr, 
                            global_conf_thresh,
                            color_map
                        )
                        end_time = time.time()
                        
                        # run_inference returns a BGR image, convert to RGB for Streamlit
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    st.image(
                        annotated_image_rgb,
                        caption=f"Threshold: {global_conf_thresh:.2f} | Time: {end_time - start_time:.2f}s | Detections: {len(detections)}"
                    )

                col_idx = (col_idx + 1) % num_cols

# MAIN
if __name__ == "__main__":
    st.sidebar.title("Image Detection App")
    st.sidebar.info(
        "This app is for static image analysis. "
        "Run `run_realtime_app.py` for live webcam tracking."
    )
    render_upload_page()