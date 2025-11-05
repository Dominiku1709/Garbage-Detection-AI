import streamlit as st
import torch
from PIL import Image
import os
import cv2
import numpy as np
import random
import time
from utils import *
from utils import camera_utils
from utils import utils

# =========================================================
# IMPORT MODEL LOADERS AND UTILITIES
# =========================================================
from models import (
    MODEL_NAMES,
    load_all_models,
    load_single_model,
    run_inference,
)
from utils.utils import (
    get_device,
    open_camera,
    capture_frame,
    release_camera,
    draw_boxes,
    box_cxcywh_to_xyxy
)

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(layout="wide")
device = get_device()


@st.cache_resource
def cached_load_all_models():
    return load_all_models()


@st.cache_resource
def cached_load_single_model(model_name):
    return load_single_model(model_name)


from utils import class_remap
from utils.class_remap import TACO_CLASS_NAMES, TACO_TO_6CLASS, get_color_map


# =========================================================
# PREDICTION FUNCTION (Không thay đổi)
# =========================================================
def predict_and_draw(model, model_name, image_np, conf_threshold=0.3, color_map=None):
    # ... (Giữ nguyên code của hàm này) ...
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    outputs = run_inference(model, image_pil)
    boxes_to_draw, labels_to_draw = [], []
    if isinstance(outputs, list):
        results = outputs[0]
        class_names = results.names
        for box in results.boxes:
            conf = box.conf.item()
            if conf >= conf_threshold:
                coords = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls.item())
                label = f"{class_names.get(class_id, 'N/A')}: {conf:.2f}"
                boxes_to_draw.append(coords)
                labels_to_draw.append(label)
    elif isinstance(outputs, dict) and "pred_logits" in outputs:
        class_names = class_remap.TACO_CLASS_NAMES
        raw_logits, pred_boxes = outputs["pred_logits"][0], outputs["pred_boxes"][0]
        pred_probs = raw_logits.sigmoid()
        scores, labels = pred_probs.max(dim=-1)
        if "ConvNeXt" in model_name:
            scores = scores * 4.0
        keep = scores > conf_threshold
        if torch.sum(keep) > 0:
            final_boxes = box_cxcywh_to_xyxy(pred_boxes[keep], image_pil.width, image_pil.height)
            for i in range(final_boxes.shape[0]):
                score_val = scores[keep][i].item()
                class_id = labels[keep][i].item()
                fine_name = class_names.get(class_id, f'class_{class_id}')
                broad_name = class_remap.CLASS_REMAP.get(fine_name, "Unknown")
                label_text = f"{broad_name}: {score_val:.2f}"

                boxes_to_draw.append(final_boxes[i].cpu().numpy())
                labels_to_draw.append(label_text)
    annotated_image = draw_boxes(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), boxes_to_draw, labels_to_draw, color_map)
    return annotated_image


# =========================================================
# UI HELPER FUNCTIONS
# =========================================================
def render_upload_page():
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("📸 Choose an image to analyze", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        st.header("2. Configure & Predict")

        # --- Tạo các thanh trượt cho từng model ---
        st.sidebar.subheader("Confidence Thresholds")
        thresholds = {}
        # Sắp xếp tên model để hiển thị nhất quán
        sorted_model_names = sorted(MODEL_NAMES.values())
        for name in sorted_model_names:
            thresholds[name] = st.sidebar.slider(f"Threshold for {name}", 0.0, 1.0, 0.25, 0.05)

        if st.button("🚀 Analyze with All Models", type="primary", use_container_width=True):
            st.subheader("3. Detection Results")

            with st.spinner("Loading all models... This may take a moment."):
                all_models, errors = cached_load_all_models()

            # Hiển thị lỗi nếu có
            if errors:
                for name, error_msg in errors.items():
                    st.error(f"Could not load **{name}**: `{error_msg}`")

            if not all_models:
                st.error("No models were loaded. Please check the console and file paths.")
                return

            num_cols = min(len(all_models), 3)
            cols = st.columns(num_cols)
            col_idx = 0

            image_np_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            color_map = class_remap.get_color_map(list(TACO_CLASS_NAMES.values()))

            for model_name, model in sorted(all_models.items()):
                with cols[col_idx]:
                    st.markdown(f"##### {model_name}")
                    conf_thresh = thresholds[model_name]  # Lấy ngưỡng của model tương ứng

                    with st.spinner(f"Running {model_name}..."):
                        start_time = time.time()
                        annotated_image = predict_and_draw(model, model_name, image_np_bgr, conf_thresh, color_map)
                        end_time = time.time()

                    st.image(annotated_image,
                             caption=f"Threshold: {conf_thresh:.2f} | Time: {end_time - start_time:.2f}s")

                col_idx = (col_idx + 1) % num_cols


def render_realtime_page():
    st.header("Real-time Webcam Detection")

    model_name = st.sidebar.selectbox("Select a model for real-time", list(MODEL_NAMES.values()))
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    with st.spinner(f"Loading {model_name}..."):
        model = cached_load_single_model(model_name)
    st.sidebar.success(f"✅ {model_name} loaded!")

    st.info("Click 'Start Camera' to begin detection.")

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Camera", type="primary"): st.session_state.stop_camera = False
    if col2.button("⏹️ Stop Camera"): st.session_state.stop_camera = True

    if 'stop_camera' not in st.session_state: st.session_state.stop_camera = True

    frame_placeholder = st.empty()

    if not st.session_state.stop_camera:
        color_map = get_color_map(list(TACO_CLASS_NAMES.values()))
        cap = open_camera()
        try:
            while cap.isOpened() and not st.session_state.get('stop_camera', True):
                frame = capture_frame(cap)
                if frame is None:
                    st.warning("Failed to capture frame. Stream may have ended.")
                    break

                annotated_frame = predict_and_draw(model, model_name, frame, confidence_threshold, color_map)
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        finally:
            release_camera(cap)
            st.info("Camera feed stopped.")


# =========================================================
# MAIN APP LOGIC
# =========================================================
def main():
    st.title("♻️ Garbage Detection: Model Comparison Dashboard")

    input_source = st.radio("Choose Mode", ["Compare All Models (Image Upload)", "Real-time (Webcam)"], horizontal=True,
                            label_visibility="collapsed")

    if input_source == "Compare All Models (Image Upload)":
        render_upload_page()
    else:
        render_realtime_page()

