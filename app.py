import streamlit as st
import torch
from PIL import Image
import os
import cv2
import numpy as np
import time

# =========================================================
# IMPORT UTILS AND MODELS
# =========================================================
from utils import class_remap
from utils.class_remap import TACO_CLASS_NAMES, get_color_map
from utils.draw_utils import draw_boxes
from utils.camera_utils import manual_roi_setup
from utils.inference_utils import run_realtime_detection
from models import (
    MODEL_NAMES,
    load_all_models,
    load_single_model,
    run_inference,
)
from utils.utils import (
    get_device,
    box_cxcywh_to_xyxy
)

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="Garbage Detection Dashboard", layout="wide")
device = get_device()


@st.cache_resource
def cached_load_all_models():
    """Cache all models to speed up reload."""
    return load_all_models()


@st.cache_resource
def cached_load_single_model(model_name):
    """Cache single model for real-time detection."""
    return load_single_model(model_name)


# =========================================================
# IMAGE DETECTION FUNCTION (for upload mode)
# =========================================================
def predict_and_draw(model, model_name, image_np, conf_threshold=0.3, color_map=None):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    outputs = run_inference(model, image_pil)
    boxes_to_draw, labels_to_draw = [], []

    # --- YOLO MODELS ---
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

    # --- RT-DETR MODELS ---
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
                label_text = f"{fine_name}: {score_val:.2f}"

                boxes_to_draw.append(final_boxes[i].cpu().numpy())
                labels_to_draw.append(label_text)

    annotated_image = draw_boxes(
        cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB),
        boxes_to_draw,
        labels_to_draw,
        color_map
    )
    return annotated_image


# =========================================================
# PAGE 1: IMAGE UPLOAD & MODEL COMPARISON
# =========================================================
def render_upload_page():
    st.header("📸 Image Upload & Model Comparison")

    uploaded_file = st.file_uploader("Choose an image to analyze", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        st.header("⚙️ Configuration")
        st.sidebar.subheader("Confidence Thresholds")
        thresholds = {}
        sorted_model_names = sorted(MODEL_NAMES.values())
        for name in sorted_model_names:
            thresholds[name] = st.sidebar.slider(f"Threshold for {name}", 0.0, 1.0, 0.25, 0.05)

        if st.button("🚀 Analyze with All Models", type="primary", use_container_width=True):
            st.subheader("🧩 Detection Results")

            with st.spinner("Loading all models..."):
                all_models, errors = cached_load_all_models()

            if errors:
                for name, error_msg in errors.items():
                    st.error(f"Could not load **{name}**: `{error_msg}`")

            if not all_models:
                st.error("No models were loaded.")
                return

            num_cols = min(len(all_models), 3)
            cols = st.columns(num_cols)
            col_idx = 0

            image_np_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            color_map = class_remap.get_color_map(list(TACO_CLASS_NAMES.values()))

            for model_name, model in sorted(all_models.items()):
                with cols[col_idx]:
                    st.markdown(f"##### {model_name}")
                    conf_thresh = thresholds[model_name]
                    with st.spinner(f"Running {model_name}..."):
                        start_time = time.time()
                        annotated_image = predict_and_draw(model, model_name, image_np_bgr, conf_thresh, color_map)
                        end_time = time.time()

                    st.image(annotated_image, caption=f"Threshold: {conf_thresh:.2f} | Time: {end_time - start_time:.2f}s")

                col_idx = (col_idx + 1) % num_cols


# =========================================================
# PAGE 2: REAL-TIME LOCAL CAMERA DETECTION
# =========================================================
def render_realtime_page():
    st.header("🎥 Real-time Local Camera Detection")

    # Get mapping of display name -> internal key
    display_to_key = {v: k for k, v in MODEL_NAMES.items()}
    display_names = list(display_to_key.keys())

    # User selects by display name
    selected_display_name = st.sidebar.selectbox("Select a model", display_names)
    model_key = display_to_key[selected_display_name]  # get internal key
    model_name = MODEL_NAMES[model_key]

    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.info("⚙️ Click below to start camera feed — it will open in a new OpenCV window.")

    if st.button("▶️ Start Local Camera Feed", type="primary", use_container_width=True):
        with st.spinner(f"Loading {model_name}..."):
            model = cached_load_single_model(model_name)
        st.success(f"✅ {model_name} loaded successfully!")

        color_map = get_color_map()
        st.info("Draw 3 regions for 'Dry recycle', 'Organic', and 'Trash' bins.")
        rois = manual_roi_setup()
        if not rois:
            st.warning("No ROIs defined — please draw at least one region.")
            return

        # Run the real-time OpenCV feed
        run_realtime_detection(model, color_map, rois, device)

# =========================================================
# MAIN NAVIGATION
# =========================================================
def main():
    st.sidebar.title("♻️ Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["🖼️ Image Upload & Comparison", "🎥 Real-time Local Camera"],
        label_visibility="collapsed"
    )

    if page == "🖼️ Image Upload & Comparison":
        render_upload_page()
    elif page == "🎥 Real-time Local Camera":
        render_realtime_page()


if __name__ == "__main__":
    main()
