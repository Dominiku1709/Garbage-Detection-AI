import streamlit as st
import torch
from PIL import Image
import os
import cv2
import numpy as np
import random

# =========================================================
# IMPORT MODEL LOADERS AND UTILITIES
# =========================================================
from models import (
    load_Rtdetrv2,
    load_DistillConv,
    load_DistillVit,
    load_YOLO,
    run_inference,
)
from utils import (
    get_device,
    save_result,
    open_camera,
    capture_frame,
    release_camera,
    draw_boxes,
    box_cxcywh_to_xyxy
)

# =========================================================
# CONFIGURATION
# =========================================================
device = get_device()

@st.cache_resource
def load_model(model_name):
    if model_name == "Rtdetrv2": return load_Rtdetrv2()
    if model_name == "Distill-Convnet": return load_DistillConv()
    if model_name == "Distill-Vit": return load_DistillVit()
    if model_name == "YOLOv11l": return load_YOLO()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTDETR_CLASS_NAMES = {i: f"class_{i}" for i in range(60)}

@st.cache_data
def get_color_map(class_names):
    return {name: [random.randint(0, 255) for _ in range(3)] for name in class_names}

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_and_draw(model, image_np, conf_threshold=0.3, color_map=None):
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
                label = f"{class_names[class_id]}: {conf:.2f}"
                boxes_to_draw.append(coords)
                labels_to_draw.append(label)
    
    elif isinstance(outputs, dict) and "pred_logits" in outputs:
        class_names = RTDETR_CLASS_NAMES
        
        # LẤY LOGITS GỐC
        raw_logits = outputs["pred_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]

        pred_probs = raw_logits.sigmoid()

        scores, labels = pred_probs.max(dim=-1)
        
        keep = scores > conf_threshold
        
        print("\n" + "="*50)
        print(f"Ngưỡng tin cậy: {conf_threshold}")
        print(f"10 giá trị score cao nhất (sau sigmoid): {torch.topk(scores, 10).values.cpu().numpy()}")
        print(f"Số lượng box vượt qua ngưỡng: {torch.sum(keep).item()}")
        print("="*50 + "\n")

        if torch.sum(keep) > 0:
            filtered_boxes = pred_boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]

            final_boxes = box_cxcywh_to_xyxy(filtered_boxes, image_pil.width, image_pil.height)
            
            for i in range(final_boxes.shape[0]):
                score_val = filtered_scores[i].item()
                class_id = filtered_labels[i].item()
                label_text = f"{class_names.get(class_id, f'class_{class_id}')}: {score_val:.2f}"
                boxes_to_draw.append(final_boxes[i].cpu().numpy())
                labels_to_draw.append(label_text)

    annotated_image = draw_boxes(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), boxes_to_draw, labels_to_draw, color_map)
    return annotated_image

# =========================================================
# STREAMLIT APP
# =========================================================
def main():
    st.title("♻️ Garbage Detection & Classification")
    st.markdown("Upload an image or use your webcam for real-time detection.")

    st.sidebar.header("⚙️ Settings")
    model_name = st.sidebar.selectbox("Select model", ["YOLOv11l", "Rtdetrv2", "Distill-Convnet", "Distill-Vit"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05)
    
    with st.spinner(f"Loading {model_name}..."):
        model = load_model(model_name)
    st.sidebar.success(f"✅ {model_name} loaded successfully!")
    
    class_names_list = list(getattr(model, 'names', RTDETR_CLASS_NAMES).values())
    color_map = get_color_map(class_names_list)

    input_source = st.radio("Choose input source", ["Upload Image", "Real-time Webcam"], horizontal=True)

    if input_source == "Upload Image":
        if 'stop_camera' in st.session_state:
            st.session_state.stop_camera = True
            
        uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🚀 Predict"):
                with st.spinner("Running inference..."):
                    image_np = np.array(image)
                    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    annotated_image = predict_and_draw(model, image_np_bgr, confidence_threshold, color_map)
                    st.image(annotated_image, caption="Detection Result")

    elif input_source == "Real-time Webcam":
        st.info("Click 'Start Camera' to begin real-time detection.")
        
        col1, col2 = st.columns(2)
        start_button = col1.button("▶️ Start Camera", type="primary")
        stop_button = col2.button("⏹️ Stop Camera")

        if start_button: st.session_state.stop_camera = False
        if stop_button: st.session_state.stop_camera = True
        if 'stop_camera' not in st.session_state: st.session_state.stop_camera = True

        frame_placeholder = st.empty()
        
        if not st.session_state.stop_camera:
            cap = open_camera()
            try:
                while cap.isOpened() and not st.session_state.stop_camera:
                    frame = capture_frame(cap)
                    if frame is None:
                        st.warning("Failed to capture frame from camera. Stream may have ended.")
                        break
                    
                    annotated_frame = predict_and_draw(model, frame, confidence_threshold, color_map)
                    
                    frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
            finally:
                release_camera(cap)
                st.info("Camera feed stopped.")

if __name__ == "__main__":
    main()