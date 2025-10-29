import streamlit as st
import torch
from PIL import Image
import os
from datetime import datetime

# =========================================================
# 🔹 IMPORT MODEL LOADERS AND UTILITIES
# =========================================================
from models import (
    load_Rtdetrv2,
    load_DistillConv,
    load_DistillVit,
    load_YOLO,
    run_inference,  # ✅ unified inference function
)
from utils import preprocess_image, save_result, get_device


# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
device = get_device()

MODEL_PATHS = {
    "Rtdetrv2": "FINAL/FINETUNE_BASELINE/rtdetrv2_finetune_taco_BASELINE/last.pth",
    "Distill-Convnet": "FINAL/FINETUNE_DISTILLED/rtdetrv2_finetune_taco_convnext_teacher/last.pth",
    "Distill-Vit": "FINAL/FINETUNE_DISTILLED/rtdetrv2_finetune_taco_vit_teacher/best.pth",
    "YOLOv11l": "FINAL/YOLO/yolo11n.pt",
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 🔹 PREDICTION FUNCTION
# =========================================================
def predict(model, img, model_name):
    """Run inference depending on model type."""
    outputs = run_inference(model, img)

    # YOLO models return Results list
    if hasattr(outputs, "boxes"):
        boxes = outputs.boxes
        if boxes is None or len(boxes) == 0:
            return "No object detected", 0.0

        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        top_idx = confs.argmax()
        label = outputs.names[clss[top_idx]]
        conf = float(confs[top_idx])
        return label, conf

    # RT-DETR models return dict with logits/boxes
    elif isinstance(outputs, dict):
        pred_logits = outputs["pred_logits"].sigmoid()[0]
        pred_boxes = outputs["pred_boxes"][0]
        scores, labels = pred_logits.max(-1)
        top_idx = scores.argmax().item()
        return f"class_{labels[top_idx].item()}", float(scores[top_idx].item())

    else:
        return "Unknown output format", 0.0


# =========================================================
# 🎨 STREAMLIT APP
# =========================================================
def main():
    st.title("♻️ Garbage Detection & Classification")
    st.markdown("Upload an image and choose a model to detect recyclable objects.")

    model_name = st.selectbox("Select model", list(MODEL_PATHS.keys()))

    # Lazy-load model only when needed
    with st.spinner(f"Loading {model_name}..."):
        if model_name == "Rtdetrv2":
            model = load_Rtdetrv2()
        elif model_name == "Distill-Convnet":
            model = load_DistillConv()
        elif model_name == "Distill-Vit":
            model = load_DistillVit()
        else:
            model = load_YOLO()
    st.success(f"✅ {model_name} loaded successfully!")

    uploaded = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)

        if st.button("🚀 Predict"):
            with st.spinner("Running inference..."):
                label, conf = predict(model, img, model_name)
                st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

                folder = save_result(img, {"label": label, "confidence": conf})
                st.info(f"Results saved in `{folder}`")


if __name__ == "__main__":
    main()
