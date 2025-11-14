# =========================================================
# models/loader.py
# =========================================================
import torch
from ultralytics import YOLO
import sys
import os
from PIL import Image
import numpy as np
import cv2

#from models.rtdetr_loader import load_rtdetr_model

# ---------------------------------------------------------
# Device setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model names and paths (7 total)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = {
    # Nhóm DINOv3
    "yolo_dinov3": "YOLO (DINOv3 Baseline)",
    "rtdetr_dinov3_baseline": "RT-DETR (DINOv3 Baseline)",
    "rtdetr_dinov3_distill_convnext": "RT-DETR (Distill DinoV3 w/ ConvNeXt)",
    "rtdetr_dinov3_distill_vit": "RT-DETR (Distill DinoV3 w/ ViT)",

    # Nhóm CodeTR
    "yolo_codetr": "YOLO (CodeTR Baseline)",
    "rtdetr_codetr_baseline": "RT-DETR (CodeTR Baseline)",
    "rtdetr_codetr_distilled": "🚀RT-DETR (Distill CodeTR)",
}

MODEL_PATHS = {
    MODEL_NAMES["yolo_dinov3"]: os.path.join(PROJECT_ROOT, "FINAL/YOLO_DINOV3/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt"),
    MODEL_NAMES["rtdetr_dinov3_baseline"]: os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_BASELINE_DINOv3/rtdetrv2_finetune_taco_finetune_BASELINE/best.pth"),
    MODEL_NAMES["rtdetr_dinov3_distill_convnext"]: os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_DISTILLED_DINOv3/rtdetrv2_finetune_taco_convnext_teacher/best.pth"),
    MODEL_NAMES["rtdetr_dinov3_distill_vit"]: os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_DISTILLED_DINOv3/rtdetrv2_finetune_taco_vit_teacher/best.pth"),

    MODEL_NAMES["yolo_codetr"]: os.path.join(PROJECT_ROOT, "FINAL/YOLO_CODER/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt"),
    MODEL_NAMES["rtdetr_codetr_baseline"]: os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_BASELINE_CODETR/best.pth"),
    MODEL_NAMES["rtdetr_codetr_distilled"]: os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_DISTILLED_CODETR/best.pth"),
}

CONFIG_PATHS = {
    MODEL_NAMES["rtdetr_dinov3_baseline"]: os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_BASELINE.yml"),
    MODEL_NAMES["rtdetr_dinov3_distill_convnext"]: os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_convnext.yml"),
    MODEL_NAMES["rtdetr_dinov3_distill_vit"]: os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_vit.yml"),
    
    # !!! QUAN TRỌNG: Đảm bảo các file này tồn tại sau khi chạy script generate_configs !!!
    MODEL_NAMES["rtdetr_codetr_baseline"]: os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_baseline_codetr.yml"),
    MODEL_NAMES["rtdetr_codetr_distilled"]: os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_distilled_codetr.yml"),
}

# ---------------------------------------------------------
# Load all models
# ---------------------------------------------------------
def load_all_models():
    """Load all models, return dict and errors dict."""
    models, errors = {}, {}
    for name, path in MODEL_PATHS.items():
        try:
            if "YOLO" in name:
                models[name] = YOLO(path).to(device)
            elif "RT-DETR" in name:
                config_path = CONFIG_PATHS.get(name)
                models[name] = load_rtdetr_model(path, config_path)
            else:
                errors[name] = "Unknown model type"
        except Exception as e:
            errors[name] = str(e)
            print(f"Failed to load {name}: {e}")
    return models, errors


def load_rtdetr_model(model_path: str, config_path: str):
    # ... (giữ nguyên code của hàm này)
    original_cwd = os.getcwd()
    rtdetr_code_path = os.path.join(PROJECT_ROOT, "rtdetr")
    if not os.path.isdir(rtdetr_code_path):
        raise FileNotFoundError(f"Thư mục 'rtdetr' không được tìm thấy: {rtdetr_code_path}")
    
    original_sys_path = sys.path[:]
    sys.path.insert(0, rtdetr_code_path)

    try:
        from src.core import YAMLConfig
        abs_config_path = os.path.join(original_cwd, config_path)
        cfg = YAMLConfig(abs_config_path)
        model = cfg.model

        abs_model_path = os.path.join(original_cwd, model_path)
        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"File trọng số không tồn tại: {abs_model_path}")
            
        ckpt = torch.load(abs_model_path, map_location=device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device).eval()
        return model
    finally:
        sys.path[:] = original_sys_path


# ---------------------------------------------------------
# Load a single model (used for real-time)
# ---------------------------------------------------------
def load_single_model(model_name: str):
    """Load one model by display name (for real-time use)."""
    path = MODEL_PATHS.get(model_name)
    if not path:
        raise ValueError(f"❌ Model not found: {model_name}")

    if "YOLO" in model_name:
        print(f"[INFO] Loading YOLO model: {model_name}")
        return YOLO(path).to(device)
    elif "RT-DETR" in model_name:
        print(f"[INFO] Loading RT-DETR model: {model_name}")
        config_path = CONFIG_PATHS.get(model_name)
        return load_rtdetr_model(path, config_path)
    else:
        raise ValueError(f"❌ Unsupported model type: {model_name}")



def get_model_type(model_name: str):
    """Return model type: 'yolo', 'rtdetr', or 'unknown'."""
    if model_name.startswith("YOLO"):
        return "yolo"
    if model_name.startswith("RT-DETR") or model_name.startswith("🚀RT-DETR"):
        return "rtdetr"
    return "unknown"


def get_clean_model_name(model_name: str):
    """Return short ID like yolo_dinov3 or rtdetr_codetr_baseline."""
    for key, value in MODEL_NAMES.items():
        if value == model_name:
            return key
    return None
