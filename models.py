# ===== .\models.py =====
import torch
from ultralytics import YOLO
import sys
import os
from PIL import Image
import numpy as np
import cv2

# =========================================================
# MODEL AND CONFIG PATHS
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
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


# =========================================================
# RT-DETR LOAD HELPER (Không thay đổi)
# =========================================================
def _load_rtdetrv2_model_from_config(model_path: str, config_path: str):
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
        
# =========================================================
# HÀM TẢI CHUNG (Sửa để trả về lỗi thay vì gọi st.error)
# =========================================================
def load_all_models():
    """Tải tất cả các mô hình, trả về cả các model đã tải và thông tin lỗi."""
    models = {}
    errors = {}
    for name, path in MODEL_PATHS.items():
        print(f"Loading {name}...")
        try:
            if "YOLO" in name:
                models[name] = YOLO(path).to(device)
            elif "RT-DETR" in name:
                models[name] = _load_rtdetrv2_model_from_config(path, CONFIG_PATHS[name])
            else:
                errors[name] = "Unknown model type. Skipped."
        except Exception as e:
            # Ghi lại lỗi thay vì crash app
            error_message = f"Error: {e}"
            print(f"❌ Error loading model '{name}': {error_message}")
            errors[name] = error_message
    return models, errors

def load_single_model(model_name: str):
    """Tải một mô hình duy nhất theo tên."""
    # ... (giữ nguyên code của hàm này)
    path = MODEL_PATHS.get(model_name)
    if not path:
        raise ValueError(f"Model '{model_name}' not found in MODEL_PATHS.")

    if "YOLO" in model_name:
        return YOLO(path).to(device)
    elif "RT-DETR" in model_name:
        config_path = CONFIG_PATHS.get(model_name)
        if not config_path:
            raise ValueError(f"Config path for '{model_name}' not found.")
        return _load_rtdetrv2_model_from_config(path, config_path)
    raise TypeError(f"Unknown model type for '{model_name}'.")


# =========================================================
# LOAD CHOSEN MODEL(load model được chọn)
# =========================================================
def load_final_model():
    """
    Load the final production model: 🚀RT-DETR (Distill CodeTR)

    Returns:
        model (torch.nn.Module): Loaded and ready-to-run model
    Raises:
        FileNotFoundError, ValueError, RuntimeError
    """
    final_model_name = "🚀RT-DETR (Distill CodeTR)"

    print("\n==============================")
    print("🚀 LOADING FINAL PRODUCTION MODEL")
    print("==============================")

    model_path = MODEL_PATHS.get(final_model_name)
    config_path = CONFIG_PATHS.get(final_model_name)

    if not model_path:
        raise ValueError(f"Model path not defined for '{final_model_name}'")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file not found at: {model_path}")
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Config path: {config_path}")

    try:
        model = _load_rtdetrv2_model_from_config(model_path, config_path)
        print(f"[✅] Successfully loaded final model: {final_model_name}")
        print("==============================\n")
        return model
    except Exception as e:
        print(f"Failed to load final model: {e}")
        raise RuntimeError(f"Error loading final model: {e}")


# =========================================================
# INFERENCE FUNCTION (Không thay đổi)
# =========================================================
@torch.inference_mode()
def run_inference(model, image):
    # ... (giữ nguyên code của hàm này)
    if isinstance(model, YOLO):
        return model.predict(image, device=device, verbose=False)
    else:
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if not isinstance(image, Image.Image):
             image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR_RGB))
             
        image_tensor = transform(image).unsqueeze(0).to(device)
        return model(image_tensor)