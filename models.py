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

MODEL_PATHS = {
    "Rtdetrv2": os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_BASELINE/rtdetrv2_finetune_taco_BASELINE/best.pth"),
    "Distill-Convnet": os.path.join(PROJECT_ROOT, "FINAL/DISTILL-CONVNEXT/distilled_rtdetr_convnext_teacher_BEST.pth"),
    "Distill-Vit": os.path.join(PROJECT_ROOT, "FINAL/FINETUNE_DISTILLED/rtdetrv2_finetune_taco_vit_teacher/best.pth"),
    "YOLOv11l": os.path.join(PROJECT_ROOT, "FINAL/YOLO/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt")
}

CONFIG_PATHS = {
    "Rtdetrv2": os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_BASELINE.yml"),
    "Distill-Convnet": os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_convnext.yml"),
    "Distill-Vit": os.path.join(PROJECT_ROOT, "FINAL/CONFIG/rtdetrv2_taco_finetune_vit.yml"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# YOLO LOADER
# =========================================================
def load_YOLO():
    model = YOLO(MODEL_PATHS["YOLOv11l"])
    model.to(device)
    return model

# =========================================================
# RT-DETR LOAD HELPER
# =========================================================
def _load_rtdetrv2_model_from_config(model_path: str, config_path: str):
    original_cwd = os.getcwd()
    rtdetr_code_path = os.path.join(PROJECT_ROOT, "rtdetr")

    if not os.path.isdir(rtdetr_code_path):
        raise FileNotFoundError(
            f"Thư mục 'rtdetr' không được tìm thấy tại gốc dự án: {rtdetr_code_path}"
        )

    try:
        os.chdir(rtdetr_code_path)
        from src.core import YAMLConfig

        abs_config_path = os.path.join(original_cwd, config_path)
        cfg = YAMLConfig(abs_config_path)
        model = cfg.model

        abs_model_path = os.path.join(original_cwd, model_path)
        ckpt = torch.load(abs_model_path, map_location=device)

        if 'model' in ckpt: state_dict = ckpt['model']
        elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        else: state_dict = ckpt
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device).eval()
        
        return model

    finally:
        os.chdir(original_cwd)

# =========================================================
# INDIVIDUAL LOADERS
# =========================================================
def load_Rtdetrv2():
    return _load_rtdetrv2_model_from_config(MODEL_PATHS["Rtdetrv2"], CONFIG_PATHS["Rtdetrv2"])
def load_DistillConv():
    return _load_rtdetrv2_model_from_config(MODEL_PATHS["Distill-Convnet"], CONFIG_PATHS["Distill-Convnet"])
def load_DistillVit():
    return _load_rtdetrv2_model_from_config(MODEL_PATHS["Distill-Vit"], CONFIG_PATHS["Distill-Vit"])

# =========================================================
# INFERENCE FUNCTION (KHÔI PHỤC NORMALIZE)
# =========================================================
@torch.inference_mode()
def run_inference(model, image):
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
             image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
             
        image_tensor = transform(image).unsqueeze(0).to(device)
        return model(image_tensor)