import torch
from ultralytics import YOLO
from lightly_train.model_wrappers import RTDETRModelWrapper

# =========================================================
# 📦 MODEL PATHS
# =========================================================
MODEL_PATHS = {
    "Rtdetrv2": "FINAL/FINETUNE_BASELINE/rtdetrv2_finetune_taco_BASELINE/last.pth",
    "Distill-Convnet": "FINAL/FINETUNE_DISTILLED/rtdetrv2_finetune_taco_convnext_teacher/last.pth",
    "Distill-Vit": "FINAL/FINETUNE_DISTILLED/rtdetrv2_finetune_taco_vit_teacher/best.pth",
    "YOLOv11l": "FINAL/YOLO/yolo11n.pt"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 🧩 1️⃣ YOLO LOADER
# =========================================================
def load_YOLO():
    """Load YOLOv11 model."""
    model = YOLO(MODEL_PATHS["YOLOv11l"])
    model.to(device)
    return model


# =========================================================
# 🧠 2️⃣ RT-DETRv2 LOAD HELPER
# =========================================================
def _load_rtdetr_model(model_path: str):
    """Generic RT-DETRv2 model loader with fallback for checkpoints."""
    base_model = torch.hub.load('lyuwenyu/RT-DETR', 'rtdetrv2_l', pretrained=False, trust_repo=True)
    model = RTDETRModelWrapper(base_model.model)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


# =========================================================
# 🧱 3️⃣ INDIVIDUAL LOADERS
# =========================================================
def load_Rtdetrv2():
    """Load baseline RT-DETRv2 model."""
    return _load_rtdetr_model(MODEL_PATHS["Rtdetrv2"])


def load_DistillConv():
    """Load RT-DETRv2 distilled from ConvNeXt teacher."""
    return _load_rtdetr_model(MODEL_PATHS["Distill-Convnet"])


def load_DistillVit():
    """Load RT-DETRv2 distilled from ViT teacher."""
    return _load_rtdetr_model(MODEL_PATHS["Distill-Vit"])


# =========================================================
# 🚀 4️⃣ INFERENCE FUNCTION
# =========================================================
@torch.inference_mode()
def run_inference(model, image):
    """
    Run inference on a single image (PIL or tensor).
    Returns model predictions.
    """
    if isinstance(model, YOLO):
        # YOLO inference
        results = model.predict(image, device=device)
        return results

    elif isinstance(model, RTDETRModelWrapper):
        # RT-DETR inference
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms
            image = transforms.ToTensor()(image).unsqueeze(0).to(device)
        else:
            image = image.unsqueeze(0) if image.ndim == 3 else image.to(device)

        outputs = model(image)
        return outputs

    else:
        raise TypeError("Unsupported model type for inference.")