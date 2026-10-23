import os
from pathlib import Path

from utils.class_remap import CLASS_REMAP_6

CATEGORIES = ["Plastic", "Paper", "Metal", "Glass", "Organic waste", "Other waste"]
MODEL_CANDIDATES = (
    Path("FINAL/YOLO_DINOV3/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt"),
    Path("FINAL/YOLO_CODER/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt"),
)

CATEGORY_ALIASES = {
    "plastic": "Plastic",
    "paper": "Paper",
    "metal": "Metal",
    "glass": "Glass",
    "organic": "Organic waste",
    "food waste": "Organic waste",
    "trash": "Other waste",
    "other": "Other waste",
    "other waste": "Other waste",
}


def get_model_path():
    configured = os.environ.get("GARBAGE_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GARBAGE_MODEL_PATH does not point to a file: {path}")
        return str(path)

    root = Path(__file__).resolve().parent.parent
    for candidate in MODEL_CANDIDATES:
        path = root / candidate
        if path.is_file():
            return str(path)
    return None


def load_detector(model_path):
    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to use a YOLO model.") from exc
    try:
        return YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load YOLO checkpoint '{model_path}': {exc}") from exc


def _category_for_label(label):
    normalized = " ".join(str(label).strip().lower().replace("_", " ").split())
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]

    broad_category = CLASS_REMAP_6.get(str(label).strip())
    if broad_category:
        return CATEGORY_ALIASES.get(broad_category.lower(), "Other waste")
    if any(word in normalized for word in ("plastic", "polypropylene", "polystyrene", "styrofoam")):
        return "Plastic"
    if any(word in normalized for word in ("paper", "carton", "cardboard", "tissue")):
        return "Paper"
    if any(word in normalized for word in ("metal", "aluminium", "aluminum", "steel", "can")):
        return "Metal"
    if any(word in normalized for word in ("glass", "jar")):
        return "Glass"
    if any(word in normalized for word in ("food", "organic", "compost")):
        return "Organic waste"
    return "Other waste"


def detect_waste(model, image, confidence_threshold=0.25):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to process images.") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    try:
        prediction = model.predict(
            image,
            conf=confidence_threshold,
            verbose=False,
        )[0]
        names = prediction.names
        boxes = prediction.boxes
        detections = []
        for box in boxes:
            class_id = int(box.cls.item())
            label = names.get(class_id, f"class_{class_id}") if isinstance(names, dict) else names[class_id]
            confidence = float(box.conf.item())
            coordinates = [float(value) for value in box.xyxy[0].tolist()]
            detections.append({
                "category": _category_for_label(label),
                "confidence": confidence,
                "box": coordinates,
                "class_name": str(label),
            })
        return detections
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc
