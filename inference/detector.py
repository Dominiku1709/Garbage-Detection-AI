# inference/detector.py
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import time
from torchvision.ops import nms


from utils.class_remap import TACO_CLASS_NAMES, CLASS_REMAP_6
from utils.draw import draw_boxes
from utils.utils import box_cxcywh_to_xyxy

# NEW: saving utilities
from utils.output_saver import create_new_session, save_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GLOBAL state for saving
SESSION_FOLDER = create_new_session()
LAST_SAVE_TIME = 0          # global cooldown
SAVE_COOLDOWN = 3           # seconds


# Preprocessing helper
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


# Unified inference interface + SAVING LOGIC
@torch.inference_mode()
def run_inference(model, image, conf_threshold=0.25, color_map=None):
    global LAST_SAVE_TIME

    image_pil = preprocess_image(image)
    boxes_to_draw, labels_to_draw, detections = [], [], []

    # Detect model type / name
    model_name = getattr(model, "name", model.__class__.__name__)

    # YOLO MODELS
    if hasattr(model, "predict"):
        results = model.predict(image_pil, device=device, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf.item())
            if conf >= conf_threshold:
                coords = box.xyxy[0].detach().cpu().numpy().tolist()
                cls_id = int(box.cls.item())
                fine_name = results.names.get(cls_id, f"class_{cls_id}")
                broad_name = CLASS_REMAP_6.get(fine_name, "Unknown")

                detections.append((broad_name, conf, coords))
                boxes_to_draw.append(coords)
                labels_to_draw.append(f"{broad_name} {conf:.2f}")

                # SAVE DETECTION (3-sec cooldown)
                if time.time() - LAST_SAVE_TIME >= SAVE_COOLDOWN:
                    save_detection(
                        session_folder=SESSION_FOLDER,
                        frame_bgr=image.copy(),
                        cls_name=broad_name,
                        confidence=conf,
                        box_xyxy=coords,
                        model_name=model_name,
                        detection_type="image"
                    )
                    LAST_SAVE_TIME = time.time()

    
    # RT-DETR MODELS
    
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        outputs = model(image_tensor)

        if isinstance(outputs, dict) and "pred_logits" in outputs:
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]

            pred_probs = pred_logits.sigmoid()
            scores, labels = pred_probs.max(dim=-1)

            # Step 1: filter by confidence
            keep = scores > conf_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_boxes[keep]

            if len(boxes) > 0:
                # Convert to xyxy
                final_boxes = box_cxcywh_to_xyxy(
                    boxes, image_pil.width, image_pil.height
                ).to(device)

                # Step 2: Apply NMS
                keep_indices = nms(final_boxes, scores, iou_threshold=0.5)

                for idx in keep_indices:
                    conf = float(scores[idx].item())
                    class_id = int(labels[idx].item())
                    fine_name = TACO_CLASS_NAMES.get(class_id, f"class_{class_id}")
                    broad_name = CLASS_REMAP_6.get(fine_name, "Unknown")

                    coords = final_boxes[idx].detach().cpu().numpy().tolist()

                    detections.append((broad_name, conf, coords))
                    boxes_to_draw.append(coords)
                    labels_to_draw.append(f"{broad_name} {conf:.2f}")


    # Validation & Drawing
    detections = [
        d for d in detections
        if len(d) == 3 and isinstance(d[2], (list, np.ndarray)) and len(d[2]) == 4
    ]

    annotated_image = draw_boxes(
        cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR),
        boxes_to_draw,
        labels_to_draw,
        color_map
    )

    return annotated_image, detections
