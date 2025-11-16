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

from utils.output_saver import create_new_session, save_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving state
SESSION_FOLDER = create_new_session()
LAST_SAVE_TIME = 0
SAVE_COOLDOWN = 3  # seconds


def preprocess_image(image):
    """Converts numpy -> PIL, keeps PIL unchanged."""
    if isinstance(image, np.ndarray):
        # BGR (OpenCV) → RGB (PIL)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


@torch.inference_mode()
def run_inference(model, image, conf_threshold=0.25, color_map=None):
    global LAST_SAVE_TIME

    image_pil = preprocess_image(image)   # Image now PIL
    model_name = getattr(model, "name", model.__class__.__name__)

    boxes_to_draw, labels_to_draw, detections = [], [], []

    
    # YOLO MODE (Ultralytics)
    
    if hasattr(model, "predict"):
        results = model.predict(image_pil, device=device, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < conf_threshold:
                continue

            coords = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls.item())

            fine_name = results.names.get(cls_id, f"class_{cls_id}")
            broad_name = CLASS_REMAP_6.get(fine_name, "Unknown")

            detections.append((broad_name, conf, coords))
            boxes_to_draw.append(coords)
            labels_to_draw.append(f"{broad_name} {conf:.2f}")

            # Save detection (image type)
            if time.time() - LAST_SAVE_TIME >= SAVE_COOLDOWN:
                save_detection(
                    SESSION_FOLDER,
                    frame_bgr=image.copy(),
                    cls_name=broad_name,
                    confidence=conf,
                    box_xyxy=coords,
                    model_name=model_name,
                    detection_type="image"
                )
                LAST_SAVE_TIME = time.time()

    
    # RT-DETR MODE (torch)
    
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        outputs = model(input_tensor)

        # RT-DETR dictionary output style
        if isinstance(outputs, dict) and "pred_logits" in outputs:
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]

            pred_probs = pred_logits.sigmoid()
            scores, labels = pred_probs.max(dim=-1)

            # Threshold filter
            keep = scores > conf_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_boxes[keep]

            if len(boxes) > 0:
                final_boxes = box_cxcywh_to_xyxy(
                    boxes, image_pil.width, image_pil.height
                ).to(device)

                keep_idx = nms(final_boxes, scores, 0.5)

                for idx in keep_idx:
                    conf = float(scores[idx].item())
                    class_id = int(labels[idx].item())
                    fine_name = TACO_CLASS_NAMES.get(class_id, f"class_{class_id}")
                    broad_name = CLASS_REMAP_6.get(fine_name, "Unknown")
                    coords = final_boxes[idx].cpu().numpy().tolist()

                    detections.append((broad_name, conf, coords))
                    boxes_to_draw.append(coords)
                    labels_to_draw.append(f"{broad_name} {conf:.2f}")

    # Unknown model case
    else:
        print("[ERROR] ❌ Unsupported model type detected.")
        return image, []

    # ===============================
    # Final drawing and return
    # ===============================
    detections = [
        d for d in detections
        if len(d) == 3 and isinstance(d[2], (list, np.ndarray)) and len(d[2]) == 4
    ]

    annotated = draw_boxes(
        cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR),
        boxes_to_draw,
        labels_to_draw,
        color_map
    )

    return annotated, detections
