# =========================================================
# utils/output_saver.py
# =========================================================
"""
Session-based saving structure using timestamp folders.

output/YYYY-MM-DD_HH-MM/
    detection_0001/
    detection_0002/
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================================================
# Create session folder using timestamp
# =========================================================
def create_new_session():
    """
    Creates: output/YYYY-MM-DD_HH-MM/
    Example: output/2025-11-14_23-41/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_folder = os.path.join(OUTPUT_ROOT, timestamp)

    # auto-increment if folder exists (rare but possible if new session starts within same minute)
    if os.path.exists(session_folder):
        i = 1
        while True:
            alt = f"{timestamp}_{i}"
            alt_path = os.path.join(OUTPUT_ROOT, alt)
            if not os.path.exists(alt_path):
                session_folder = alt_path
                break
            i += 1

    os.makedirs(session_folder, exist_ok=True)
    return session_folder


# =========================================================
# Save one detection inside a session folder
# =========================================================
def save_detection(session_folder, frame_bgr, cls_name, confidence, box_xyxy,
                   model_name="unknown", detection_type="realtime"):
    """
    Creates:
    session_xxxx/detection_yyyy/
        crop.png
        meta.json
    """

    if not isinstance(box_xyxy, (list, tuple, np.ndarray)) or len(box_xyxy) != 4:
        raise ValueError("box_xyxy must be [x1, y1, x2, y2]")

    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = frame_bgr.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # ID auto increment
    existing = [
        d for d in os.listdir(session_folder)
        if d.startswith("detection_") and os.path.isdir(os.path.join(session_folder, d))
    ]

    if not existing:
        det_id = 1
    else:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.replace("detection_", "")))
            except:
                pass
        det_id = max(nums) + 1 if nums else 1

    det_folder = os.path.join(session_folder, f"detection_{det_id:04d}")
    os.makedirs(det_folder, exist_ok=True)

    # Save crop
    crop = frame_bgr[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(det_folder, "crop.png"), crop)

    # Save metadata
    metadata = {
        "class": cls_name,
        "confidence": float(confidence),
        "box_xyxy": [x1, y1, x2, y2],
        "model_name": model_name,
        "detection_type": detection_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(det_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return det_folder
