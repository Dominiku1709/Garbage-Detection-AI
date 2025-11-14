# ===== run_camera.py =====
import cv2
import torch
import time
import threading
import queue
import numpy as np
from model import load_final_model, run_inference
from utils.class_remap import CLASS_REMAP_3, get_color_map
from utils.draw_utils import draw_boxes, draw_summary

# =========================================================
# 🔹 GLOBAL SETTINGS
# =========================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABELS = ["Dry recycle", "Organic", "Trash"]
COLORS = [(0, 0, 255), (0, 255, 0), (255, 128, 0)]  # BGR colors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 🔹 ROI Drawing & Setup
# =========================================================
def draw_rois(frame, rois):
    for roi in rois:
        color = roi["color"]
        cv2.rectangle(frame, (roi["x"], roi["y"]),
                      (roi["x"] + roi["w"], roi["y"] + roi["h"]),
                      color, 2)
        cv2.putText(frame, roi["label"], (roi["x"], roi["y"] - 5),
                    FONT, 0.7, color, 2)


def manual_roi_setup():
    """Draw 3 trashcan regions manually before starting detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return []

    rois = []
    drawing = False
    start_pt = (0, 0)
    temp_roi = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start_pt, temp_roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            temp_roi = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x0, y0 = start_pt
            temp_roi = {
                "x": min(x, x0),
                "y": min(y, y0),
                "w": abs(x - x0),
                "h": abs(y - y0),
                "label": LABELS[len(rois)] if len(rois) < len(LABELS) else f"ROI {len(rois)+1}",
                "color": COLORS[len(rois) % len(COLORS)]
            }
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if temp_roi and temp_roi["w"] > 20 and temp_roi["h"] > 20:
                rois.append(temp_roi)
                print(f"[INFO] ROI added: {temp_roi['label']}")

    cv2.namedWindow("ROI Setup")
    cv2.setMouseCallback("ROI Setup", mouse_cb)
    print("[INFO] Draw up to 3 ROIs. Press 's' to save, 'r' to reset, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if temp_roi:
            cv2.rectangle(frame, (temp_roi["x"], temp_roi["y"]),
                          (temp_roi["x"] + temp_roi["w"], temp_roi["y"] + temp_roi["h"]),
                          temp_roi["color"], 1)
        draw_rois(frame, rois)

        cv2.putText(frame, "Draw ROIs (s=save/start, r=reset, q=quit)",
                    (10, 30), FONT, 0.6, (255, 255, 255), 1)
        cv2.imshow("ROI Setup", frame)

        key = cv2.waitKey(20) & 0xFF
        if key in [27, ord('q')]:
            rois = []
            break
        elif key == ord('r'):
            rois.clear()
            print("[INFO] ROIs reset.")
        elif key == ord('s'):
            if len(rois) > 0:
                print(f"[INFO] Saved {len(rois)} ROIs.")
                break
            else:
                print("[WARN] No ROIs drawn yet!")

    cap.release()
    cv2.destroyAllWindows()
    return rois


# =========================================================
# 🔹 Placement Validation
# =========================================================
def is_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return roi["x"] <= cx <= roi["x"] + roi["w"] and roi["y"] <= cy <= roi["y"] + roi["h"]


def check_garbage_placement(boxes, labels, rois):
    results = []
    for box, fine_label in zip(boxes, labels):
        broad_label = CLASS_REMAP_3.get(fine_label, "Unknown")
        region, correct = None, False
        for roi in rois:
            if is_inside_roi(box, roi):
                region = roi["label"]
                correct = (
                    (broad_label == "Dry recycle" and region.lower() == "dry recycle")
                    or (broad_label == "Organic" and region.lower() == "organic")
                    or (broad_label == "Trash" and region.lower() == "trash")
                )
                break
        results.append((fine_label, broad_label, region, correct))
    return results


# =========================================================
# 🔹 Inference Thread
# =========================================================
def inference_worker(model, frame_queue, result_queue, color_map):
    """Thread for async model inference to prevent frame lag."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        try:
            outputs = run_inference(model, frame)
            boxes, labels = parse_model_output(outputs, frame)
            result_queue.put((boxes, labels))
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")
            result_queue.put(([], []))


# =========================================================
# 🔹 Parse Model Output
# =========================================================
def parse_model_output(outputs, frame):
    boxes, labels = [], []
    if isinstance(outputs, dict) and "pred_logits" in outputs:
        logits = outputs["pred_logits"][0]
        boxes_pred = outputs["pred_boxes"][0]
        probs = logits.sigmoid()
        scores, classes = probs.max(dim=-1)
        keep = scores > 0.3
        for box, cls, score in zip(boxes_pred[keep], classes[keep], scores[keep]):
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * frame.shape[1])
            y1 = int((cy - h / 2) * frame.shape[0])
            x2 = int((cx + w / 2) * frame.shape[1])
            y2 = int((cy + h / 2) * frame.shape[0])
            boxes.append([x1, y1, x2, y2])
            labels.append(str(int(cls.item())))
    return boxes, labels


# =========================================================
# 🔹 Real-time Detection Main Loop
# =========================================================
def run_realtime_detection():
    print("[INFO] Loading final production model...")
    model = load_final_model()
    color_map = get_color_map()

    rois = manual_roi_setup()
    if not rois:
        print("[WARN] No ROIs defined, exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    # Set camera window
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("♻️ Real-time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("♻️ Real-time Detection", 960, 540)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    # Start inference thread
    threading.Thread(target=inference_worker,
                     args=(model, frame_queue, result_queue, color_map),
                     daemon=True).start()

    correct = wrong = total = 0
    prev_time = 0
    print("[INFO] Press 'Q' to quit | 'R' to reset counters")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - prev_time < 1 / 30:
            continue
        prev_time = now

        # Send frame for inference
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Retrieve results
        boxes, labels = ([], [])
        if not result_queue.empty():
            boxes, labels = result_queue.get()

        # Draw detections and ROIs
        frame = draw_boxes(frame, boxes, labels, color_map)
        draw_rois(frame, rois)

        # Check placement correctness
        results = check_garbage_placement(boxes, labels, rois)
        for (_, _, region, ok) in results:
            if region:
                total += 1
                if ok:
                    correct += 1
                else:
                    wrong += 1

        draw_summary(frame, correct, wrong, total)
        cv2.putText(frame, "Model: 🚀RT-DETR (Distill CodeTR)", (20, 40),
                    FONT, 0.7, (255, 255, 255), 2)
        cv2.imshow("♻️ Real-time Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key == ord('r'):
            correct = wrong = total = 0
            print("[INFO] Counters reset.")

    frame_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# 🔹 ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_realtime_detection()
