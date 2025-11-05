import cv2
import torch
import threading
import queue
import time
import numpy as np
from PIL import Image

# ====== Import your modules ======
from models import load_single_model, run_inference
from utils.camera_utils import manual_roi_setup
from utils.class_remap import CLASS_REMAP_3, get_color_map
from utils.draw_utils import draw_boxes
from utils.inference_utils import check_garbage_placement


# =========================================================
# 🧩 FRAME GRABBER THREAD (Producer)
# =========================================================
class FrameGrabber(threading.Thread):
    def __init__(self, camera_id=0, frame_queue=None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_id)
        self.running = True
        self.frame_queue = frame_queue
        self.fps_limit = 30  # Max 30 FPS
        self.prev_time = 0

        # Optional: set initial window size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Frame rate limiting
            now = time.time()
            if now - self.prev_time < 1.0 / self.fps_limit:
                continue
            self.prev_time = now

            # Push latest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        self.cap.release()

    def stop(self):
        self.running = False


# =========================================================
# 🧠 INFERENCE THREAD (Consumer)
# =========================================================
class InferenceWorker(threading.Thread):
    def __init__(self, model, device, frame_queue, result_queue, color_map, rois):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.color_map = color_map
        self.rois = rois
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Run model inference
            with torch.inference_mode():
                boxes, labels = self.run_model_inference(frame)

            # Check placement correctness
            placements = check_garbage_placement(boxes, labels, self.rois)

            # Draw detections
            annotated = draw_boxes(frame.copy(), boxes, labels, color_map=self.color_map)

            # Draw ROIs
            for roi in self.rois:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), roi["color"], 2)
                cv2.putText(annotated, roi["label"], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi["color"], 2)

            # Overlay correctness
            y_offset = 30
            for fine, broad, region, correct in placements:
                text = f"{fine} → {region or 'None'} [{'OK' if correct else 'WRONG'}]"
                color = (0, 255, 0) if correct else (0, 0, 255)
                cv2.putText(annotated, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            # Push results
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(annotated)

    def run_model_inference(self, frame):
        """Run inference using YOLO or RT-DETR models from model.py"""
        # Convert frame for PIL input if needed
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        outputs = run_inference(self.model, image_pil)

        boxes, labels = [], []
        # YOLO models
        if isinstance(outputs, list):
            results = outputs[0]
            for box in results.boxes:
                conf = box.conf.item()
                if conf > 0.3:
                    coords = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls.item())
                    name = results.names.get(class_id, f"id_{class_id}")
                    labels.append(name)
                    boxes.append(coords)

        # RT-DETR models
        elif isinstance(outputs, dict) and "pred_logits" in outputs:
            raw_logits, pred_boxes = outputs["pred_logits"][0], outputs["pred_boxes"][0]
            pred_probs = raw_logits.sigmoid()
            scores, label_ids = pred_probs.max(dim=-1)
            keep = scores > 0.3
            if torch.sum(keep) > 0:
                boxes_tensor = pred_boxes[keep]
                width, height = frame.shape[1], frame.shape[0]
                for b, s, lid in zip(boxes_tensor, scores[keep], label_ids[keep]):
                    x_c, y_c, w, h = b.cpu().numpy()
                    x1, y1 = (x_c - w / 2) * width, (y_c - h / 2) * height
                    x2, y2 = (x_c + w / 2) * width, (y_c + h / 2) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(f"class_{lid.item()}")

        return boxes, labels

    def stop(self):
        self.running = False


# =========================================================
# 🚀 MAIN
# =========================================================
def main():
    print("♻️ Garbage Detection | Real-time Multi-threaded Test")

    # ---- Load Model ----
    model_name = "YOLO (DINOv3 Baseline)"  # Change if needed
    print(f"[INFO] Loading model: {model_name}")
    model = load_single_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ---- Setup ROIs ----
    print("[INFO] Draw regions for Dry recycle, Organic, Trash.")
    rois = manual_roi_setup()
    if not rois:
        print("❌ No ROIs drawn. Exiting.")
        return

    color_map = get_color_map()

    # ---- Thread Queues ----
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    # ---- Start Threads ----
    grabber = FrameGrabber(0, frame_queue)
    worker = InferenceWorker(model, device, frame_queue, result_queue, color_map, rois)
    grabber.start()
    worker.start()

    print("[INFO] Press 'q' to quit.")
    fps_start = time.time()
    frames = 0

    # ---- Main Display Loop ----
    while True:
        if not result_queue.empty():
            frame = result_queue.get()
            frames += 1

            # Compute FPS
            if frames >= 10:
                fps = frames / (time.time() - fps_start)
                fps_start = time.time()
                frames = 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("♻️ Real-time Garbage Detection", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    grabber.stop()
    worker.stop()
    grabber.join()
    worker.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
