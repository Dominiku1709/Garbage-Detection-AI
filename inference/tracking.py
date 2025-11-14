# =========================================================
# inference/tracking.py
# =========================================================
"""
Real-time detection + object tracking.
Saving happens HERE (because real-time requires cooldown).
"""

import cv2
import time
import numpy as np

from inference.detector import run_inference
from utils.draw import draw_summary
from utils.output_saver import save_detection


# =========================================================
# ⚙️ Simple Centroid Tracker
# =========================================================
class SimpleCentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.max_distance = max_distance

    def update(self, boxes):
        new_objects = {}
        results = []

        for box in boxes:
            if not isinstance(box, (list, np.ndarray)) or len(box) != 4:
                continue

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            min_id, min_dist = None, 1e9
            for oid, (ox, oy) in self.objects.items():
                dist = np.linalg.norm([cx - ox, cy - oy])
                if dist < min_dist and dist < self.max_distance:
                    min_dist, min_id = dist, oid

            if min_id is not None:
                new_objects[min_id] = (cx, cy)
                results.append((min_id, box))
            else:
                new_objects[self.next_id] = (cx, cy)
                results.append((self.next_id, box))
                self.next_id += 1

        self.objects = new_objects
        return results


# =========================================================
# ⚡ Real-time Detection + Tracking
# =========================================================
def run_realtime_tracking(model, color_map, camera_index=0, conf_threshold=0.3):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = SimpleCentroidTracker()

    fps = 0.0
    last_time = time.time()
    total_detections = 0

    # =====================================================
    # Save session folder name (created once)
    # =====================================================
    from utils.output_saver import create_new_session
    SESSION_FOLDER = create_new_session()

    SAVE_COOLDOWN = 3.0     # 3-second cooldown
    LAST_SAVE_TIME = 0

    print("[INFO] Starting real-time detection + tracking. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detections = run_inference(
            model, frame, conf_threshold, color_map
        )

        # Extract raw boxes
        boxes_xyxy = []
        for det in detections:
            if len(det) == 3 and isinstance(det[2], (list, np.ndarray)):
                boxes_xyxy.append(det[2])

        # Tracker update
        tracked = tracker.update(boxes_xyxy)

        # -----------------------------------------------------
        # 🔥 REAL-TIME SAVE LOGIC (with 3s cooldown)
        # -----------------------------------------------------
        now = time.time()
        should_save = (now - LAST_SAVE_TIME >= SAVE_COOLDOWN)

        if should_save:
            for tid, box in tracked:
                # Find matching detection
                match = None
                for det in detections:
                    if np.allclose(det[2], box):
                        match = det
                        break
                if match is None:
                    continue

                cls_name, conf, coords = match

                save_detection(
                    session_folder=SESSION_FOLDER,
                    frame_bgr=frame,
                    cls_name=cls_name,
                    confidence=conf,
                    box_xyxy=coords,
                    model_name=getattr(model, "name", model.__class__.__name__),
                    detection_type="realtime"
                )

            LAST_SAVE_TIME = now  # Reset cooldown

        # -----------------------------------------------------
        # Draw tracks
        # -----------------------------------------------------
        for tid, box in tracked:
            # Find original detection
            match = None
            for det in detections:
                if np.allclose(det[2], box):
                    match = det
                    break

            if match is None:
                continue

            cls_name = match[0]
            color = color_map.get(cls_name, (0, 255, 0))

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                f"{cls_name} #{tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # FPS + Summary
        fps = 1.0 / (time.time() - last_time)
        last_time = time.time()
        total_detections += len(detections)

        draw_summary(annotated_frame, 0, 0, total_detections)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("♻️ Real-time Detection + Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Real-time tracking stopped.")
