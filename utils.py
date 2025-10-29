import cv2
import torch


import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime


# ========================================
# 🔹 DEVICE SETUP
# ========================================
def get_device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================================
# 🔹 IMAGE PREPROCESSING
# ========================================
def preprocess_image(image, size=(640, 640)):
    """
    Prepare image for model inference.
    Accepts PIL or NumPy image, returns Tensor (1, C, H, W).
    """
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


# ========================================
# 🔹 CAMERA HANDLING (OpenCV)
# ========================================
def open_camera(source=0, width=640, height=480):
    """
    Initialize webcam or video source.
    source: int (for webcam) or str (path/URL)
    """
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open video source: {source}")
    return cap


def capture_frame(cap):
    """Read one frame from the camera."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Failed to capture frame from camera.")
    return frame


def release_camera(cap):
    """Release camera resource safely."""
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


# ========================================
# 🔹 SAVE OUTPUTS
# ========================================
def save_result(image, result_dict, output_dir="output"):
    """
    Save image and inference metadata to timestamped folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = os.path.join(output_dir, ts)
    os.makedirs(save_folder, exist_ok=True)

    # Save image
    image_path = os.path.join(save_folder, "image.jpg")
    if isinstance(image, Image.Image):
        image.save(image_path)
    else:
        cv2.imwrite(image_path, image)

    # Save result as text
    result_path = os.path.join(save_folder, "result.txt")
    with open(result_path, "w") as f:
        for k, v in result_dict.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Result saved at: {save_folder}")
    return save_folder


# ========================================
# 🔹 VISUALIZATION
# ========================================
def draw_boxes(image, boxes, labels=None, color=(0, 255, 0)):
    """
    Draw bounding boxes on image (for object detection results).
    boxes: list of [x1, y1, x2, y2]
    labels: list of strings
    """
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if labels:
            cv2.putText(img, labels[i], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img
