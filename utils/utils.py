# utils\utils.py 
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from datetime import datetime
import random


# DEVICE SETUP

def get_device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  IMAGE PREPROCESSING
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


# CAMERA HANDLING (OpenCV)
def open_camera(source=0, width=1800, height=1000):
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
        return None # Return None if stream ends or fails
    return frame


def release_camera(cap):
    """Release camera resource safely."""
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


#  SAVE OUTPUTS
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
        # Ensure image is in BGR format for cv2.imwrite
        if image.ndim == 3 and image.shape[2] == 3:
             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)


    # Save result as text
    result_path = os.path.join(save_folder, "result.txt")
    with open(result_path, "w") as f:
        f.write(str(result_dict))

    print(f"✅ Result saved at: {save_folder}")
    return save_folder

# VISUALIZATION
def box_cxcywh_to_xyxy(x, width, height):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b = torch.stack(b, dim=1)
    b *= torch.tensor([width, height, width, height], dtype=torch.float32, device=x.device)
    return b

def draw_boxes(image_np, boxes, labels=None, color_map=None):
    """
    Draw bounding boxes on a NumPy image.
    boxes: list of [x1, y1, x2, y2]
    labels: list of strings
    color_map: dict mapping class names to colors
    """
    if color_map is None:
        color_map = {}

    img = image_np.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label_text = labels[i] if labels else ""
        class_name = label_text.split(':')[0] # Get class name from "name: conf"
        
        # Get color from map or generate a random one
        color = color_map.get(class_name, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img