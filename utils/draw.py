# =========================================================
# utils/draw.py
# =========================================================
import cv2
import numpy as np

# Define some common colors (BGR format)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DEFAULT = (0, 0, 255) # Red for "Unknown"

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LINE_THICKNESS = 2

def draw_boxes(image: np.ndarray, boxes: list, labels: list, color_map: dict) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image.
    
    Args:
        image: The image to draw on (BGR NumPy array).
        boxes: A list of bounding boxes, each as [x1, y1, x2, y2].
        labels: A list of label strings (e.g., "Plastic: 0.85").
        color_map: A dictionary mapping class_name -> [B, G, R] color.
        
    Returns:
        A new image (copy) with annotations.
    """
    # Create a copy to avoid modifying the original image
    img_out = image.copy()
    
    for box, label in zip(boxes, labels):
        # --- Get Color ---
        # Extract the class name (e.g., "Plastic") from the label ("Plastic: 0.85")
        class_name = label.split(':')[0]
        color = color_map.get(class_name, COLOR_DEFAULT)
        
        # --- Draw Bounding Box ---
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        
        # --- Draw Label Background and Text ---
        
        # Get text size to draw a background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        
        # Set text position (default: just above the box)
        text_y = y1 - LINE_THICKNESS - baseline
        
        # Calculate background rectangle coordinates
        bg_y1 = text_y - text_h - 2
        bg_y2 = text_y + baseline + 2
        bg_x2 = x1 + text_w
        
        # --- Handle text going off-screen (top) ---
        # If the background box would go above the image (y < 0),
        # move the text to be *inside* the bounding box at the top.
        if bg_y1 < 0:
            text_y = y1 + text_h + 2 # Place text y-coordinate inside box
            bg_y1 = y1 + 2
            bg_y2 = y1 + text_h + baseline + 2
            
        # Draw the filled background rectangle
        cv2.rectangle(img_out, (x1, bg_y1), (bg_x2, bg_y2), color, -1) # -1 thickness = filled
        
        # Draw the text on top of the background
        cv2.putText(img_out, label, (x1, text_y), FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
        
    return img_out


def draw_summary(frame: np.ndarray, arg1, arg2, total_detections: int):
    """
    Draws the total detection count on the frame (modifies in-place).
    
    NOTE: The signature (arg1, arg2) is kept to match the unusual
    call `draw_summary(frame, 0, 0, total_detections)` in inference/tracking.py.
    The arguments `arg1` and `arg2` are unused.
    
    This function draws text *below* the FPS counter, which tracking.py
    handles separately.
    """
    # This text is drawn at (20, 70)
    text = f"Total Detections: {total_detections}"
    
    # This position is chosen to be just below the FPS text
    # which inference/tracking.py draws at (20, 40).
    pos = (20, 70) 
    
    # Use the same style as the FPS counter for consistency
    color = (255, 255, 0) # Yellow (BGR)
    thickness = 2
    font_scale = 0.7
    
    cv2.putText(frame, text, pos, FONT, font_scale, color, thickness)