from utils.camera_utils import manual_roi_setup
from utils.inference_utils import run_realtime_detection
import torch
from utils.class_remap import get_color_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("FINAL/YOLO_DINOV3/yolo_checkpoints/yolo11l_finetune_baseline/weights/best.pt", map_location=device)
color_map = get_color_map()

rois = manual_roi_setup()
run_realtime_detection(model, device, color_map, rois)
