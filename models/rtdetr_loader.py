# models/rtdetr_loader.py

import os
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_rtdetr_model(model_path: str, config_path: str):
    """
    Loads RT-DETRv2 model from config and checkpoint.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # The RT-DETR code directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rtdetr_dir = os.path.join(PROJECT_ROOT, "rtdetr")
    if not os.path.isdir(rtdetr_dir):
        raise FileNotFoundError(f"RT-DETR code directory missing: {rtdetr_dir}")

    # Temporarily add to sys.path
    original_sys_path = sys.path.copy()
    sys.path.insert(0, rtdetr_dir)

    try:
        from src.core import YAMLConfig

        # Load config
        cfg = YAMLConfig(config_path)
        model = cfg.model

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device).eval()
        print(f"[INFO] RT-DETR model loaded successfully from {model_path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load RT-DETR model: {e}")

    finally:
        sys.path = original_sys_path
