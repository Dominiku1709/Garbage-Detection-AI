import streamlit as st
import torch
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# IMPORT UTILITIES AND MODELS
from utils.class_remap import get_color_map
from models.loader import MODEL_NAMES, load_single_model
from inference.tracking import run_realtime_tracking # Use the dedicated tracking script

# CONFIGURATION
st.set_page_config(page_title="Real-time Tracking", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource

def single_model_cache():
    return load_single_model()

def cached_load_single_model(model_name):
    """Cache single model for real-time detection."""
    st.toast(f"Loading {model_name}...")
    try:
        model = load_single_model(model_name)
        st.toast("✅ Model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None

# REAL-TIME LOCAL CAMERA DETECTION PAGE
def render_realtime_page():
    st.header("🎥 Real-time Local Camera Tracking")
    st.info(
        "This app will use Streamlit to select a model, then "
        "launch a **separate OpenCV window** for real-time tracking."
    )

    # Invert dictionary to map Display Name -> Key
    display_to_key = {v: k for k, v in MODEL_NAMES.items()}
    display_names = sorted(list(display_to_key.keys()))

    selected_display_name = st.selectbox("Select a model", display_names, index=len(display_names)-1)
    
    # Find the original key (e.g., "rtdetr_codetr_distilled")
    model_key = display_to_key[selected_display_name]
    # Get the display name (which is what load_single_model expects)
    model_name = MODEL_NAMES[model_key] 

    confidence_threshold = st.slider(
        "Confidence Threshold (for live detection)",
        0.0, 1.0, 0.3, 0.05
    )
    
    camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)

    if st.button("▶️ Start Local Camera Feed", type="primary", use_container_width=True):
        
        model = cached_load_single_model(model_name)
        
        if model is not None:
            color_map = get_color_map() # Get 6-class color map
            
            st.info("Starting real-time tracking... Check for the 'OpenCV' window.")
            st.info("Press 'q' or 'Esc' in the OpenCV window to quit.")
            
            # This function is imported from inference/tracking.py
            # It contains the full OpenCV loop.
            try:
                run_realtime_tracking(
                    model, 
                    color_map, 
                
                    int(camera_index), 
                    confidence_threshold
                )
                st.success("Real-time tracking session finished.")
            except Exception as e:
                st.error(f"An error occurred during tracking: {e}")
        else:
            st.error("Cannot start tracking: Model was not loaded.")


# MAIN
if __name__ == "__main__":
    st.sidebar.title("Real-time App")
    st.sidebar.info(
        "This app launches a real-time OpenCV tracking session. "
        "Run `run_image_app.py` for static image analysis."
    )
    render_realtime_page()