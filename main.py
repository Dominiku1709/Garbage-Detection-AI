# =========================================================
# main_app.py
# =========================================================
import os
import sys
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import page renderers
from apps.image_detection import render_upload_page
from apps.realtime_detection import render_realtime_page

# Streamlit page setup
st.set_page_config(page_title="Garbage Detection System", layout="centered")

# Centered main title
st.markdown("<h1 style='text-align: center;'>♻️ Garbage Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Choose an operation mode below</p>", unsafe_allow_html=True)

# Centered buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    btn1 = st.button("🖼️ Image Detection Mode", use_container_width=True)
    btn2 = st.button("🎥 Real-time Tracking Mode", use_container_width=True)

# Handle redirections
if btn1:
    st.session_state["page"] = "image"
elif btn2:
    st.session_state["page"] = "realtime"

# Page switching logic
if "page" in st.session_state:
    if st.session_state["page"] == "image":
        render_upload_page()
    elif st.session_state["page"] == "realtime":
        render_realtime_page()
