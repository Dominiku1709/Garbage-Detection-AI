import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from services.classifier import CATEGORIES, detect_waste, get_model_path, load_detector
from services.history import load_history, save_analysis, summarize_history

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

RECOMMENDATIONS = {
    "Plastic": "Empty and rinse containers. Check local guidance for films and mixed plastics before recycling.",
    "Paper": "Keep paper and cardboard clean and dry. Greasy or food-soiled paper usually belongs in organics or residual waste.",
    "Metal": "Empty and rinse cans and foil. Place accepted metal packaging in your local recycling stream.",
    "Glass": "Empty containers and separate by local color rules. Wrap broken glass safely; do not place it loose in recycling.",
    "Organic waste": "Put food scraps in a local organics or compost collection. Keep packaging out of the bin.",
    "Other waste": "Check local disposal guidance. Batteries, electronics, and hazardous items need dedicated collection.",
}

CATEGORY_COLORS = {
    "Plastic": "#0f9b74",
    "Paper": "#d59b42",
    "Metal": "#70859a",
    "Glass": "#4d9cb5",
    "Organic waste": "#71a65b",
    "Other waste": "#9a8d7e",
}


@st.cache_resource(show_spinner=False)
def _cached_detector(model_path):
    return load_detector(model_path)


def _load_uploaded_image(uploaded_file):
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("The selected file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Images must be 10 MB or smaller.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Please upload a JPEG, PNG, or WebP image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image is too large to process (maximum 40 megapixels).")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB"), data
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be decoded as a valid image.") from exc


def _annotate_image(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for item in detections:
        box = item.get("box")
        if not box:
            continue
        color = CATEGORY_COLORS[item["category"]]
        x1, y1, x2, y2 = (int(value) for value in box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        confidence = item.get("confidence")
        label = item["category"]
        if confidence is not None:
            label = f"{label}  {confidence:.0%}"
        bounds = draw.textbbox((x1, max(0, y1 - 24)), label, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((x1, max(0, y1 - 24)), label, fill="white", font=font)
    return annotated


def _render_sidebar():
    st.sidebar.markdown("## ♻️ SortWise")
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Analyze image", "Detection history", "Recycling guide", "Live camera"],
        label_visibility="collapsed",
    )
    try:
        model_path = get_model_path()
    except FileNotFoundError as exc:
        st.session_state["model_configuration_error"] = str(exc)
        st.sidebar.error(str(exc))
    else:
        st.session_state.pop("model_configuration_error", None)
        if model_path:
            st.sidebar.success(f"AI model configured: `{Path(model_path).name}`")
        else:
            st.sidebar.info("No trained checkpoint found. Image analysis will use manual review.")
    st.sidebar.caption("Images are processed locally and are not saved to history.")
    return page


def _render_overview():
    history = load_history()
    summary = summarize_history(history)
    st.title("Waste management dashboard")
    st.write("A clear picture of what you've sorted and where it can go next.")

    metrics = st.columns(4)
    metrics[0].metric("Images analyzed", summary["analyses"])
    metrics[1].metric("Items classified", summary["detections"])
    metrics[2].metric("Recyclable items", summary["recyclable"])
    metrics[3].metric("Recycling share", f'{summary["recycling_rate"]:.0f}%')

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Waste by category")
        st.bar_chart(
            pd.DataFrame(
                {"Items": [summary["categories"][category] for category in CATEGORIES]},
                index=CATEGORIES,
            ),
            color="#11866f",
        )
    with right:
        st.subheader("Latest analyses")
        if history:
            for record in history[:5]:
                when = record["analyzed_at"].replace("T", " ")
                st.markdown(
                    f"**{record['filename']}**  \n"
                    f"{when} · {record['total_detections']} item(s) · {record['model_name']}"
                )
                st.divider()
        else:
            st.caption("Your image analyses will appear here.")

    st.subheader("Quick sorting guide")
    guide_cols = st.columns(3)
    for index, category in enumerate(CATEGORIES):
        with guide_cols[index % 3]:
            st.markdown(f"**{category}**")
            st.caption(RECOMMENDATIONS[category])


def _render_analyzer():
    st.title("Analyze an image")
    st.write("Upload a clear image of waste to classify it and get a sorting recommendation.")

    try:
        model_path = get_model_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Correct `GARBAGE_MODEL_PATH` or remove it to use the manual-review fallback.")
        return
    if model_path:
        st.success("A local trained YOLO checkpoint is configured. Inference runs on this device.")
    else:
        st.warning(
            "No trained checkpoint is available in this checkout. The offline fallback records "
            "a category you select; it does not claim to perform AI detection or provide an AI confidence score. "
            "See the README to configure your own trained checkpoint."
        )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPEG, PNG, or WebP; maximum 10 MB.",
    )
    if uploaded_file is None:
        return

    try:
        image, image_bytes = _load_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    fallback_category = None
    if not model_path:
        fallback_category = st.selectbox(
            "Choose the category after reviewing the image",
            CATEGORIES,
            help="This is a manual classification, not an AI prediction.",
        )
    else:
        confidence_threshold = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

    if st.button("Classify waste", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                if model_path:
                    model = _cached_detector(model_path)
                    detections = detect_waste(model, image, confidence_threshold)
                    model_name = f"YOLO · {Path(model_path).name}"
                    if not detections:
                        st.info("No waste was detected above the model's confidence threshold.")
                else:
                    detections = [{
                        "category": fallback_category,
                        "confidence": None,
                        "box": None,
                        "class_name": "Manual review",
                    }]
                    model_name = "Manual review (offline fallback)"

            saved_id = save_analysis(
                filename=uploaded_file.name,
                model_name=model_name,
                detections=detections,
            )
            st.session_state["latest_analysis"] = {
                "image_hash": image_hash,
                "image": _annotate_image(image, detections),
                "detections": detections,
                "model_name": model_name,
                "saved_id": saved_id,
            }
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
            return

    result = st.session_state.get("latest_analysis")
    if not result or result["image_hash"] != image_hash:
        return

    st.subheader("Detection results")
    if result["detections"]:
        st.image(result["image"], caption="Analyzed image", use_container_width=True)
        result_columns = st.columns(min(3, len(result["detections"])))
        for index, item in enumerate(result["detections"]):
            with result_columns[index % len(result_columns)]:
                st.markdown(f"### {item['category']}")
                if item["confidence"] is None:
                    st.caption("AI confidence: not available (manual review)")
                else:
                    st.metric("Confidence", f"{item['confidence']:.1%}")
                st.write(RECOMMENDATIONS[item["category"]])
                if item.get("class_name") not in {None, item["category"], "Manual review"}:
                    st.caption(f"Model label: {item['class_name']}")
    else:
        st.caption("No detections to display. Try a clearer image or a lower confidence threshold in your model.")
    st.caption(f"Saved to local history · {result['model_name']}")


def _render_history():
    st.title("Detection history")
    history = load_history()
    if not history:
        st.info("No analyses yet. Upload an image to get started.")
        return

    rows = []
    for record in history:
        categories = ", ".join(
            f"{category} ({count})" for category, count in record["category_counts"].items()
        ) or "No detections"
        rows.append({
            "Analyzed at": record["analyzed_at"].replace("T", " "),
            "Image": record["filename"],
            "Items": record["total_detections"],
            "Categories": categories,
            "Method": record["model_name"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Only filenames and classification results are stored; uploaded image files are not retained.")


def _render_guide():
    st.title("Recycling guide")
    st.write("Local recycling rules vary. Use these general tips and check your municipality's guidance.")
    for category in CATEGORIES:
        with st.container(border=True):
            st.markdown(f"### {category}")
            st.write(RECOMMENDATIONS[category])


def _render_live_camera():
    st.title("Live camera")
    st.info("Live camera detection remains available through the existing camera workflow.")
    try:
        from apps.realtime_detection import render_realtime_page

        render_realtime_page()
    except Exception as exc:
        st.error(f"The live camera workflow could not be loaded: {exc}")


def main():
    st.set_page_config(page_title="SortWise | Waste management", page_icon="♻️", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {background: #f3f7f5; border: 1px solid #e3ece7; padding: 1rem; border-radius: 14px;}
        [data-testid="stSidebar"] {background: #f6f8f7;}
        @media (max-width: 700px) {
          .block-container {padding: 1rem 1rem 2rem;}
          [data-testid="stMetric"] {padding: .7rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page = _render_sidebar()
    if page == "Overview":
        _render_overview()
    elif page == "Analyze image":
        _render_analyzer()
    elif page == "Detection history":
        _render_history()
    elif page == "Recycling guide":
        _render_guide()
    else:
        _render_live_camera()


if __name__ == "__main__":
    main()
