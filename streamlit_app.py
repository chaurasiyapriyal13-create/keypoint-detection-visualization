"""
streamlit_app.py
-----------------
Web UI for the Keypoint Detection and Visualization System.

Lets users upload an image, choose which detector(s) to run
(Harris / ORB / SIFT), tune a few key parameters, and view the
annotated results, heatmap, and comparison chart right in the browser.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy for free (no install needed for viewers):
    https://share.streamlit.io  -> connect this GitHub repo -> deploy
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detectors import detect_harris, detect_orb, detect_sift
from visualize import annotate, draw_harris_heatmap, side_by_side, summary_chart

st.set_page_config(
    page_title="Keypoint Detection & Visualization",
    page_icon="🔍",
    layout="wide",
)

MAX_DIM = 1200  # downscale very large uploads so the UI stays responsive


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR array, downscaled if huge."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def bgr_to_display(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (Streamlit/PIL display)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------
# Sidebar — inputs & parameters
# --------------------------------------------------------------------------
st.sidebar.title("🔍 Keypoint Detection")
st.sidebar.caption("Harris Corner Detection · ORB · SIFT")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
)

use_sample = st.sidebar.checkbox("Use built-in sample image instead", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.subheader("Detectors to run")
run_harris = st.sidebar.checkbox("Harris Corner Detection", value=True)
run_orb = st.sidebar.checkbox("ORB", value=True)
run_sift = st.sidebar.checkbox("SIFT", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

with st.sidebar.expander("Harris settings", expanded=False):
    harris_block_size = st.slider("Block size", 2, 10, 2)
    harris_ksize = st.select_slider("Sobel aperture (ksize)", options=[3, 5, 7], value=3)
    harris_k = st.slider("Harris free parameter (k)", 0.01, 0.10, 0.04, step=0.01)
    harris_threshold = st.slider("Response threshold ratio", 0.001, 0.10, 0.01, step=0.001)

with st.sidebar.expander("ORB settings", expanded=False):
    orb_n_features = st.slider("Max features", 50, 2000, 500, step=50)

with st.sidebar.expander("SIFT settings", expanded=False):
    sift_contrast = st.slider("Contrast threshold", 0.01, 0.10, 0.04, step=0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Built with OpenCV + Streamlit")

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Keypoint Detection and Visualization System")
st.write(
    "Upload an image and compare how **Harris Corner Detection**, **ORB**, "
    "and **SIFT** identify important / distinctive points in it."
)

# Resolve input image
if uploaded_file is not None and not use_sample:
    image = load_image_from_upload(uploaded_file)
    image_label = uploaded_file.name
else:
    from make_sample_image import generate_scene
    image = generate_scene()
    image_label = "synthetic_scene.png (built-in sample)"

st.image(bgr_to_display(image), caption=f"Input: {image_label}", use_container_width=True)

if not (run_harris or run_orb or run_sift):
    st.info("Select at least one detector in the sidebar to see results.")
    st.stop()

run_button = st.button("Run detection", type="primary")

if run_button or "last_results" in st.session_state:
    if run_button:
        with st.spinner("Running detectors..."):
            results = {}
            if run_harris:
                results["Harris"] = detect_harris(
                    image,
                    block_size=harris_block_size,
                    ksize=harris_ksize,
                    k=harris_k,
                    threshold_ratio=harris_threshold,
                )
            if run_orb:
                results["ORB"] = detect_orb(image, n_features=orb_n_features)
            if run_sift:
                results["SIFT"] = detect_sift(image, contrast_threshold=sift_contrast)
        st.session_state["last_results"] = results
        st.session_state["last_image"] = image
    else:
        results = st.session_state["last_results"]
        image = st.session_state["last_image"]

    # --- Metrics row ---
    st.markdown("### Summary")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        col.metric(label=name, value=f"{res.num_keypoints} keypoints", delta=f"{res.runtime_ms:.1f} ms")

    # --- Tabs per detector + comparison ---
    tab_names = list(results.keys()) + (["Comparison"] if len(results) > 1 else [])
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, results.keys()):
        with tab:
            res = results[name]
            annotated = annotate(image, res)
            st.image(bgr_to_display(annotated), use_container_width=True)

            if name == "Harris":
                st.markdown("**Response heatmap**")
                heatmap = draw_harris_heatmap(image, res)
                st.image(bgr_to_display(heatmap), use_container_width=True)

            st.download_button(
                f"Download {name} result (PNG)",
                data=to_png_bytes(annotated),
                file_name=f"{name.lower()}_keypoints.png",
                mime="image/png",
            )

            if res.descriptors is not None:
                st.caption(f"Descriptor shape: {res.descriptors.shape}")
            else:
                st.caption("Harris has no descriptor vector — keypoint locations only.")

    if len(results) > 1:
        with tabs[-1]:
            panel = side_by_side(image, results, order=[n for n in ("Harris", "ORB", "SIFT") if n in results])
            st.image(bgr_to_display(panel), use_container_width=True)
            st.download_button(
                "Download comparison panel (PNG)",
                data=to_png_bytes(panel),
                file_name="comparison_panel.png",
                mime="image/png",
            )

            chart_path = "/tmp/_streamlit_summary_chart.png"
            summary_chart(results, chart_path)
            st.image(chart_path, use_container_width=True)
else:
    st.caption("Click **Run detection** to see results.")
