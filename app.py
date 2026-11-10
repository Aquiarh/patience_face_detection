import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import glob
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Patience Face Detection & Recognition Portal",
    page_icon="👩🏽‍💻",
    layout="wide"
)

st.title("👩🏽‍💻 Patience Face Detection & Recognition System")
st.write(
    "Automated biometric computer vision portal trained using **Ultralytics YOLO** "
    "for personal face detection and bounding box localization."
)
st.markdown("---")

# 2. Sidebar Controls
st.sidebar.title("Model & Evaluation Controls")
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

# Detect local weights or cloud fallback
yolo_model = None
local_ready = False
weight_file = "exp-2.pt" if os.path.exists("exp-2.pt") else ("best.pt" if os.path.exists("best.pt") else None)

if weight_file:
    try:
        from ultralytics import YOLO
        @st.cache_resource
        def load_yolo(path):
            return YOLO(path)
        yolo_model = load_yolo(weight_file)
        local_ready = True
        st.sidebar.success(f"✅ Local Model Weights Loaded: `{weight_file}`")
    except Exception:
        local_ready = False

if not local_ready:
    st.sidebar.info("🌐 Ultralytics Cloud Run Inference Active")
    CLOUD_RUN_URL = "https://predict-6ae863de0efc9562c8847473-dproatj77a-nw.a.run.app"

# 3. Interactive Tabs
tabs = st.tabs([
    "🎯 Lecturer Evaluation (Preset Samples)",
    "📸 Live Camera & Custom Upload",
    "📊 Model Architecture & Training Summary"
])

# ----------------- TAB 1: DYNAMIC TEST SAMPLES FOR LECTURER -----------------
with tabs[0]:
    st.subheader("Pre-loaded Test Images (Lecturer Verification Gallery)")
    st.info(
        "Select any pre-loaded sample image below to evaluate "
        "bounding box accuracy, target classification, and confidence metrics without uploading local files."
    )

    # Automatically scan for all sample images (sample*.jpg, sample*.jpeg, sample*.png)
    sample_files = sorted(
        glob.glob("sample*.jpg") + glob.glob("sample*.jpeg") + glob.glob("sample*.png")
    )

    if sample_files:
        sample_options = {
            f"Test Sample ({f})": f for f in sample_files
        }
        selected_label = st.selectbox("Select Image to Test:", list(sample_options.keys()))
        selected_file = sample_options[selected_label]
        sample_img = Image.open(selected_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Input Image: `{selected_file}`**")
            st.image(sample_img, use_container_width=True)

        with col2:
            st.markdown("**Inference Output**")
            if st.button("Run Detection on Selected Image", key="btn_sample", use_container_width=True):
                with st.spinner("Executing YOLO detection..."):
                    if local_ready and yolo_model is not None:
                        results = yolo_model.predict(sample_img, conf=conf_threshold)
                        annotated = results[0].plot()
                        st.image(annotated, use_container_width=True)

                        boxes = results[0].boxes
                        st.success(f"Detection Complete! Found: **{len(boxes)} face(s)**")
                        for idx, b in enumerate(boxes):
                            cls_name = results[0].names[int(b.cls[0])]
                            conf_val = float(b.conf[0])
                            st.write(f"- **Detection #{idx+1}:** `{cls_name}` (Confidence: **{conf_val*100:.2f}%**)")
                    else:
                        try:
                            buf = io.BytesIO()
                            sample_img.save(buf, format="JPEG")
                            buf.seek(0)
                            files = {"image": ("sample.jpg", buf.getvalue(), "image/jpeg")}
                            res = requests.post(CLOUD_RUN_URL, files=files, timeout=40)
                            if res.status_code == 200:
                                st.success("Inference successful via Cloud Run endpoint:")
                                st.json(res.json())
                            else:
                                st.warning(f"Cloud endpoint response code: {res.status_code}")
                        except Exception as e:
                            st.error(f"Inference error: {e}")
    else:
        st.warning("No sample files found yet (`sample1.jpg`, `sample2.jpg`, etc.). Drop images into the folder to see them here.")

# ----------------- TAB 2: LIVE UPLOAD / WEBCAM -----------------
with tabs[1]:
    st.subheader("Custom Ingestion (Webcam or File Upload)")
    input_source = st.radio("Choose Input Method:", ["Take Snapshot with Camera", "Upload Custom Image"], horizontal=True)

    test_image = None
    if input_source == "Take Snapshot with Camera":
        test_image = st.camera_input("Take a photo")
    else:
        test_image = st.file_uploader("Upload an image file (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        raw_img = Image.open(test_image).convert("RGB")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Input Snapshot**")
            st.image(raw_img, use_container_width=True)

        with col_b:
            st.markdown("**Detection Output**")
            with st.spinner("Detecting face..."):
                if local_ready and yolo_model is not None:
                    results = yolo_model.predict(raw_img, conf=conf_threshold)
                    annotated = results[0].plot()
                    st.image(annotated, use_container_width=True)

                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for idx, b in enumerate(boxes):
                            cls_name = results[0].names[int(b.cls[0])]
                            conf_val = float(b.conf[0])
                            st.success(f"Detected: **{cls_name}** | Confidence: **{conf_val*100:.2f}%**")
                    else:
                        st.warning("No face detected above the chosen confidence threshold.")
                else:
                    try:
                        buf = io.BytesIO()
                        raw_img.save(buf, format="JPEG")
                        buf.seek(0)
                        files = {"image": ("upload.jpg", buf.getvalue(), "image/jpeg")}
                        res = requests.post(CLOUD_RUN_URL, files=files, timeout=40)
                        if res.status_code == 200:
                            st.json(res.json())
                    except Exception as e:
                        st.error(f"Error: {e}")

# ----------------- TAB 3: TRAINING SUMMARY & SPECS -----------------
with tabs[2]:
    st.subheader("Model Specifications & Architecture")
    st.markdown("""
    - **Target Identity:** Patience Korbla (`patience_face_detection`)
    - **Model Framework:** Ultralytics YOLO Object Detection Architecture
    - **Trained Weights:** `exp-2.pt` / `best.pt`
    - **Task Formulation:** Single-class bounding box regression and facial identification
    - **Evaluation Metrics:**
      - **Precision (P):** Bounding box accuracy of true positive detections
      - **Recall (R):** Coverage of target face appearances across varied lighting and angles
      - **mAP@0.5:** Mean Average Precision calculated at 0.50 Intersection-over-Union (IoU)
    """)