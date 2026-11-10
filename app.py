import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import glob

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
    value=0.40,
    step=0.05
)

# 3. Model Loading (Ultralytics Native)
@st.cache_resource
def load_yolo_model():
    model_path = None
    if os.path.exists("best.pt"):
        model_path = "best.pt"
    elif os.path.exists("exp-2.pt"):
        model_path = "exp-2.pt"

    if model_path is None:
        return None, "Weights file ('best.pt' or 'exp-2.pt') not found in repository."
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, model_path
    except Exception as e:
        return None, str(e)

yolo_model, model_status = load_yolo_model()

if yolo_model is not None:
    st.sidebar.success(f"✅ Active Model: `{model_status}`")
else:
    st.sidebar.error(f"⚠️ Model load failure: {model_status}")

# 4. Interactive Tabs
tabs = st.tabs([
    "🎯 Lecturer Evaluation (Preset Samples)",
    "📸 Live Camera & Custom Upload",
    "📊 Model Architecture & Training Summary"
])

# ----------------- TAB 1: DYNAMIC TEST SAMPLES -----------------
with tabs[0]:
    st.subheader("Pre-loaded Test Images (Lecturer Verification Gallery)")
    st.info(
        "Select any pre-loaded sample image below to evaluate "
        "bounding box accuracy, target classification, and confidence scores directly in the browser."
    )

    sample_files = sorted(
        glob.glob("sample*.jpg") + glob.glob("sample*.jpeg") + glob.glob("sample*.png")
    )

    if sample_files:
        selected_file = st.selectbox(
            "Select Image to Test:", 
            sample_files,
            format_func=lambda x: f"Evaluation Image: {x}"
        )
        sample_img = Image.open(selected_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Input Image: `{selected_file}`**")
            st.image(sample_img, use_container_width=True)

        with col2:
            st.markdown("**Inference Output**")
            if st.button("Run Detection on Selected Image", key="btn_sample", use_container_width=True):
                if yolo_model is not None:
                    with st.spinner("Executing YOLO detection..."):
                        results = yolo_model.predict(sample_img, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        st.image(annotated_img, use_container_width=True)

                        boxes = results[0].boxes
                        st.success(f"Detection Complete! Found: **{len(boxes)} face(s)**")
                        for idx, b in enumerate(boxes):
                            cls_name = results[0].names[int(b.cls[0])]
                            conf_val = float(b.conf[0])
                            st.write(f"- **Detection #{idx+1}:** `{cls_name}` (Confidence: **{conf_val*100:.2f}%**)")
                else:
                    st.error(f"Cannot run detection: {model_status}")
    else:
        st.warning("No sample files found matching 'sample*.jpg'. Place your sample images into the repository root.")

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
            if yolo_model is not None:
                with st.spinner("Detecting face..."):
                    results = yolo_model.predict(raw_img, conf=conf_threshold)
                    annotated_img = results[0].plot()
                    st.image(annotated_img, use_container_width=True)

                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for idx, b in enumerate(boxes):
                            cls_name = results[0].names[int(b.cls[0])]
                            conf_val = float(b.conf[0])
                            st.success(f"Detected: **{cls_name}** | Confidence: **{conf_val*100:.2f}%**")
                    else:
                        st.warning("No face detected above the chosen confidence threshold.")
            else:
                st.error(f"Model unavailable: {model_status}")

# ----------------- TAB 3: TRAINING SUMMARY & SPECS -----------------
with tabs[2]:
    st.subheader("Model Specifications & Architecture")
    st.markdown("""
    - **Target Identity:** Patience Korbla (`patience_face_detection`)
    - **Model Framework:** Ultralytics YOLO Object Detection Architecture
    - **Model Weights:** `best.pt` (Experiment 2 checkpoint)
    - **Task Formulation:** Single-class bounding box regression and facial identification
    - **Key Evaluation Metrics:**
      - **Precision (P):** Accuracy of positive facial bounding box predictions
      - **Recall (R):** Sensitivity across varying lighting conditions, orientations, and angles
      - **mAP@0.5:** Mean Average Precision at IoU threshold 0.50
    """)