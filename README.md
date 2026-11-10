# Patience Face Detection & Recognition System (`patience_face_detection`)

An intelligent biometric computer vision web portal built using a custom-trained Ultralytics YOLO model and Streamlit for real-time face detection, bounding box localization, and personal identification.

## Project Overview
This project implements an end-to-end computer vision pipeline trained specifically on personal facial data (`patience_face_detection`). The web application provides automated facial detection from multi-source visual inputs (pre-loaded verification samples, webcam capture, and manual image uploads) and outputs localized bounding boxes, target classifications, and confidence scores.

## Key Features
- **Custom YOLO Architecture:** Fine-tuned on custom facial annotations using Ultralytics YOLO (`exp-2.pt` / `best.pt`).
- **Flexible Inference Engine:** Features local PyTorch model execution with seamless cloud-native REST fallback via Google Cloud Run (`predict-6ae863de0efc9562c8847473-dproatj77a-nw.a.run.app`).
- **Lecturer Verification Gallery:** Dynamic pre-loaded sample dataset (`sample*.jpg`) allowing evaluators to verify localization accuracy and confidence scoring without uploading local files.
- **Real-Time Camera Integration:** Direct client-side webcam capture for live biometric verification.
- **Adjustable Parameters:** Interactive sidebar confidence threshold slider (0.10 to 1.00) for empirical precision/recall evaluation.

## Local Setup & Run
```bash
# Clone the repository
git clone [https://github.com/Aquiarh/patience_face_detection.git](https://github.com/Aquiarh/patience_face_detection.git)
cd patience_face_detection

# Install dependencies
py -m pip install -r requirements.txt

# Launch application
py -m streamlit run app.py