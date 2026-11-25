# Patience Face Recognition

**Real-Time Single-Class Facial Identity Localization using Ultralytics YOLO26n & ONNX Runtime**

- **Author:** Korbla Patience Aku Dela (`PUIT/24120001`)
- **Institution:** Department of Information Technology, Pentecost University
- **Live Web Application:** [https://patience-face-detection-suiw.vercel.app](https://patience-face-detection-suiw.vercel.app)
- **GitHub Repository:** [https://github.com/Aquiarh/patience_face_detection](https://github.com/Aquiarh/patience_face_detection)
- **Ultralytics Dataset (1,000 Images):** [https://platform.ultralytics.com/patience-korbla/datasets/patience-detection](https://platform.ultralytics.com/patience-korbla/datasets/patience-detection)
- **Executed Analysis Notebook:** [`patience_face_recognition_analysis.ipynb`](./patience_face_recognition_analysis.ipynb)

---

## Overview

**Patience Face Recognition** is a full-stack computer vision web application engineered to detect and localize target subject **Dela** (`Class 0`) across indoor, outdoor, group, and mobile camera frames. The underlying **Ultralytics YOLO26n** detector (`121` fused layers, `2,389,735` parameters, `5.4 GFLOPs`) is exported to an FP32 ONNX graph (`best.onnx`, `9.4 MB`) with `0.0 ms` Non-Maximum Suppression (NMS) overhead and served via a FastAPI Python serverless backend on Vercel.

---

## Dataset Summary (`patience-detection`)

| Partition / Subset | Image Count | Proportion | Ground-Truth Instances (`Dela`) |
| :--- | :---: | :---: | :---: |
| **Training Set (`train`)** | 679 | 67.90% | Annotated |
| **Validation Set (`val`)** | 321 | 32.10% | 282 |
| — Positive Target Frames | 282 | 28.20% | 282 |
| — Negative / Background Frames | 39 | 3.90% | 0 |
| **Total Curated Dataset** | **1,000** | **100.00%** | — |

---

## Validation Benchmark Metrics (`runs/detect/val`)

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **mAP@50** | **94.87%** (`0.9487`) | Primary detection accuracy across 321 validation images |
| **mAP@50-95** | **66.02%** (`0.6602`) | Strict multi-threshold IoU localization |
| **Box Precision (PPV)** | **92.92%** (`276 / 297`) | False Discovery Rate (FDR): `7.08%` |
| **Box Recall (TPR)** | **97.87%** (`276 / 282`) | False Negative Rate (FNR): `2.13%` |
| **Peak F1-Score** | **0.9533** (`95.33%`) | Achieved at confidence threshold $\tau = 0.40$ |
| **NMS Post-Processing** | **0.0 ms** | Native one-to-one end-to-end head |

---

## Repository Structure

```text
patience_face_detection/
├── api/                                      # FastAPI serverless endpoints (/api/detect, /api/health)
├── public/
│   ├── index.html                            # Responsive 6-tab inference & analytics web interface
│   └── sample1.jpg ... sample10.PNG          # Embedded validation sample images
├── app.py                                    # Local FastAPI / standalone web server entry point
├── inference_api.py                          # ONNX Runtime preprocessing & inference module
├── export_onnx.py                            # PyTorch (best.pt) to FP32 ONNX (best.onnx) export script
├── validate_ultralytics.py                   # Validation benchmark evaluation script
├── benchmark.py                              # Inference latency & throughput benchmarking script
├── test_all_samples.py                       # Automated test runner across validation samples
├── best.onnx                                 # Exported FP32 ONNX model weights (9.4 MB)
├── best.pt                                   # Trained Ultralytics YOLO26n PyTorch weights
├── patience_face_recognition_analysis.ipynb  # Executed Jupyter Notebook (EDA, regression & metrics)
├── requirements.txt                          # Serverless Python dependencies
├── vercel.json                               # Vercel routing & serverless build configuration
└── README.md                                 # Project documentation