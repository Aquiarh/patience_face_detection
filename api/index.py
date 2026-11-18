from pathlib import Path
from io import BytesIO
import time

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = ROOT_DIR / "best.onnx"
PT_MODEL_PATH = ROOT_DIR / "best.pt"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
INPUT_SIZE = 640
CLASS_NAMES = {0: "Dela"}

app = FastAPI(
    title="Patience Face Detection API (YOLO26n)",
    description="YOLO26n single-class face detection and localization API for target class Dela.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ort_session = None
yolo_model = None
ACTIVE_BACKEND = "none"

if ONNX_AVAILABLE and ONNX_MODEL_PATH.exists():
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    ACTIVE_BACKEND = "onnxruntime (best.onnx)"
elif ULTRALYTICS_AVAILABLE and PT_MODEL_PATH.exists():
    yolo_model = YOLO(str(PT_MODEL_PATH))
    ACTIVE_BACKEND = "ultralytics (best.pt)"


def run_onnx_inference(pil_image: Image.Image, conf_threshold: float):
    orig_w, orig_h = pil_image.size
    img_resized = pil_image.resize((INPUT_SIZE, INPUT_SIZE))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_np})
    preds = outputs[0]

    detections = []

    # Case A: YOLO26n end-to-end NMS-free output [1, N, 6] -> (x1, y1, x2, y2, conf, cls_id)
    if preds.ndim == 3 and preds.shape[-1] == 6:
        for row in preds[0]:
            x1, y1, x2, y2, score, cls_id = row
            if float(score) >= conf_threshold:
                cid = int(cls_id)
                detections.append({
                    "label": CLASS_NAMES.get(cid, "Dela"),
                    "class_id": cid,
                    "confidence": round(float(score), 4),
                    "bbox": [
                        round(float(x1 / INPUT_SIZE * orig_w), 2),
                        round(float(y1 / INPUT_SIZE * orig_h), 2),
                        round(float(x2 / INPUT_SIZE * orig_w), 2),
                        round(float(y2 / INPUT_SIZE * orig_h), 2),
                    ]
                })

    # Case B: Standard YOLO output [1, 5, 8400] -> (cx, cy, w, h, conf)
    elif preds.ndim == 3 and preds.shape[1] >= 5:
        rows = np.transpose(preds[0], (1, 0))
        boxes, scores = [], []
        for row in rows:
            cx, cy, w, h = row[:4]
            score = float(np.max(row[4:]))
            if score >= conf_threshold:
                x1 = (cx - w / 2.0) / INPUT_SIZE * orig_w
                y1 = (cy - h / 2.0) / INPUT_SIZE * orig_h
                x2 = (cx + w / 2.0) / INPUT_SIZE * orig_w
                y2 = (cy + h / 2.0) / INPUT_SIZE * orig_h
                boxes.append([x1, y1, x2, y2])
                scores.append(score)

        if boxes:
            order = np.argsort(scores)[::-1]
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                rest = order[1:]
                xx1 = np.maximum(boxes[i][0], [boxes[j][0] for j in rest])
                yy1 = np.maximum(boxes[i][1], [boxes[j][1] for j in rest])
                xx2 = np.minimum(boxes[i][2], [boxes[j][2] for j in rest])
                yy2 = np.minimum(boxes[i][3], [boxes[j][3] for j in rest])
                inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_r = np.array([(boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) for j in rest])
                iou = inter / (area_i + area_r - inter + 1e-6)
                order = rest[iou < 0.45]

            for idx in keep:
                detections.append({
                    "label": "Dela",
                    "class_id": 0,
                    "confidence": round(float(scores[idx]), 4),
                    "bbox": [round(float(v), 2) for v in boxes[idx]]
                })

    return detections


@app.get("/api")
@app.get("/api/")
def home():
    return {
        "message": "Patience Face Detection API (YOLO26n) is running.",
        "student": "Korbla Patience Aku Dela - PUIT/24120001",
        "backend": ACTIVE_BACKEND,
        "classes": CLASS_NAMES
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": ACTIVE_BACKEND != "none",
        "backend": ACTIVE_BACKEND,
        "classes": CLASS_NAMES
    }


@app.post("/api/detect")
async def detect_face(
    image: UploadFile = File(...),
    confidence: float = Form(0.40)
):
    if not 0.10 <= confidence <= 1.00:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.10 and 1.00.")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPG, PNG and WEBP images are supported.")

    image_bytes = await image.read(MAX_FILE_SIZE + 1)
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    width, height = original_image.size

    try:
        start_time = time.perf_counter()

        if ort_session is not None:
            detections = run_onnx_inference(original_image, confidence)
            mode = "live_onnx"
        elif yolo_model is not None:
            results = yolo_model.predict(source=original_image, conf=confidence, verbose=False)
            result = results[0]
            detections = []
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                detections.append({
                    "label": "Dela",
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": [round(float(value), 2) for value in coordinates]
                })
            mode = "live_ultralytics"
        else:
            raise HTTPException(
                status_code=500,
                detail="Neither best.onnx nor best.pt found. Run export_onnx.py first."
            )

        inference_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": True,
            "mode": mode,
            "filename": image.filename,
            "image_width": width,
            "image_height": height,
            "confidence_threshold": confidence,
            "detection_count": len(detections),
            "inference_ms": round(inference_ms, 2),
            "detections": detections
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"YOLO26n inference failed: {str(error)}")
   from fastapi.staticfiles import StaticFiles

# Serve public/ static files locally when running uvicorn (Vercel handles this via vercel.json)
PUBLIC_DIR = ROOT_DIR / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")