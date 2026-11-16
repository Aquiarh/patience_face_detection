
from pathlib import Path
from io import BytesIO
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Patience Face Detection API",
    description="YOLO-based single-class face detection and localization API.",
    version="1.0.0"
)

# Development configuration.
# We will restrict this when connecting the production frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 2. LOAD YOLO MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print("Classes:", model.names)

# --------------------------------------------------
# 3. HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Patience Face Detection API is running.",
        "model": "best.pt",
        "task": model.task,
        "classes": model.names
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_task": model.task,
        "classes": model.names
    }

# --------------------------------------------------
# 4. FACE DETECTION ENDPOINT
# --------------------------------------------------

@app.post("/detect")
async def detect_face(
    image: UploadFile = File(...),
    confidence: float = Form(0.40)
):

    # Validate confidence threshold
    if not 0.10 <= confidence <= 1.00:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0.10 and 1.00."
        )

    # Validate file format
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    # Read image and enforce upload size
    image_bytes = await image.read(MAX_FILE_SIZE + 1)

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        original_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    width, height = original_image.size

    # Execute actual YOLO inference
    try:
        start_time = time.perf_counter()

        results = model.predict(
            source=original_image,
            conf=confidence,
            verbose=False
        )

        inference_ms = (
            time.perf_counter() - start_time
        ) * 1000

        result = results[0]
        detections = []

        for box in result.boxes:

            coordinates = box.xyxy[0].tolist()

            detections.append({
                "label": result.names[int(box.cls[0])],
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": [
                    round(float(value), 2)
                    for value in coordinates
                ]
            })

        return {
            "success": True,
            "mode": "live",
            "filename": image.filename,
            "image_width": width,
            "image_height": height,
            "confidence_threshold": confidence,
            "detection_count": len(detections),
            "inference_ms": round(inference_ms, 2),
            "detections": detections
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"YOLO inference failed: {str(error)}"
        )
