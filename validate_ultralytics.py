from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

model = YOLO("best.pt")
metrics = model.val(
    data="https://platform.ultralytics.com/patience-korbla/datasets/patience-detection",
    imgsz=640
)

print("=== REAL ULTRALYTICS VALIDATION METRICS ===")
print(f"mAP@50:    {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
