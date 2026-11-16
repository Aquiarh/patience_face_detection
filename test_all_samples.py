from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import time
import re

model = YOLO("best.pt")

image_files = [
    p for p in Path(".").iterdir()
    if p.is_file()
    and p.name.lower().startswith("sample")
    and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

image_files.sort(
    key=lambda p: [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", p.name)
    ]
)

output_dir = Path("validation_outputs")
output_dir.mkdir(exist_ok=True)

print(f"\nFound {len(image_files)} sample images.")

if not image_files:
    raise SystemExit("No verification images found.")

# Warm-up
model.predict(
    Image.open(image_files[0]).convert("RGB"),
    conf=0.40,
    verbose=False
)

print("\n========== MODEL EVALUATION ==========")

for image_path in image_files:
    image = Image.open(image_path).convert("RGB")

    start = time.perf_counter()

    result = model.predict(
        image,
        conf=0.40,
        verbose=False
    )[0]

    elapsed = (time.perf_counter() - start) * 1000

    result.save(
        filename=str(output_dir / image_path.name)
    )

    boxes = result.boxes

    print(f"\nImage: {image_path.name}")
    print(f"Detections: {len(boxes)}")
    print(f"Inference time: {elapsed:.2f} ms")

    for box in boxes:
        confidence = float(box.conf[0]) * 100
        class_name = result.names[int(box.cls[0])]

        print(
            f"  Class: {class_name} | "
            f"Confidence: {confidence:.2f}%"
        )

print("\n======================================")
print(f"Annotated images saved in: {output_dir}")
