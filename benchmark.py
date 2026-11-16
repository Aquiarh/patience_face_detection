from ultralytics import YOLO
from PIL import Image
import time

print("Loading YOLO model...")
model = YOLO("best.pt")
image = Image.open("sample1.jpg").convert("RGB")

print("Warming up model...")
model.predict(image, conf=0.40, verbose=False)

times = []

print("\nRunning 3 benchmark tests...")

for i in range(3):
    start = time.perf_counter()

    results = model.predict(
        image,
        conf=0.40,
        verbose=False
    )

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    print(f"Run {i + 1}: {elapsed:.2f} ms")

print("\n========== BENCHMARK RESULTS ==========")
print(f"Average inference time: {sum(times)/len(times):.2f} ms")
print(f"Device: {model.device}")
print(f"Detections: {len(results[0].boxes)}")
