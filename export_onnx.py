from ultralytics import YOLO

# Export your trained YOLO26n weights (best.pt) to lightweight ONNX format for Vercel
model = YOLO("best.pt")
exported_path = model.export(format="onnx", imgsz=640, opset=17, simplify=True)
print(f"Success! Exported model to: {exported_path}")