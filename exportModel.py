from ultralytics import YOLO

# Load your model
model = YOLO("runs/detect/train2/weights/best.pt")

# Export as ONNX model (this creates a file like best.onnx)
model.export(format="onnx")

from ultralytics import YOLO

# Load your model
model = YOLO("runs/detect/train2/weights/best.pt")

# Export as TorchScript model (this creates a file like best.torchscript.pt)
model.export(format="torchscript")
