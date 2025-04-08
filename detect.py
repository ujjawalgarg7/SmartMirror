from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Run inference on test images
results = model.predict(source="shapes_dataset/images/test", save=True, conf=0.25)

# Print a summary of results
for idx, result in enumerate(results):
    print(f"Image {idx}: {result.boxes}")
