from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")  # or yolov8n.pt if fine-tuning
    model.train(data="shapes.yaml", epochs=10, imgsz=640)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, but safe
    main()
