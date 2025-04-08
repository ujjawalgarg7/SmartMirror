from ultralytics import YOLO

def main():
    # Load the trained model (change the path to your weight file if necessary)
    model = YOLO("runs/detect/train2/weights/best.pt")
    
    # Evaluate the model using your dataset configuration
    results = model.val(data="shapes.yaml")
    print(results)

if __name__ == "__main__":
    # For extra safety (especially if freezing an executable)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
