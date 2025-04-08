import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Open default camera (change parameter if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # Draw detections on frame (YOLOv8 returns annotated image)
    annotated_frame = results[0].plot()

    # Show the image in a window
    cv2.imshow("Real-Time Detection", annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
