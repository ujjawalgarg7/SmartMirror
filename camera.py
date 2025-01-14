import cv2

def capture_image():

    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    print("Press 'c' to capture an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        
        cv2.imshow("Camera", frame)

        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as 'captured_image.jpg'")
        elif key == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()


capture_image()
