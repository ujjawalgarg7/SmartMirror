import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
else:
    print("Press 'c' to capture an image or 'q' to quit.")

    captured_frame = None  # Variable to store the captured frame

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow('Camera Preview', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture the image when 'c' is pressed
            captured_frame = frame
            print("Image captured and stored in variable.")
            print(type(captured_frame))
            
        elif key == ord('q'):  # Quit when 'q' is pressed
            print("Exiting without capturing.")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    # The captured frame is stored in 'captured_frame'
    if captured_frame is not None:
        print("Captured frame is now available in the variable.")

