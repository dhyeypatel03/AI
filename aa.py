import cv2

def main():
    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(2)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # If the frame was not captured successfully, break the loop
        if not ret:
            break

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Wait for a key press (the delay can be adjusted as needed)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
