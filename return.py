import cv2
import webbrowser


# Load the pre-trained HOG people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Run the HOG detector on the captured frame
    detected, _ = hog.detectMultiScale(frame)

    # Flag to track if a human has been detected
    human_detected = False

    # Draw rectangles around the detected humans
    for (x, y, w, h) in detected:
        # Open a web page only once when a human is detected
        if not human_detected:
            webbrowser.open(r"C:\Users\LENOVO\OneDrive\Desktop\wave1.html", new=2)
            human_detected = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Human Detection', frame)

    # Delay for 5 seconds
    

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
