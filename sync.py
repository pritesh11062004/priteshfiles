import cv2
import webbrowser
import time
import threading

def people_detection():
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

        # Draw rectangles around the detected people
        for (x, y, w, h) in detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('People Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def human_detection():
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

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def smile_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

            for (sx, sy, sw, sh) in smiles:
                # Draw a rectangle around the detected face and smile
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)

                # Introduce a delay of 5 seconds before opening the web page
                time.sleep(5)

                # Open a web page once the delay is over
                webbrowser.open(r"C:\Users\LENOVO\OneDrive\Desktop\wave2.html", new=2)

        # Display the resulting frame
        cv2.imshow('Smile Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Create threads for each detection task
    people_thread = threading.Thread(target=people_detection)
    human_thread = threading.Thread(target=human_detection)
    smile_thread = threading.Thread(target=smile_detection)

    # Start all threads
    people_thread.start()
    human_thread.start()
    smile_thread.start()

    # Wait for all threads to finish
    people_thread.join()
    human_thread.join()
    smile_thread.join()

if __name__ == "__main__":
    main()
