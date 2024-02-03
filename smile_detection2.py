import cv2
import webbrowser
import time

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
            time.sleep(10)

            # Open a web page once the delay is over
            webbrowser.open(r"file:///B:/techno%20pro/html%20part/AMINATION%20TEST.html", new=2)
    # Display the resulting frame
    cv2.imshow('Smile Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
