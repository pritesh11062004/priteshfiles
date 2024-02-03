import cv2
import time
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
cap = cv2.VideoCapture(0)

desired_people_count = 1
output_folder = r"C:\Users\LENOVO\OneDrive\Desktop\captured photos\\"
global people_detected

def people_detection():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if len(faces) + len(upper_bodies) == desired_people_count:
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        # Capture the frame and save it as a photo with a unique filename
        file_name = f"captured_people_{timestamp}.jpeg"
        cv2.imwrite(output_folder + file_name, frame)
        print(f"{desired_people_count} people detected and captured! Photo saved as {file_name}")

        # Delay for 3 seconds before capturing the next photo
        time.sleep(3)
        global people_detected = true

    # Display the frame with rectangles
    cv2.imshow('Frame', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    return people_detected

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
