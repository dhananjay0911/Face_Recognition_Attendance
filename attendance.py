import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time

# Load the face recognition model and other initial setup
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to capture face data for training
def capture_faces(person_id, sample_size=50):
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"dataset/User.{person_id}.{count}.jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("Capturing Faces", frame)

            if count >= sample_size:
                break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train the recognizer with face data
def train_recognizer():
    face_samples = []
    ids = []
    image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset')]

    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        person_id = int(os.path.split(image_path)[-1].split(".")[1])
        face_samples.append(gray_img)
        ids.append(person_id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.write('trainer.yml')
    print("Training completed.")

# Function to mark attendance in the Excel file
def mark_attendance(person_id):
    df = pd.read_excel("attendance.xlsx")
    
    if person_id not in df['Person_ID'].values:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.concat([df, pd.DataFrame([{"Person_ID": person_id, "Time": current_time}])], ignore_index=True)
        df.to_excel("attendance.xlsx", index=False)
        print(f"Attendance marked for Person ID {person_id}")
    else:
        print(f"Person ID {person_id} has already marked attendance.")

# Function to recognize faces and mark attendance
def recognize_faces():
    recognizer.read('trainer.yml')
    cap = cv2.VideoCapture(0)

    start_time = time.time()  # Record the start time

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 50:
                    mark_attendance(id_)
                    cv2.putText(frame, f"ID: {id_}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Recognizing Faces", frame)

        # Exit after 5 seconds
        if time.time() - start_time > 5:
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to initialize the Excel file
def initialize_excel():
    if not os.path.exists("attendance.xlsx"):
        df = pd.DataFrame(columns=["Person_ID", "Time"])
        df.to_excel("attendance.xlsx", index=False)
        print("Initialized attendance Excel file.")

# Main function
if __name__ == "__main__":
    initialize_excel()
    
    while True:
        option = input("Select option: \n1. Capture Faces\n2. Train Recognizer\n3. Recognize Faces\n4. Exit\nEnter: ")

        if option == '1':
            person_id = input("Enter Person ID: ")
            capture_faces(person_id)
        elif option == '2':
            train_recognizer()
        elif option == '3':
            recognize_faces()
        elif option == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option!")
