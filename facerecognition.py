import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import face_recognition
import cv2
import os
import datetime
import csv

# Function to load images and encode faces
def load_images_and_encode_faces(images_path):
    known_faces = []
    known_names = []

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load image
            image_path = os.path.join(images_path, filename)
            image = face_recognition.load_image_file(image_path)

            # Encode face
            face_encoding = face_recognition.face_encodings(image)[0]

            # Extract name from the filename (assuming filename is in the format "Name.jpg")
            name = os.path.splitext(filename)[0]

            # Store the encoded face and name
            known_faces.append(face_encoding)
            known_names.append(name)

    return known_faces, known_names

# Function to check if a student is already present in the CSV file
def is_student_present(name, csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == name and row[1] == "Present":
                return True
    return False

# Function to mark attendance
def mark_attendance(name, date):
    # Define the CSV file name based on the date
    csv_file = f"attendance_{date}.csv"

    # Check if the CSV file exists, create it if not
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Status", "Timestamp"])

    # Check if the student is already marked present
    if not is_student_present(name, csv_file):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Append the data to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, "Present", timestamp])

        print(f"{name} marked present at {timestamp} on {date}")
    else:
        print(f"{name} is already present on {date}")

# Function to recognize faces in real-time video stream
def recognize_faces(video_source=0, min_confidence=0.5):
    # Load known faces and names
    known_faces, known_names = load_images_and_encode_faces("images")

    # Open video capture
    cap = cv2.VideoCapture(video_source)

    # Get today's date
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")

    while True:
        ret, frame = cap.read()

        # Find faces in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=min_confidence)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # Mark attendance
                mark_attendance(name, date_today)

            # Draw rectangle and display the name on the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Function to open camera and start attendance
def start_attendance():
    messagebox.showinfo("Welcome", "Starting attendance system. Close the window to stop.")
    recognize_faces()

# Create the GUI
root = tk.Tk()
root.title("Face Recognition Attendance System")

# Welcome message
welcome_label = tk.Label(root, text="Welcome to Face Recognition Attendance System", font=("Helvetica", 16))
welcome_label.pack(pady=20)

# Load GIFs
camera_gif = Image.open("camera.gif")
camera_gif = camera_gif.resize((150, 150))
camera_gif = ImageTk.PhotoImage(camera_gif)



# Button to start attendance
start_button = tk.Button(root, text="Start Attendance", font=("Helvetica", 12), command=start_attendance, image=camera_gif, compound=tk.TOP)
start_button.pack(pady=10)

# Button to quit
quit_button = tk.Button(root, text="Quit", font=("Helvetica", 12), command=root.quit,compound=tk.TOP)
quit_button.pack(pady=10)

root.mainloop()
