from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from datetime import date, datetime
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Register the custom filter for date formatting
@app.template_filter('date')
def format_date(value, format="%Y-%m-%d"):
    if isinstance(value, datetime):
        return value.strftime(format)
    return value

# Constants
date_today = date.today().strftime("%m_%d_%y")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ensure necessary directories
os.makedirs("Attendance", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)
os.makedirs("static/models", exist_ok=True)

if f"Attendance-{date_today}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{date_today}.csv", "w") as f:
        f.write("Name,USN,Subject,Login Time,Logout Time\n")

# Helper Functions
def total_registered():
    return len(os.listdir("static/faces"))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces

def train_model():
    faces = []
    labels = []
    for user_dir in os.listdir("static/faces"):
        user_path = os.path.join("static/faces", user_dir)
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            img = cv2.imread(image_path)
            if img is not None:
                resized = cv2.resize(img, (50, 50)).ravel()
                faces.append(resized)
                labels.append(user_dir)
    if faces:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), labels)
        joblib.dump(knn, "static/models/face_recognition_model.pkl")
        return True
    return False

def identify_face(facearray):
    model = joblib.load("static/models/face_recognition_model.pkl")
    return model.predict(facearray)

def record_attendance(name, usn, subject, is_logout=False):
    current_time = datetime.now().strftime("%H:%M:%S")
    csv_path = f"Attendance/Attendance-{date_today}.csv"
    df = pd.read_csv(csv_path)
    existing_entry = df[(df["USN"] == usn) & (df["Subject"] == subject)]

    if not is_logout:
        if existing_entry.empty:
            new_entry = {"Name": name, "USN": usn, "Subject": subject, "Login Time": current_time, "Logout Time": ""}
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df.loc[existing_entry.index, "Login Time"] = current_time
    else:
        if not existing_entry.empty and pd.isna(existing_entry["Logout Time"].iloc[0]):
            df.loc[existing_entry.index, "Logout Time"] = current_time
        else:
            return False

    df.to_csv(csv_path, index=False)
    return True

def get_attendance_data():
    csv_path = f"Attendance/Attendance-{date_today}.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).values.tolist()
    return []

# Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form["name"]
        usn = request.form["usn"]
        folder_path = os.path.join("static/faces", f"{name}_{usn}")
        os.makedirs(folder_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        while count < 10:
            ret, frame = cap.read()
            if not ret:
                continue
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                count += 1
                face = frame[y:y+h, x:x+w]
                cv2.imwrite(f"{folder_path}/{count}.jpg", face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Capturing Faces", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        flash("Student added successfully!", "success")
        return redirect(url_for("add_student"))
    return render_template("add.html")

@app.route("/train", methods=["GET", "POST"])
def train_images():
    if request.method == "POST":
        if train_model():
            flash("Images trained successfully!", "success")
        else:
            flash("No faces found to train. Please add students first!", "error")
        return redirect(url_for("train_images"))
    return render_template("train.html")

@app.route("/take", methods=["GET", "POST"])
def take_attendance():
    if request.method == "POST":
        action = request.form["action"]
        subject = request.form["subject"]

        cap = cv2.VideoCapture(0)
        success = False  # Flag to check if attendance was recorded

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
                try:
                    name_usn = identify_face(face)[0]
                    name, usn = name_usn.split("_")
                    if record_attendance(name, usn, subject, is_logout=(action == "logout")):
                        success = True
                except Exception as e:
                    print(f"Error: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({usn})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Taking Attendance", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'Esc'
                break

        cap.release()
        cv2.destroyAllWindows()

        if success:
            flash("Attendance recorded successfully!", "success")
        else:
            flash("No attendance recorded. Please try again!", "error")
        
        return redirect(url_for("take_attendance"))

    return render_template("take.html")

@app.route("/view")
def view_attendance():
    attendance_data = get_attendance_data()
    return render_template("view.html", attendance_data=attendance_data)

if __name__ == "__main__":
    app.run(debug=True)