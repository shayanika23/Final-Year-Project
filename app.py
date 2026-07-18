import os
import sqlite3
from datetime import datetime
from uuid import uuid4
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ------------------------
# Setup Pathing and Model
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load model once globally
model = YOLO(MODEL_PATH)

# ------------------------
# App initialization
# ------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Database setup
# ------------------------
DB_NAME = os.path.join(BASE_DIR, "detections.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            class TEXT,
            confidence REAL,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------
# Prediction API
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid file name"}), 400

    stored_filename = f"{uuid4().hex}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    file.save(file_path)

    detections = []
    try:
        results = model(file_path)[0]
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

                cursor.execute("""
                    INSERT INTO detections
                    (image_name, class, confidence, x1, y1, x2, y2, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (filename, class_name, confidence, int(x1), int(y1), int(x2), int(y2), 
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"detections": detections})

# ------------------------
# Get stored detections
# ------------------------
@app.route("/detections", methods=["GET"])
def get_detections():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections")
    rows = cursor.fetchall()
    conn.close()

    data = [{
        "id": row[0], "image_name": row[1], "class": row[2],
        "confidence": row[3], "bbox": [row[4], row[5], row[6], row[7]],
        "timestamp": row[8]
    } for row in rows]

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)