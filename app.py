from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

# ------------------------
# App initialization
# ------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Load YOLOv11 model
# ------------------------
model = YOLO("best.pt")

# ------------------------
# Prediction API
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLO inference
    results = model(file_path)[0]

    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            })

    return jsonify({
        "detections": detections
    })

# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)


# To run the app, use the command: python app.py