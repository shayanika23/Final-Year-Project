# AI-Powered Crime Scene Analysis: Detecting and Logging Forensic Objects using YOLO

An AI-powered forensic evidence detection system that uses a custom-trained YOLO model to identify objects such as weapons, bloodstains, bullets, and casings in crime scene images. Built as a final-year B.Tech Computer Science & Engineering project.

🔗 **Live Demo:** [https://yolo-ai-crimescene-analysis.netlify.app](https://yolo-ai-crimescene-analysis.netlify.app)

---

## Overview

Crime scene investigation traditionally relies on manual inspection, which is slow, error-prone, and dependent on expert availability. This project automates:

- Detection of forensic objects (weapons, bloodstains, bullets, casings) in uploaded images
- Confidence-scored bounding box detection using a custom-trained YOLO model
- Evidence logging with timestamp and metadata in a database
- A clean web interface for uploading images and viewing analysis results

## Tech Stack

**Backend**
- Python, Flask, Flask-CORS
- Ultralytics YOLO (custom-trained model)
- OpenCV
- SQLite (detection logging)
- Gunicorn (production WSGI server)
- Docker

**Frontend**
- React + TypeScript
- Vite
- Tailwind CSS

**Hosting**
- Backend: Hugging Face Spaces (Docker SDK)
- Frontend: Netlify

## Project Structure

```
.
├── Backend/
│   ├── app.py              # Flask API (predict, detections endpoints)
│   ├── best.pt              # Trained YOLO model weights (tracked via Git LFS)
│   ├── requirements.txt
│   └── Dockerfile
├── Frontend/
│   ├── src/
│   │   └── App.tsx          # Main React app, handles image upload + API calls
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## API Endpoints

### `POST /predict`
Accepts an image file and returns detected forensic objects.

**Request:** `multipart/form-data` with a `file` field containing the image.

**Response:**
```json
{
  "detections": [
    {
      "class": "gun",
      "confidence": 0.88,
      "bbox": [146, 187, 569, 430]
    },
    {
      "class": "blood",
      "confidence": 0.40,
      "bbox": [332, 165, 471, 228]
    }
  ]
}
```

### `GET /detections`
Returns all previously logged detections stored in the database.

## Running Locally

### Backend
```bash
cd Backend
pip install -r requirements.txt
python app.py
```
The API will run on `http://127.0.0.1:5000`.

### Frontend
```bash
cd Frontend
npm install
npm run dev
```
The app will run on `http://localhost:5173`. Update the API URL in `src/App.tsx` if testing against a local backend instead of the hosted one.

## Deployment

**Backend (Hugging Face Spaces):**
1. Create a new Space with the **Docker** SDK
2. Install Git LFS locally: `git lfs install`
3. Clone the Space repo and copy in `app.py`, `best.pt`, `requirements.txt`, and a `Dockerfile`
4. Push via Git using a Hugging Face access token for authentication

**Frontend (Netlify):**
1. Update the backend URL in `src/App.tsx`
2. Build the production bundle: `npm run build`
3. Deploy the resulting `dist/` folder via [Netlify Drop](https://app.netlify.com/drop)
4. Claim the deployment with a free Netlify account to make it permanent







