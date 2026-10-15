# Workflow: AI-Based Weapon Detection System

## 1. Project Overview

This project is an AI-powered weapon detection application built using a React frontend, a FastAPI backend, and a YOLO object detection model.

The backend accepts uploaded images and videos, runs model inference, and returns detection results or a processed video.

## 2. System Architecture

The current architecture uses direct inference through the FastAPI backend.

**Technology stack**

- **Frontend:** React and Vite
- **Backend:** FastAPI and Uvicorn
- **Object Detection:** Ultralytics YOLO
- **Image and Video Processing:** OpenCV and NumPy
- **Model Weights:** `backend/models/best.pt`
- **Hardware Acceleration:** CUDA when available, otherwise CPU

### Processing Flow

1. The user selects an image or video in the frontend.
2. The frontend sends the uploaded file to the appropriate backend endpoint.
3. FastAPI receives the file.
4. The YOLO model processes the image or video frames.
5. For images, the backend returns detection labels, confidence scores, and bounding-box coordinates.
6. For videos, the backend returns a processed video with detection annotations.

## 3. Backend Development

### Model Initialization

The backend creates a persistent detector worker and loads the trained YOLO model from `models/best.pt`.

The inference device is selected automatically:

- CUDA GPU when available.
- CPU when CUDA is unavailable.

The current detection confidence threshold is `0.5`.

### Image Detection

**Endpoint:** `POST /detect`

The endpoint accepts an uploaded image and returns JSON containing a `detections` array.

Each detection contains:

- `bbox`: Bounding-box coordinates in `[x1, y1, x2, y2]` format.
- `confidence`: Model confidence score.
- `label`: Detected class name.

### Video Detection

**Endpoint:** `POST /video`

The endpoint accepts an uploaded video, saves it temporarily, processes its frames with the YOLO model, and returns the annotated video.

The current implementation uses OpenCV's video capture and writer functionality.

## 4. Frontend Integration

The frontend is responsible for providing the user interface and communicating with the backend.

The expected integration includes:

1. Selecting an image or video.
2. Sending the file to the corresponding FastAPI endpoint.
3. Displaying image detection results.
4. Presenting the processed video when video processing succeeds.
5. Showing appropriate loading indicators and error messages.

Frontend behavior should be updated here as additional features are implemented and verified.

## 5. Local Development Setup

### Prerequisites

- Python installed.
- Node.js and npm installed.
- Git installed.
- Project model weights available at the expected path.

### Start the Backend

From the backend directory, install the dependencies listed in `requirements.txt`, then run:

```bash
uvicorn main:app --reload
```

FastAPI's interactive API documentation is normally available at:

`http://127.0.0.1:8000/docs`

### Start the Frontend

From the frontend directory, install the dependencies and start the Vite development server:

```bash
npm install
npm run dev
```

Use the URL printed by Vite to open the frontend.

The frontend's API URL and the backend's CORS configuration must be set appropriately for the local development environment.

## 6. Testing and Validation

Before releasing changes, verify the following:

- The backend starts and loads the model successfully.
- Image uploads return the expected detection response.
- Bounding-box coordinates, labels, and confidence scores are valid.
- Video uploads produce a playable annotated video.
- Missing or unsupported files produce appropriate errors.
- The frontend handles failed requests and long-running video processing.
- The application works on CPU, with GPU acceleration tested separately when available.

## 7. Security and Reliability

Before deploying the application publicly:

- Restrict CORS to trusted frontend origins.
- Validate uploaded file types and sizes.
- Generate safe temporary filenames and clean up temporary files.
- Handle invalid images, videos, and model-loading failures.
- Configure appropriate request limits and error handling.
- Avoid exposing credentials, private datasets, or sensitive information.

## 8. Future Enhancements

The following are potential improvements and are not assumed to be implemented in the current codebase:

- WebSocket-based live video detection.
- Distributed inference using Ray Serve.
- Distributed model training using Ray Train.
- Docker containerization.
- Kubernetes deployment and scaling.
- Automated tests and continuous integration.
- Performance evaluation using documented precision, recall, and mAP metrics.

## 9. Development Workflow

1. Make changes in the backend or frontend.
2. Run the application locally.
3. Test the affected functionality.
4. Review changes and check for accidental secrets.
5. Commit changes using descriptive Git messages.
6. Push changes to GitHub.
7. Update this document whenever the implementation or architecture changes.

---

**Note:** This document describes the architecture evidenced by the current backend implementation. Future infrastructure and features should be documented as implemented only after they have been developed and tested.