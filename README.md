# AI-Based Weapon Detection System

An AI-powered weapon detection system that uses computer vision and deep learning to detect weapons in images and real-time video streams.

The project combines a YOLO-based object detection model with a FastAPI backend and a React.js frontend to provide an interactive weapon detection application.

---

## Features

- AI-based weapon detection using YOLO
- Real-time detection through webcam/video input
- Image-based weapon detection
- Bounding boxes around detected objects
- Confidence scores for detections
- React-based web interface
- FastAPI backend for model inference
- REST API communication between frontend and backend
- Custom-trained weapon detection model

---

## Tech Stack

### Machine Learning
- Python
- YOLO
- Ultralytics
- PyTorch
- OpenCV

### Backend
- FastAPI
- Python
- REST API

### Frontend
- React.js
- Vite
- JavaScript
- HTML
- CSS

---

## System Architecture

```text
                 ┌──────────────────┐
                 │   React Frontend │
                 └────────┬─────────┘
                          │
                   HTTP Requests
                          ▼
                 ┌──────────────────┐
                 │  FastAPI Backend │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ YOLO Detection   │
                 │      Model       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Detection Result │
                 │ Bounding Boxes   │
                 │ Confidence Score │
                 └──────────────────┘
```

---

## Final Project Demonstration

This video demonstrates the working of the AI-Based Weapon Detection System.

**Watch the final demonstration:** [View Demo Video] https://drive.google.com/file/d/1KnDW6Arc-TSo6yGBGOq8WM5J54SoXOLS/view?usp=drive_link

---

## Project Structure

```text
AI212-Weapon-Detection/
│
├── backend/
│   ├── models/
│   │   └── best.pt
│   ├── main.py
│   ├── requirements.txt
│   ├── train_weapons_system.py
│   └── train_wihtout_ray.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── photos/
├── README.md
├── workflow.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI212-Weapon-Detection
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will normally be available at:

http://127.0.0.1:8000

### 3. Frontend Setup

Open a **second terminal** from the project root directory and run:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm run dev
```

Vite will display the frontend URL in the terminal.

---

## Model

The project uses a YOLO object detection model for weapon detection.

The model weights are stored at:

`backend/models/best.pt`

The backend loads the model to perform object detection on user-provided input.

---

## Detection Workflow

```text
Input Image / Video / Webcam
              ↓
        Frame Processing
              ↓
       YOLO Model Inference
              ↓
        Weapon Detection
              ↓
    Bounding Box + Confidence
              ↓
       Detection Result
              ↓
      Frontend Visualization
```

---

## Project Objective

The objective of this project is to develop an AI-based computer vision system capable of detecting weapons automatically from visual input.

The system demonstrates how deep learning, object detection, computer vision, REST APIs, and web technologies can be combined to create an end-to-end AI application.

---

## Contributors

- Aditya Kumar Jatav