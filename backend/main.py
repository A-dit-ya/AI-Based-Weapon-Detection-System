from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import os
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeaponDetectorWorker:
    def __init__(self):
        self.model = YOLO("models/best.pt")
        self.model.to(DEVICE)

    def process_image(self, image_bytes):
        # Decode the bytes into an OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference on GPU
        results = self.model.predict(
            img,
            conf=0.5,
            verbose=False,
            device=DEVICE
        )

        # Format results for the frontend
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "label": self.model.names[int(box.cls)]
                })

        return detections

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 for web browsers
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            results = self.model.predict(
                frame,
                conf=0.5,
                verbose=False,
                device=DEVICE
            )

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return True


# FASTAPI LIFECYCLE
TEMP_DIR = os.path.abspath("temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

# Create one persistent worker instance
detector_worker = WeaponDetectorWorker()


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Direct call instead of Ray
    detections = detector_worker.process_image(image_bytes)

    return {"detections": detections}


@app.post("/video")
async def detect_video(file: UploadFile = File(...)):
    input_path = os.path.join(TEMP_DIR, f"in_{file.filename}")
    output_path = os.path.join(TEMP_DIR, f"out_{file.filename}")

    # 1. Save video to disk
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Process video directly
    print(f"Processing {file.filename}...")

    detector_worker.process_video(input_path, output_path)

    # 3. Cleanup input and return output
    os.remove(input_path)

    return FileResponse(output_path, media_type="video/mp4")