import time
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO

from .models import Detection, DetectionResponse, BoundingBox

# Initialize FastAPI app
app = FastAPI(
    title="YOLO Object Detection API",
    description="API for object detection using YOLOv8",
    version="1.0.0"
)

# Load YOLO model (lazy loading)
model: Optional[YOLO] = None
MODEL_NAME = "yolov8n.pt"

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_model() -> YOLO:
    """Lazy load YOLO model"""
    global model
    if model is None:
        print(f"Loading YOLO model: {MODEL_NAME}")
        model = YOLO(MODEL_NAME)
    return model


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "YOLO Object Detection API",
        "model": MODEL_NAME
    }


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    yolo = get_model()
    return {
        "model_name": MODEL_NAME,
        "task": yolo.task,
        "classes_count": len(yolo.names),
        "classes": yolo.names
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
        file: UploadFile = File(..., description="Image file to process"),
        confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
        save_annotated: bool = Query(True, description="Save annotated image")
):
    """
    Detect objects in an uploaded image

    - **file**: Image file (jpg, jpeg, png, webp)
    - **confidence**: Minimum confidence threshold (0.0-1.0)
    - **save_annotated**: Whether to save the annotated image
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        height, width = image.shape[:2]

        # Run inference
        yolo = get_model()
        start_time = time.time()
        results = yolo(image, conf=confidence, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Process detections
        detections = []
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            detection = Detection(
                class_id=class_id,
                class_name=result.names[class_id],
                confidence=round(conf, 4),
                bbox=BoundingBox(
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    x2=round(x2, 2),
                    y2=round(y2, 2)
                )
            )
            detections.append(detection)

        # Save outputs
        image_stem = Path(file.filename).stem

        # Save JSON output
        json_output = {
            "image_name": file.filename,
            "detections": [d.model_dump() for d in detections]
        }
        json_path = OUTPUT_DIR / f"{image_stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        # Save annotated image
        if save_annotated:
            annotated = result.plot()
            annotated_path = OUTPUT_DIR / f"{image_stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated)

        return DetectionResponse(
            success=True,
            image_name=file.filename,
            image_size={"width": width, "height": height},
            detections_count=len(detections),
            detections=detections,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Download an output file (JSON or annotated image)"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)