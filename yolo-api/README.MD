# YOLO API + Evaluation

## 📋 Prerequisites

### Download the YOLOv8 Model

Before running the application, download the YOLOv8 nano model:

```bash
# From the yolo-api directory
cd yolo-api

# Download YOLOv8n model (will auto-download on first run, or manually):
# The model will automatically download when you first use it, or download manually:
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**Alternative**: The Ultralytics library will automatically download `yolov8n.pt` on first use if not present.

---

## Session Summary: Built a FastAPI-based REST API for YOLOv8 object detection with a `/detect` endpoint, JSON output, and an evaluation script to measure model quality using Precision, Recall, and F1-Score metrics.

## What I Did During the Session

1. **Set Up Project Environment**:
   - Created a virtual environment named `yolo-api` using Python's built-in `venv`
   - Configured `requirements.txt` with compatible package versions for Python 3.9
   - Resolved NumPy version compatibility issues (1.24.4 for Python 3.9 instead of 1.26.3)
   - Installed FastAPI, Uvicorn, Ultralytics, and OpenCV dependencies

2. **Designed API Data Models (`models.py`)**:
   - Created `BoundingBox` Pydantic model for coordinate representation (x1, y1, x2, y2)
   - Built `Detection` model containing class_id, class_name, confidence, and bbox
   - Designed `DetectionResponse` for structured API responses with inference timing
   - Added `EvaluationMetrics` model for precision/recall/F1 score results

3. **Implemented FastAPI Application (`main.py`)**:
   - Set up FastAPI app with metadata (title, description, version)
   - Created health check endpoint (`GET /`) for API status monitoring
   - Built model info endpoint (`GET /model/info`) to retrieve YOLO model details
   - Implemented lazy loading pattern for YOLO model to optimize startup time

4. **Built Object Detection Endpoint (`POST /detect`)**:
   - Accepted image uploads via `multipart/form-data` with file validation
   - Added configurable confidence threshold parameter (0.0-1.0, default 0.25)
   - Processed images using YOLOv8 inference with OpenCV integration
   - Returned structured JSON with detections, bounding boxes, and inference time
   - Saved annotated images and JSON outputs to `outputs/` directory

5. **Created Evaluation Script (`evaluate.py`)**:
   - Implemented IoU (Intersection over Union) calculation for bounding box comparison
   - Built ground truth annotation loader supporting JSON format
   - Created detection evaluation logic matching predictions to ground truth
   - Calculated True Positives, False Positives, and False Negatives
   - Computed Precision, Recall, and F1-Score metrics with configurable IoU threshold

6. **Developed API Testing Script (`test_api.py`)**:
   - Created Python script using `requests` library for API testing
   - Implemented image upload functionality to `/detect` endpoint
   - Parsed and displayed JSON detection responses
   - Provided alternative to PowerShell's limited `-Form` parameter support

7. **Configured Project Structure**:
   - Organized code into `app/` (API) and `evaluation/` (metrics) packages
   - Set up `test_images/` for test dataset and `evaluation/annotations/` for ground truth
   - Created `outputs/` directory for detection results and evaluation reports
   - Added `__init__.py` files for proper Python package structure

## What I Learned

1. **FastAPI for ML Model Serving**:
   - FastAPI provides automatic OpenAPI/Swagger documentation at `/docs`
   - Pydantic models enable request/response validation and serialization
   - `UploadFile` handles multipart file uploads with async support
   - Query parameters with `Query()` allow validation constraints (ge, le, default)
   - Lazy loading patterns prevent slow startup when loading heavy ML models

2. **REST API Design for Computer Vision**:
   - Use `multipart/form-data` content type for image uploads
   - Return structured JSON responses with metadata (image size, inference time)
   - Include confidence scores and bounding box coordinates in detection results
   - Validate file types server-side to prevent processing invalid uploads
   - Save both raw JSON and annotated images for different use cases

3. **Object Detection Evaluation Metrics**:
   - **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth boxes
   - **Precision**: TP / (TP + FP) — How many predictions are correct
   - **Recall**: TP / (TP + FN) — How many ground truth objects were found
   - **F1-Score**: Harmonic mean of Precision and Recall for balanced evaluation
   - IoU threshold (typically 0.5) determines if a detection counts as correct

4. **Ground Truth Annotation Format**:
   - JSON-based annotations link image filenames to bounding box coordinates
   - Bounding boxes stored as [x1, y1, x2, y2] with class labels
   - Matching predictions to ground truth requires both IoU and class comparison
   - Sort predictions by confidence to prioritize high-confidence matches

5. **Python Virtual Environments**:
   - `python -m venv <name>` creates isolated Python environments
   - Activation differs by OS: `Scripts\activate` (Windows) vs `source bin/activate` (Linux/Mac)
   - Virtual environments start empty — packages must be installed after creation
   - Package version conflicts (like NumPy) depend on Python version compatibility

6. **Running Uvicorn Server**:
   - `uvicorn app.main:app` follows the pattern `module:variable`
   - `--reload` enables auto-restart on code changes (development mode)
   - `--host 0.0.0.0` binds to all network interfaces
   - Use `python -m uvicorn` if the command isn't in PATH
   - Swagger UI at `/docs` provides interactive API testing

7. **API Testing Methods**:
   - Swagger UI (`/docs`) is the easiest way to test endpoints interactively
   - Python `requests` library handles file uploads with `files={"file": open(..., "rb")}`
   - PowerShell's `-Form` parameter requires version 6+ (use Python script as alternative)
   - `curl.exe` (not `curl`) avoids PowerShell alias issues on Windows

This session provided hands-on experience building production-style ML APIs, implementing evaluation pipelines for model quality assessment, and understanding the full workflow from model inference to metrics reporting.