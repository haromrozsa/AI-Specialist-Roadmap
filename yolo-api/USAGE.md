# Usage Guide - YOLO Object Detection API

## Prerequisites

### System Requirements
- Python 3.9+
- pip

### Download YOLOv8 Model (Required)

Before running the application, download the YOLOv8 nano model:

```bash
# From the yolo-api directory
cd yolo-api

# Download YOLOv8n model
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**Alternative**: The Ultralytics library will automatically download `yolov8n.pt` on first use if not present.

---

## Starting the Application

### Install Dependencies

```bash
cd yolo-api
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will start on `http://localhost:8000`

---

## API Documentation

Once the application is running, access the interactive Swagger UI at:

**Swagger UI**: http://localhost:8000/docs

---

## Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "healthy",
  "service": "YOLO Object Detection API",
  "model": "yolov8n.pt"
}
```

### 2. Object Detection

Upload an image for object detection:

```bash
curl -X POST "http://localhost:8000/detect?confidence=0.25" \
  -F "file=@path/to/image.jpg"
```

**Parameters:**
- `confidence` (optional): Confidence threshold (0.0-1.0, default: 0.25)
- `save_annotated` (optional): Save annotated image (default: true)

**Response:**
```json
{
  "success": true,
  "image_name": "image.jpg",
  "image_size": {
    "width": 640,
    "height": 480
  },
  "detections_count": 3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": {
        "x1": 120.5,
        "y1": 45.2,
        "x2": 340.8,
        "y2": 450.6
      }
    }
  ],
  "inference_time_ms": 45.32
}
```

### 3. Download Output Files

Download saved outputs (JSON or annotated images):

```bash
curl http://localhost:8000/outputs/image_detections.json
curl http://localhost:8000/outputs/image_annotated.jpg --output result.jpg
```

---

## Python Example

```python
import requests

# Upload and detect
with open('test.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/detect',
        files=files,
        params={'confidence': 0.3}
    )

result = response.json()
print(f"Found {result['detections_count']} objects")
for detection in result['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.2f}")
```

---

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

---

## Output Directory

Detection results are saved to `outputs/`:
- `{filename}_detections.json` - JSON with all detections
- `{filename}_annotated.jpg` - Image with bounding boxes

---

## Performance Tips

1. **Batch Processing**: Process multiple images by calling the endpoint in a loop
2. **Confidence Threshold**: Adjust confidence to reduce false positives
3. **Model Selection**: YOLOv8n is fastest; consider yolov8s/m/l/x for better accuracy

---

## Troubleshooting

### Model Not Found
If you see model loading errors:
1. Ensure `yolov8n.pt` is in the `yolo-api/` directory
2. Or let Ultralytics auto-download on first run

### Port Already in Use
Change the port when starting:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Memory Issues
YOLOv8n requires ~500MB RAM. For larger models, ensure sufficient memory.

---

## Next Steps

- Explore the interactive Swagger UI for testing: http://localhost:8000/docs
- Adjust confidence thresholds for your use case
- Integrate with your application
- Deploy with Docker (see `../multi-service-docker/`)
