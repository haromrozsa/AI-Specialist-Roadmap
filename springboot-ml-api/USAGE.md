# Usage Guide - Spring Boot ML API

## Prerequisites

### System Requirements
- Java 25
- Maven 3.6+

### Download MNIST Model (Required)

Before running the application, download the MNIST ONNX model:

```bash
# Navigate to the models directory
cd src/main/resources/models

# Download the model from ONNX Model Zoo
curl -L -o mnist-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
```

**Alternative**: Download manually from [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) and place it in `src/main/resources/models/mnist-12.onnx`

---

## Starting the Application

### Run the Application

```bash
cd springboot-ml-api
mvn spring-boot:run
```

The API will start on `http://localhost:8080`

### Build Standalone JAR

```bash
mvn clean package
java -jar target/springboot-ml-api-1.0.0.jar
```

## API Documentation

Once the application is running, access the interactive Swagger UI at:

**Swagger UI**: http://localhost:8080/swagger-ui.html
**OpenAPI Docs**: http://localhost:8080/api-docs

## Endpoints

### 1. Health Check

```bash
curl http://localhost:8080/api/health
```

**Response:**
```json
{
  "status": "UP",
  "modelStatus": "LOADED",
  "version": "1.0.0",
  "timestamp": "2026-04-26T13:30:00Z"
}
```

### 2. Model Information

```bash
curl http://localhost:8080/api/model/info
```

**Response:**
```json
{
  "modelName": "MNIST Digit Classifier",
  "version": "1.0",
  "inputName": "Input3",
  "outputName": "Plus214_Output_0",
  "inputShape": "[1, 1, 28, 28]",
  "outputClasses": 10,
  "description": "Pre-trained MNIST model from ONNX Model Zoo for handwritten digit classification"
}
```

## Prediction Endpoints

### Method 1: Raw Features Array

Predict from a flattened array of 784 pixel values (28x28 image).

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, 0.0, 0.0, ... (784 values total)]
  }'
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9876,
  "probabilities": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.9876, 0.009, 0.001],
  "inferenceTimeMs": 5
}
```

### Method 2: Image File Upload

Upload an image file (PNG, JPG, etc.) containing a handwritten digit.

```bash
curl -X POST http://localhost:8080/api/predict/image  -F "file=@samples/digit.png"
```

**Windows PowerShell:**
```powershell
curl.exe -X POST http://localhost:8080/api/predict/image `
  -F "file=@samples/digit.png"
```

**Response:**
```json
{
  "prediction": 5,
  "confidence": 0.9823,
  "probabilities": [0.001, 0.002, 0.003, 0.004, 0.005, 0.9823, 0.001, 0.001, 0.001, 0.001],
  "inferenceTimeMs": 12
}
```

### Method 3: Base64-Encoded Image

Send a Base64-encoded image string.

```bash
curl -X POST http://localhost:8080/api/predict/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA..."
  }'
```

**Or without the data URI prefix:**
```bash
curl -X POST http://localhost:8080/api/predict/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAA..."
  }'
```

**Response:**
```json
{
  "prediction": 3,
  "confidence": 0.9654,
  "probabilities": [0.001, 0.002, 0.003, 0.9654, 0.005, 0.006, 0.007, 0.008, 0.009, 0.001],
  "inferenceTimeMs": 8
}
```

## Python Example - Base64 Image

```python
import requests
import base64

# Read and encode image
with open('samples/digit.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    'http://localhost:8080/api/predict/base64',
    json={'image': image_data}
)

print(response.json())
```

## Python Example - File Upload

```python
import requests

# Upload file
with open('samples/digit.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8080/api/predict/image',
        files=files
    )

print(response.json())
```

## Actuator Endpoints

Spring Boot Actuator provides production-ready monitoring endpoints:

```bash
# Health check (detailed)
curl http://localhost:8080/actuator/health

# Application info
curl http://localhost:8080/actuator/info

# Metrics
curl http://localhost:8080/actuator/metrics

# Prometheus metrics
curl http://localhost:8080/actuator/prometheus
```

## Error Handling

The API returns structured error responses:

**Example - Validation Error:**
```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'
```

**Response (400 Bad Request):**
```json
{
  "timestamp": "2026-04-26T13:30:00Z",
  "status": 400,
  "error": "Validation Failed",
  "message": "Invalid input parameters",
  "details": {
    "features": "Features array must contain exactly 784 values (28x28)"
  }
}
```

## Performance Tips

1. **Image Format**: PNG and BMP are faster to process than JPEG
2. **Image Size**: Pre-resize images to 28x28 for faster preprocessing
3. **Batch Processing**: For multiple images, consider implementing a batch endpoint
4. **Caching**: The model is loaded once at startup and reused for all requests

## Troubleshooting

### Model Not Found
If you see "Model file not found" error:
1. Ensure `mnist-12.onnx` is in `src/main/resources/models/`
2. Or update `model.path` in `application.properties`

### Port Already in Use
Change the port in `application.properties`:
```properties
server.port=8081
```

### Memory Issues
Increase JVM heap size:
```bash
java -Xmx2g -jar target/springboot-ml-api-1.0.0.jar
```

## Next Steps

- Explore the interactive Swagger UI for testing all endpoints
- Monitor application metrics via Actuator
- Integrate with your frontend application
- Deploy to production (Docker, Kubernetes, Cloud)
