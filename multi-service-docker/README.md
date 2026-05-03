# Multi-Service Docker Deployment

## Session Summary: Dockerized two existing ML services (YOLO object detection and MNIST classification) and orchestrated them using Docker Compose, demonstrating production-ready containerization for AI/ML microservices.

---

## 🏗️ Architecture Overview

This project demonstrates a **multi-service architecture** with two independent AI/ML microservices:

```
┌─────────────────────────────────────────────────┐
│          Docker Compose Network (ml-network)    │
│                                                 │
│  ┌──────────────────┐    ┌──────────────────┐ │
│  │  YOLO Service    │    │  MNIST Service   │ │
│  │  (FastAPI)       │    │  (Spring Boot)   │ │
│  │  Port: 8000      │    │  Port: 8080      │ │
│  │  YOLOv8n model   │    │  ONNX Runtime    │ │
│  └──────────────────┘    └──────────────────┘ │
│          │                        │            │
│          ▼                        ▼            │
│    outputs/ logs            logs/ directory    │
└─────────────────────────────────────────────────┘
```

---

## 📦 Services

### 1. **YOLO Object Detection API** (Python/FastAPI)
- **Technology**: FastAPI, Ultralytics YOLOv8, OpenCV
- **Port**: `8000`
- **Endpoints**:
  - `GET /` - Health check
  - `POST /detect` - Object detection endpoint
  - `GET /outputs/{filename}` - Download results
- **Features**: Multi-class object detection with bounding boxes and confidence scores

### 2. **MNIST Classification API** (Java/Spring Boot)
- **Technology**: Spring Boot, ONNX Runtime, Java 25
- **Port**: `8080`
- **Endpoints**:
  - `GET /actuator/health` - Health check
  - `GET /api/model/info` - Model information
  - `POST /api/classify` - Digit classification
- **Features**: Handwritten digit recognition with ONNX model inference

---

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB+ free disk space

### 0. Download Required Models

Before building the Docker images, download both models:

```bash
# Download MNIST model for Spring Boot service
cd ../springboot-ml-api/src/main/resources/models
curl -L -o mnist-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx

# Download YOLOv8 model for FastAPI service
cd ../../../../yolo-api
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Return to docker directory
cd ../multi-service-docker
```

**Note**: Both models are excluded from git via `.gitignore` to keep repository size small.

### 1. Build and Start Services
```bash
cd multi-service-docker
docker-compose up --build
```

### 2. Verify Services

**YOLO Service:**
```bash
curl http://localhost:8000/
```

**MNIST Service:**
```bash
curl http://localhost:8080/actuator/health
```

### 3. Stop Services
```bash
docker-compose down
```

---

## 🧪 Testing the Services

### YOLO Object Detection

**Test with an image:**
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.25" \
  -F "file=@test-image.jpg" \
  | jq
```

**Expected Response:**
```json
{
  "success": true,
  "image_name": "test-image.jpg",
  "image_size": {"width": 640, "height": 480},
  "detections_count": 3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": {"x1": 120.5, "y1": 45.2, "x2": 340.8, "y2": 450.6}
    }
  ],
  "inference_time_ms": 45.32
}
```

### MNIST Classification

**Test with raw features:**
```bash
curl -X POST "http://localhost:8080/api/classify" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.0, ..., 0.0]}' \
  | jq
```

**API Documentation (Swagger UI):**
```
http://localhost:8080/swagger-ui.html
```

---

## 📂 Project Structure

```
multi-service-docker/
├── docker-compose.yml        # Orchestration configuration
├── .env.example              # Environment variables template
├── README.md                 # This file
└── logs/                     # Persistent logs (created at runtime)
    ├── yolo-outputs/         # YOLO detection results
    └── mnist-logs/           # Spring Boot logs

yolo-api/
├── Dockerfile                # YOLO service container
├── .dockerignore
├── app/                      # FastAPI application
├── requirements.txt
└── yolov8n.pt               # Pre-trained model

springboot-ml-api/
├── Dockerfile                # MNIST service container
├── .dockerignore
├── pom.xml                   # Maven configuration
├── src/                      # Spring Boot application
└── models/                   # ONNX models
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Edit as needed:
```env
YOLO_PORT=8000
MNIST_PORT=8080
JAVA_OPTS=-Xms512m -Xmx1024m
LOG_LEVEL=INFO
```

### Volume Mounts

**Logs persist to host machine:**
- YOLO outputs: `./logs/yolo-outputs/`
- MNIST logs: `./logs/mnist-logs/`

---

## 🐳 Docker Commands Reference

### Build only (without starting)
```bash
docker-compose build
```

### Start in detached mode
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f yolo-service
docker-compose logs -f mnist-service
```

### Restart a specific service
```bash
docker-compose restart yolo-service
```

### Remove containers and networks
```bash
docker-compose down
```

### Remove containers, networks, and volumes
```bash
docker-compose down -v
```

---

## 🩺 Health Checks

Both services include built-in health checks:

**YOLO Service:**
- Endpoint: `GET /`
- Interval: 30s
- Start period: 40s

**MNIST Service:**
- Endpoint: `GET /actuator/health`
- Interval: 30s
- Start period: 60s

View health status:
```bash
docker-compose ps
```

---

## 🌐 Service Communication

Services share a network (`ml-network`) for inter-service communication:

**From YOLO container to MNIST:**
```bash
curl http://mnist-service:8080/actuator/health
```

**From MNIST container to YOLO:**
```bash
curl http://yolo-service:8000/
```

---

## 🛠️ Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs <service-name>

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Port already in use
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

### Out of disk space
```bash
# Clean up Docker
docker system prune -a
```

---

## What I Did During the Session

1. **Added Dockerfiles to Existing Services**:
   - Created multi-stage `Dockerfile` for `yolo-api/` with Python 3.11-slim base, OpenCV dependencies, and optimized layer caching
   - Created multi-stage `Dockerfile` for `springboot-ml-api/` with Maven build stage (JDK 25) and runtime stage (JRE 25)
   - Used `.dockerignore` files to exclude unnecessary files (IDE configs, logs, test data) from build context

2. **Configured Docker Compose Orchestration**:
   - Defined two services (`yolo-service`, `mnist-service`) with proper port mappings (8000, 8080)
   - Created shared network (`ml-network`) with bridge driver for service-to-service communication
   - Added volume mounts for persistent logs (`./logs/yolo-outputs`, `./logs/mnist-logs`)

3. **Implemented Health Checks**:
   - Added HTTP health checks for both services using native endpoints (`/` for YOLO, `/actuator/health` for MNIST)
   - Configured appropriate intervals (30s) and start periods (40s for Python, 60s for Java) to account for startup time
   - Set restart policy to `unless-stopped` for production resilience

4. **Created Environment Configuration**:
   - Added `.env.example` template for configurable ports, JVM memory settings, and logging levels
   - Configured environment variables for Python unbuffered output and Spring profiles
   - Set Java heap sizes (`-Xms512m -Xmx1024m`) for optimal container memory usage

5. **Documented Usage and Testing**:
   - Provided `docker-compose up --build` quick start guide with example `curl` commands
   - Documented API endpoints, expected responses, and Swagger UI access
   - Added troubleshooting section for common issues (port conflicts, disk space, health check failures)

---

## What I Learned

1. **Multi-Stage Docker Builds Reduce Image Size**:
   - Separating build dependencies (Maven, compiler tools) from runtime reduces final image size by 60-70%
   - Python services benefit from `--user` pip installs copied between stages to avoid system-wide dependencies
   - Java services can drop from 800MB (JDK) to 300MB (JRE) by using separate builder and runtime base images

2. **Docker Compose Simplifies Multi-Service Orchestration**:
   - Named networks (`ml-network`) enable DNS-based service discovery (services reference each other by name, not IP)
   - Volume mounts with relative paths (`./logs/yolo-outputs:/app/outputs`) persist data outside containers
   - Health checks ensure dependent services wait for upstream services to be ready before accepting traffic

3. **Health Checks Are Critical for Production Containers**:
   - HTTP health checks (`CMD curl -f http://localhost:8080/actuator/health`) provide actual service health vs process liveness
   - Start periods (40s-60s) prevent false failures during model loading and JVM warmup
   - Retries (3) and intervals (30s) balance responsiveness with avoiding flapping during transient issues

4. **Environment Variables Enable Configuration Flexibility**:
   - `.env` files keep secrets and environment-specific settings out of docker-compose.yml
   - `JAVA_OPTS` allows tuning JVM memory without rebuilding the image
   - `PYTHONUNBUFFERED=1` ensures real-time log output for debugging (no buffering delays)

5. **Dockerization Requires Understanding Application Lifecycle**:
   - Spring Boot's fat JAR pattern requires copying `target/*.jar` from builder stage
   - FastAPI's `uvicorn` needs `--host 0.0.0.0` to bind outside the container (not just localhost)
   - Model files (YOLO weights, ONNX models) must be copied during build or mounted as volumes at runtime

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Spring Boot Docker Guide](https://spring.io/guides/gs/spring-boot-docker/)

---

**🎉 Success Criteria**: Both containers run simultaneously, respond to health checks, and serve predictions independently while sharing a common network.
