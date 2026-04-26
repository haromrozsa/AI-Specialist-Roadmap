# Spring Boot ML API - Project Structure

```
springboot-ml-api/
├── pom.xml                                    # Maven configuration
├── README.md                                  # Session summary (What I Did / What I Learned)
├── USAGE.md                                   # Comprehensive usage guide
├── .gitignore                                 # Git ignore rules
├── samples/
│   └── digit.png                              # Sample test image
└── src/
    └── main/
        ├── java/com/ai/specialist/mlapi/
        │   ├── MlApiApplication.java          # Spring Boot main class
        │   ├── config/
        │   │   └── OpenApiConfig.java         # Swagger/OpenAPI configuration
        │   ├── controller/
        │   │   ├── PredictionController.java  # /predict endpoints (3 formats)
        │   │   └── HealthController.java      # /health and /model/info endpoints
        │   ├── service/
        │   │   └── ModelService.java          # ONNX model loading & inference
        │   ├── dto/
        │   │   ├── PredictionRequest.java     # Raw features request DTO
        │   │   ├── Base64PredictionRequest.java # Base64 image request DTO
        │   │   ├── PredictionResponse.java    # Prediction response DTO
        │   │   ├── ModelInfoResponse.java     # Model info response DTO
        │   │   └── HealthResponse.java        # Health check response DTO
        │   ├── util/
        │   │   └── ImagePreprocessor.java     # Image preprocessing utilities
        │   └── exception/
        │       ├── ErrorResponse.java         # Error response DTO
        │       └── GlobalExceptionHandler.java # @RestControllerAdvice
        └── resources/
            ├── application.properties         # Spring Boot configuration
            ├── logback.xml                    # Logging configuration
            └── models/
                └── mnist-12.onnx              # MNIST ONNX model
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict from 784-element float array |
| `/api/predict/image` | POST | Predict from uploaded image file |
| `/api/predict/base64` | POST | Predict from Base64-encoded image |
| `/api/health` | GET | Health check with model status |
| `/api/model/info` | GET | Model metadata and information |
| `/swagger-ui.html` | GET | Interactive API documentation |
| `/actuator/health` | GET | Detailed health check (Actuator) |
| `/actuator/metrics` | GET | Application metrics (Actuator) |
| `/actuator/prometheus` | GET | Prometheus metrics endpoint |

## Technologies Stack

- **Spring Boot 3.2.5** - Enterprise Java framework
- **ONNX Runtime 1.18.0** - ML inference engine
- **Lombok 1.18.32** - Boilerplate reduction
- **SpringDoc OpenAPI 2.5.0** - Swagger UI generation
- **Spring Boot Actuator** - Production monitoring
- **Jakarta Bean Validation** - Input validation
- **SLF4J + Logback** - Structured logging
- **Java 17** - Runtime environment

## Running the Application

```bash
# Navigate to project directory
cd springboot-ml-api

# Run with Maven
mvn spring-boot:run

# Or build and run JAR
mvn clean package
java -jar target/springboot-ml-api-1.0.0.jar
```

Access the API at: http://localhost:8080

## Next Steps

1. Test all three prediction endpoints using Swagger UI
2. Monitor application metrics via Actuator
3. Integrate with frontend applications
4. Deploy to production (Docker, Kubernetes, AWS)
