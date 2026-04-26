# Spring Boot ML API - MNIST Digit Classification

> ⚠️ **Note**: This is a demonstration/learning project. For production deployments, secure Actuator endpoints, add authentication/authorization, and review security configurations.

## Session Summary: Built a production-ready Spring Boot REST API for MNIST digit classification using ONNX Runtime, demonstrating enterprise-grade ML inference with multiple input formats, comprehensive monitoring, and interactive API documentation.

## What I Did During the Session

1. **Architected a Production-Ready Spring Boot ML API**:
    - Created a layered architecture with clear separation: Controller → Service → Utility
    - Configured Maven with Spring Boot 3.2.5, ONNX Runtime 1.18.0, Lombok 1.18.32, and SpringDoc OpenAPI 2.5.0
    - Implemented dependency injection with `@Service`, `@RestController`, and `@Configuration` annotations
    - Added lifecycle management with `@PostConstruct` for model initialization and `@PreDestroy` for cleanup
    - Designed three prediction endpoints supporting raw features, file upload, and Base64-encoded images

2. **Integrated ONNX Runtime with Spring Boot Lifecycle**:
    - Built `ModelService` as a singleton Spring bean managing `OrtEnvironment` and `OrtSession`
    - Implemented model loading from both classpath resources and file system paths
    - Added automatic session cleanup on application shutdown to prevent memory leaks
    - Extracted model metadata (input/output names) for dynamic inference configuration
    - Wrapped ONNX tensor operations in try-with-resources for proper native memory management

3. **Implemented Multiple Input Formats for Maximum Flexibility**:
    - **Raw Features**: `POST /api/predict` accepting 784-element float arrays via JSON
    - **Image Upload**: `POST /api/predict/image` with `multipart/form-data` file upload
    - **Base64 Image**: `POST /api/predict/base64` accepting Base64-encoded image strings (with or without data URI prefix)
    - Unified preprocessing pipeline converting all formats to `[1, 1, 28, 28]` tensors
    - Consistent response format with prediction, confidence, probabilities, and inference time

4. **Added Enterprise Features for Production Readiness**:
    - **Validation**: Jakarta Bean Validation with `@Valid`, `@NotNull`, `@NotBlank`, and `@Size` constraints
    - **Exception Handling**: Global `@RestControllerAdvice` catching validation errors, illegal arguments, and unexpected exceptions
    - **Health Checks**: `/api/health` endpoint reporting service and model status with timestamps
    - **Model Info**: `/api/model/info` endpoint exposing model metadata (name, version, input/output names, shape)
    - **Actuator Metrics**: Enabled Spring Boot Actuator with health, info, metrics, and Prometheus endpoints

5. **Configured Swagger/OpenAPI for Interactive Documentation**:
    - Integrated SpringDoc OpenAPI 3 with automatic schema generation from DTOs
    - Added `@Tag`, `@Operation`, `@Schema`, and `@Parameter` annotations for rich documentation
    - Configured Swagger UI at `/swagger-ui.html` with sorted operations and tags
    - Provided example values and detailed descriptions for all request/response models
    - Enabled interactive API testing directly from the browser

## What I Learned

1. **Spring Boot's Powerful Abstraction for ML APIs**:
    - Spring Boot's dependency injection eliminates manual singleton management — `@Service` beans are automatically singletons
    - Lifecycle hooks (`@PostConstruct`, `@PreDestroy`) integrate seamlessly with ONNX Runtime's resource management
    - Spring Boot auto-configures Jackson for JSON serialization, eliminating manual DTO → JSON mapping
    - `@RestController` combines `@Controller` + `@ResponseBody`, automatically serializing return values to JSON
    - Spring Boot's embedded Tomcat means zero external server configuration — just run the JAR

2. **Jakarta Bean Validation for Type-Safe Request Handling**:
    - `@Valid` on controller parameters triggers automatic validation before method execution
    - Constraint annotations (`@NotNull`, `@Size`, `@NotBlank`) are declarative and self-documenting
    - `MethodArgumentNotValidException` provides structured field-level error details via `BindingResult`
    - Custom validators can be created with `@Constraint` for domain-specific validation logic
    - Validation happens at the HTTP layer, preventing invalid data from reaching business logic

3. **Spring Boot Actuator for Production Observability**:
    - Actuator provides `/actuator/health`, `/actuator/metrics`, and `/actuator/prometheus` endpoints out of the box
    - `management.endpoints.web.exposure.include` controls which endpoints are publicly accessible
    - Health checks can be customized with `HealthIndicator` implementations for model-specific status
    - Metrics are automatically collected for HTTP requests, JVM memory, CPU, and threading
    - Prometheus endpoint enables seamless integration with monitoring stacks (Prometheus + Grafana)

4. **SpringDoc OpenAPI for Zero-Configuration API Docs**:
    - SpringDoc scans `@RestController` classes and generates OpenAPI 3.0 schemas automatically
    - Swagger UI is served at runtime without requiring external Swagger Codegen or editor tools
    - `@Schema` annotations on DTOs control example values, descriptions, and required fields in the docs
    - `@Operation` and `@ApiResponse` annotations document endpoint behavior and HTTP status codes
    - The interactive UI supports "Try it out" for live API testing — no need for Postman during development

5. **Multipart File Upload Handling in Spring Boot**:
    - `@RequestParam("file") MultipartFile` automatically parses `multipart/form-data` requests
    - `MultipartFile.getInputStream()` provides direct access to uploaded bytes without temporary files
    - `spring.servlet.multipart.max-file-size` and `max-request-size` properties enforce upload size limits
    - `MaxUploadSizeExceededException` can be caught globally for clean error responses
    - Base64 encoding is an alternative to file upload for JSON-only APIs, avoiding multipart complexity

This session provided hands-on experience building a production-grade Spring Boot ML API with ONNX Runtime, implementing multiple input formats (features, file upload, Base64), integrating enterprise features (validation, exception handling, health checks, metrics), and generating interactive API documentation with Swagger/OpenAPI — demonstrating that Spring Boot is a first-class platform for serving ML models at scale.
