# Production-Ready FastAPI ONNX API

## Session Summary: Made an ML inference API production-ready by adding Pydantic input validation, structured logging, proper error handling, and latency measurement.

## What I Did During the Session

1. **Added Pydantic Input Validation (`schemas.py`)**:
   - Created `PredictRequest` model with strict validation for input data
   - Added `field_validator` to check feature count (must be 64) and value range (0-16)
   - Created `PredictResponse`, `HealthResponse`, and `ErrorResponse` models for typed API responses
   - Used `Field()` with descriptions for automatic OpenAPI documentation

2. **Implemented Structured Logging**:
   - Configured `logging` module with timestamp, level, and message format
   - Added dual handlers: `StreamHandler` for console and `FileHandler` for `api.log`
   - Fixed Windows Unicode encoding issue by using `encoding='utf-8'` for file and stream handlers
   - Added contextual log messages for startup, inference, and errors

3. **Built Proper Error Handling**:
   - Created custom exception handler for `RequestValidationError` (Pydantic errors)
   - Added global exception handler for unexpected errors
   - Return structured JSON error responses with timestamps
   - Used appropriate HTTP status codes (400, 422, 500, 503)

4. **Added Latency Measurement**:
   - Used `time.perf_counter()` for high-precision timing
   - Added HTTP middleware to log request/response latency for all endpoints
   - Included inference-specific latency in prediction response

5. **Created Test Suite (`test_api.py`)**:
   - Tested valid requests: single sample and batch predictions
   - Tested invalid requests: wrong feature count, out-of-range values, empty data, missing fields, malformed JSON
   - Used real data from `sklearn.datasets.load_digits` for realistic testing

6. **Fixed ONNX Model Output Handling**:
   - Discovered `skl2onnx` exports LogisticRegression with two outputs: labels and probabilities
   - Updated inference code to retrieve all model outputs instead of just the first one
   - Fixed batch processing issue where only one prediction was returned

---

## What I Learned

1. **Pydantic for API Validation**:
   - Pydantic `BaseModel` provides automatic request parsing and validation
   - `field_validator` decorator allows custom validation logic with clear error messages
   - `Field()` adds metadata for OpenAPI docs and constraints like `min_length`
   - Validation errors are automatically caught by FastAPI and can be customized with exception handlers

2. **Production Logging Best Practices**:
   - Use `logging` module instead of `print()` for configurable, structured logs
   - Always include timestamps and log levels for debugging
   - Write logs to both console (for development) and file (for production)
   - On Windows, specify `encoding='utf-8'` to handle Unicode characters (emojis)

3. **FastAPI Error Handling**:
   - `@app.exception_handler()` decorator registers custom handlers for specific exception types
   - `RequestValidationError` is raised when Pydantic validation fails
   - Return `JSONResponse` with appropriate status codes instead of raising unhandled exceptions
   - Global exception handlers catch unexpected errors and prevent stack traces in responses

4. **Latency Measurement**:
   - `time.perf_counter()` provides high-resolution timing (better than `time.time()`)
   - FastAPI middleware can wrap all requests to measure total request latency
   - Separate inference latency from total latency to identify bottlenecks
   - Include latency in API responses for client-side monitoring

5. **ONNX Model Outputs**:
   - `skl2onnx` converted models may have multiple outputs (labels + probabilities)
   - Use `session.get_outputs()` to discover all output names
   - Pass all output names to `session.run()` to get complete results
   - Scikit-learn classifiers export probabilities as list of dicts `[{class: prob}, ...]`

6. **API Testing Strategy**:
   - Test both valid inputs (single, batch) and invalid inputs (edge cases)
   - Compare predictions against known labels to verify correctness
   - Test error responses to ensure proper status codes and messages
   - Automate tests in a script for repeatable verification

---

This session provided hands-on experience making an ML API production-ready, covering Pydantic validation, structured logging, custom error handling, latency measurement, and understanding ONNX model output formats.
