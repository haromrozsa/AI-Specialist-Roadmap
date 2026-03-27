from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import onnxruntime as ort
import numpy as np
import logging
import time
import sys
from datetime import datetime
from contextlib import asynccontextmanager

from schemas import PredictRequest, PredictResponse, HealthResponse, ErrorResponse

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)),
        logging.FileHandler("api.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Global Model Session
# ─────────────────────────────────────────────────────────────
session = None
input_name = None
output_name = None

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global session, input_name, output_name

    logger.info("🚀 Starting API server...")
    try:
        session = ort.InferenceSession("model.onnx")
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.info("✅ ONNX model loaded successfully")
        logger.info(f"   Input: {input_name}, Shape: {session.get_inputs()[0].shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    yield

    logger.info("👋 Shutting down API server...")


app = FastAPI(
    title="Digit Recognition API",
    description="Production-ready ML API for digit classification",
    version=API_VERSION,
    lifespan=lifespan
)


# ─────────────────────────────────────────────────────────────
# Exception Handlers
# ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    timestamp = datetime.now().isoformat()
    errors = exc.errors()

    logger.warning(f"⚠️  Validation error: {errors}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(errors),
            "timestamp": timestamp
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    timestamp = datetime.now().isoformat()

    logger.error(f"❌ Unhandled exception: {type(exc).__name__}: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
            "timestamp": timestamp
        }
    )


# ─────────────────────────────────────────────────────────────
# Middleware for Request Logging
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and response times."""
    start_time = time.perf_counter()

    # Process request
    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"📨 {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {latency_ms:.2f}ms"
    )

    return response


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=session is not None,
        version=API_VERSION
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Run inference on input data.

    - **data**: 2D array where each row is a sample with 64 features (8x8 pixel values, 0-16)
    - Returns predicted digit class (0-9) with confidence score
    """
    if session is None:
        logger.error("❌ Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Measure inference time
    start_time = time.perf_counter()

    try:
        # Convert and normalize input
        input_array = np.array(request.data, dtype=np.float32) / 16.0

        logger.info(f"🔮 Running inference on {len(request.data)} sample(s)")

        # Run inference - get ALL outputs (labels + probabilities)
        output_names = [o.name for o in session.get_outputs()]
        results = session.run(output_names, {input_name: input_array})

        # First output: predicted labels
        predictions = results[0]
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()

        # Second output: probabilities (dict per sample)
        if len(results) > 1:
            prob_dicts = results[1]
            confidence = [max(p.values()) for p in prob_dicts]
        else:
            confidence = [1.0] * len(predictions)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"✅ Predictions: {predictions} | "
            f"Confidence: {[f'{c:.2%}' for c in confidence]} | "
            f"Inference: {latency_ms:.2f}ms"
        )

        return PredictResponse(
            prediction=predictions,
            confidence=[round(c, 4) for c in confidence],
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# ─────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 50)
    logger.info("🎯 Digit Recognition API - Production Mode")
    logger.info("=" * 50)

    uvicorn.run(app, host="127.0.0.1", port=8000)