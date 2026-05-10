"""
Local version of Lambda handler for testing without AWS/S3.

This module provides the same inference logic without S3 dependencies.
"""

import logging
import time
from typing import Dict, Any
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global session for model reuse
ort_session = None
MODEL_PATH = 'mnist-12.onnx'


def load_model():
    """Load ONNX model into memory."""
    global ort_session
    if ort_session is None:
        logger.info(f"Loading ONNX model from {MODEL_PATH}")
        start_time = time.time()
        ort_session = ort.InferenceSession(MODEL_PATH)
        load_time = (time.time() - start_time) * 1000
        logger.info(f"Model loaded in {load_time:.2f}ms")
    return ort_session


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for MNIST model.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Normalized numpy array shaped [1, 1, 28, 28]
    """
    # Load image and convert to grayscale
    img = Image.open(BytesIO(image_bytes)).convert('L')

    # Resize to 28x28
    try:
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback for older Pillow versions
        img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Reshape to [1, 1, 28, 28] (batch_size, channels, height, width)
    img_array = img_array.reshape(1, 1, 28, 28)

    logger.info(f"Preprocessed image shape: {img_array.shape}")
    return img_array


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / np.sum(exp_logits)


def run_inference(image_array: np.ndarray) -> Dict[str, Any]:
    """
    Run ONNX inference on preprocessed image.

    Args:
        image_array: Preprocessed image array [1, 1, 28, 28]

    Returns:
        Dictionary with prediction results
    """
    session = load_model()

    # Get input name dynamically
    input_name = session.get_inputs()[0].name

    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: image_array})
    inference_time = (time.time() - start_time) * 1000

    # Extract logits (first output)
    logits = outputs[0][0]

    # Compute probabilities
    probabilities = softmax(logits)

    # Get top prediction
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    logger.info(f"Predicted digit: {predicted_digit} (confidence: {confidence:.2%})")

    return {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'probabilities': probabilities.tolist(),
        'inference_time_ms': round(inference_time, 2),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    }


def lambda_handler_local(image_bytes: bytes) -> Dict[str, Any]:
    """
    Local version of Lambda handler for testing.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Inference results dictionary
    """
    try:
        # Preprocess image
        image_array = preprocess_image(image_bytes)

        # Run inference
        result = run_inference(image_array)

        return result

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise
