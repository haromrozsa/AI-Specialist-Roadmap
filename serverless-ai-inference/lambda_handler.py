"""
AWS Lambda handler for S3-triggered MNIST inference.

Event flow: S3 Upload → Lambda Trigger → ONNX Inference → Results to S3
"""

import json
import logging
import time
from io import BytesIO
from typing import Dict, Any, List
import boto3
import numpy as np
from PIL import Image
import onnxruntime as ort

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client (works with both AWS and LocalStack)
s3_client = boto3.client('s3')

# Global session for model reuse across invocations (Lambda optimization)
ort_session = None
MODEL_PATH = '/opt/ml/model/mnist-12.onnx'  # Lambda layer path


def load_model():
    """Load ONNX model into memory (called once per cold start)."""
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
        image_bytes: Raw image bytes from S3

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

    return {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'probabilities': probabilities.tolist(),
        'inference_time_ms': round(inference_time, 2)
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler triggered by S3 upload events.

    Args:
        event: S3 event notification
        context: Lambda context object

    Returns:
        Response with status code and results
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # Extract S3 bucket and key from event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info(f"Processing file: s3://{bucket}/{key}")

        # Download image from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()

        # Preprocess image
        image_array = preprocess_image(image_bytes)

        # Run inference
        result = run_inference(image_array)

        # Add metadata
        result['bucket'] = bucket
        result['key'] = key
        result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())

        logger.info(f"Prediction: {result['predicted_digit']} (confidence: {result['confidence']:.2%})")

        # Save result to output bucket
        output_bucket = bucket.replace('-input', '-output')
        output_key = f"results/{key.split('/')[-1].replace('.png', '.json')}"

        s3_client.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )

        logger.info(f"Results saved to s3://{output_bucket}/{output_key}")

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        logger.error(f"Error processing event: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
