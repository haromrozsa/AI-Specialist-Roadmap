"""
Local testing script for Lambda handler without AWS.

Simulates S3 event and tests Lambda function locally.
"""

import json
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)

# Mock Lambda handler environment
class MockContext:
    """Mock Lambda context for local testing."""
    request_id = 'local-test-request-id'
    function_name = 'mnist-inference'
    memory_limit_in_mb = 512
    invoked_function_arn = 'arn:aws:lambda:local:123456789012:function:mnist-inference'


def create_s3_event(bucket: str, key: str) -> dict:
    """
    Create a mock S3 event notification.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Mock S3 event dictionary
    """
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": "2024-01-01T00:00:00.000Z",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "bucket": {
                        "name": bucket,
                        "arn": f"arn:aws:s3:::{bucket}"
                    },
                    "object": {
                        "key": key,
                        "size": 1024
                    }
                }
            }
        ]
    }


def test_local_inference():
    """Test Lambda handler with local file without S3."""
    # Import handler after setting up environment
    # For local testing, we'll modify the handler to work with local files
    from lambda_handler_local import lambda_handler_local

    # Test with local image
    test_image = Path('samples/digit_5.png')

    if not test_image.exists():
        print(f"Error: Test image not found: {test_image}")
        print("Please create samples directory with test images first.")
        sys.exit(1)

    # Read image bytes
    with open(test_image, 'rb') as f:
        image_bytes = f.read()

    # Test inference
    result = lambda_handler_local(image_bytes)

    print("\n" + "="*60)
    print("LOCAL INFERENCE TEST RESULTS")
    print("="*60)
    print(f"Image: {test_image}")
    print(f"Predicted Digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
    print(f"\nTop 3 Predictions:")

    # Sort probabilities
    probs = result['probabilities']
    top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]

    for rank, (digit, prob) in enumerate(top3, 1):
        print(f"  {rank}. Digit {digit}: {prob:.2%}")

    print("="*60 + "\n")

    # Save result
    output_file = Path('outputs/result.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    test_local_inference()
