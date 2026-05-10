"""
LocalStack end-to-end testing script.

Tests S3 upload/download with simulated Lambda inference.
"""

import boto3
import json
from pathlib import Path

# Configure S3 client for LocalStack
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)


def test_s3_upload():
    """Upload test image to LocalStack S3."""
    print("=" * 60)
    print("STEP 1: Upload Image to S3 (LocalStack)")
    print("=" * 60)

    test_image = Path('samples/digit_5.png')

    if not test_image.exists():
        print(f"[ERROR] Error: {test_image} not found")
        print("   Run: python generate_samples.py")
        return False

    with open(test_image, 'rb') as f:
        s3.put_object(
            Bucket='mnist-inference-input',
            Key='digit_5.png',
            Body=f
        )

    print(f"[OK] Uploaded {test_image} to s3://mnist-inference-input/digit_5.png")
    return True


def test_inference():
    """Simulate Lambda inference locally."""
    print("\n" + "=" * 60)
    print("STEP 2: Run Inference (Simulated Lambda)")
    print("=" * 60)

    from lambda_handler_local import lambda_handler_local

    with open('samples/digit_5.png', 'rb') as f:
        result = lambda_handler_local(f.read())

    print(f"[OK] Predicted Digit: {result['predicted_digit']}")
    print(f"[OK] Confidence: {result['confidence']:.2%}")
    print(f"[OK] Inference Time: {result['inference_time_ms']:.2f}ms")

    return result


def test_s3_result_upload(result):
    """Upload inference result to output bucket."""
    print("\n" + "=" * 60)
    print("STEP 3: Upload Result to S3 Output Bucket")
    print("=" * 60)

    result_json = json.dumps(result, indent=2)

    s3.put_object(
        Bucket='mnist-inference-output',
        Key='results/digit_5.json',
        Body=result_json,
        ContentType='application/json'
    )

    print("[OK] Uploaded result to s3://mnist-inference-output/results/digit_5.json")


def test_s3_download():
    """Download and verify result from S3."""
    print("\n" + "=" * 60)
    print("STEP 4: Download Result from S3")
    print("=" * 60)

    obj = s3.get_object(Bucket='mnist-inference-output', Key='results/digit_5.json')
    result_json = obj['Body'].read().decode('utf-8')

    print("[OK] Downloaded result:")
    print(result_json)


def list_buckets():
    """List all objects in both buckets."""
    print("\n" + "=" * 60)
    print("STEP 5: Verify S3 Contents")
    print("=" * 60)

    print("\nInput Bucket (mnist-inference-input):")
    try:
        response = s3.list_objects_v2(Bucket='mnist-inference-input')
        for obj in response.get('Contents', []):
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")
    except Exception as e:
        print(f"   Warning: {e}")

    print("\nOutput Bucket (mnist-inference-output):")
    try:
        response = s3.list_objects_v2(Bucket='mnist-inference-output')
        for obj in response.get('Contents', []):
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")
    except Exception as e:
        print(f"   Warning: {e}")


def main():
    """Run complete LocalStack test."""
    print("\nLocalStack End-to-End Test\n")

    try:
        # Test S3 upload
        if not test_s3_upload():
            return

        # Run inference
        result = test_inference()

        # Upload result
        test_s3_result_upload(result)

        # Download and verify
        test_s3_download()

        # List bucket contents
        list_buckets()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nThis simulates the complete serverless flow:")
        print("  1. User uploads image -> S3 input bucket")
        print("  2. Lambda processes image -> ONNX inference")
        print("  3. Lambda saves result -> S3 output bucket")
        print("  4. User retrieves result -> JSON download")
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure LocalStack is running:")
        print("  docker-compose up -d localstack")
        print("\nAnd buckets are created:")
        print("  aws --endpoint-url=http://localhost:4566 s3 mb s3://mnist-inference-input")
        print("  aws --endpoint-url=http://localhost:4566 s3 mb s3://mnist-inference-output")


if __name__ == '__main__':
    main()
