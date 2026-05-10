# LocalStack Testing Guide

## 🐳 Quick Start

### 1. Start LocalStack

```bash
docker-compose up -d localstack
```

Wait ~10 seconds for LocalStack to be ready (check with `docker ps`).

### 2. Create S3 Buckets

```bash
aws --endpoint-url=http://localhost:4566 s3 mb s3://mnist-inference-input
aws --endpoint-url=http://localhost:4566 s3 mb s3://mnist-inference-output
```

### 3. Verify Buckets

```bash
aws --endpoint-url=http://localhost:4566 s3 ls
```

**Expected output:**
```
2026-05-10 12:53:40 mnist-inference-input
2026-05-10 12:53:53 mnist-inference-output
```

---

## ⚠️ Known Issue: AWS CLI v2 Upload Error

**Error you might see:**
```
InvalidRequest: The value specified in the x-amz-trailer header is not supported
```

### Solution 1: Use Python boto3 (Recommended)

Create a simple upload script:

```python
# upload_to_localstack.py
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

# Upload file
with open('samples/digit_5.png', 'rb') as f:
    s3.put_object(
        Bucket='mnist-inference-input',
        Key='digit_5.png',
        Body=f
    )

print("Uploaded digit_5.png")

# List files
response = s3.list_objects_v2(Bucket='mnist-inference-input')
for obj in response.get('Contents', []):
    print(f"  - {obj['Key']}")
```

Run it:
```bash
python upload_to_localstack.py
```

### Solution 2: Disable AWS CLI v2 Checksums

Set environment variable before AWS CLI commands:

**Linux/Mac:**
```bash
export AWS_EC2_METADATA_DISABLED=true
aws --endpoint-url=http://localhost:4566 s3 cp samples/digit_5.png s3://mnist-inference-input/
```

**Windows (PowerShell):**
```powershell
$env:AWS_EC2_METADATA_DISABLED="true"
aws --endpoint-url=http://localhost:4566 s3 cp samples/digit_5.png s3://mnist-inference-input/
```

**Windows (CMD):**
```cmd
set AWS_EC2_METADATA_DISABLED=true
aws --endpoint-url=http://localhost:4566 s3 cp samples/digit_5.png s3://mnist-inference-input/
```

### Solution 3: Use Docker Exec (Always Works)

```bash
# Copy file into container
docker cp samples/digit_5.png serverless-ai-localstack:/tmp/digit_5.png

# Upload from inside container
docker exec serverless-ai-localstack  aws --endpoint-url=http://localhost:4566 s3 cp /tmp/digit_5.png s3://mnist-inference-input/
```

---

## ✅ Testing End-to-End (Without Lambda)

Since Lambda execution in LocalStack requires the Pro version, we focus on S3 testing:

### 1. Upload Image

```python
# test_localstack.py
import boto3
import json

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

# Upload test image
with open('samples/digit_5.png', 'rb') as f:
    s3.put_object(Bucket='mnist-inference-input', Key='digit_5.png', Body=f)
    print("✓ Uploaded digit_5.png")

# Simulate Lambda result (manually run inference)
from lambda_handler_local import lambda_handler_local

with open('samples/digit_5.png', 'rb') as f:
    result = lambda_handler_local(f.read())

# Upload result to output bucket
s3.put_object(
    Bucket='mnist-inference-output',
    Key='results/digit_5.json',
    Body=json.dumps(result, indent=2),
    ContentType='application/json'
)
print("✓ Uploaded result to output bucket")

# Download and verify result
obj = s3.get_object(Bucket='mnist-inference-output', Key='results/digit_5.json')
result_json = obj['Body'].read().decode('utf-8')
print("\n" + "="*60)
print("RESULT FROM S3:")
print("="*60)
print(result_json)
```

Run:
```bash
python test_localstack.py
```

### 2. Verify Files in S3

```python
# list_buckets.py
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

print("Input Bucket:")
response = s3.list_objects_v2(Bucket='mnist-inference-input')
for obj in response.get('Contents', []):
    print(f"  - {obj['Key']} ({obj['Size']} bytes)")

print("\nOutput Bucket:")
response = s3.list_objects_v2(Bucket='mnist-inference-output')
for obj in response.get('Contents', []):
    print(f"  - {obj['Key']} ({obj['Size']} bytes)")
```

---

## 🛑 Stop LocalStack

```bash
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
rm -rf localstack-data/
```

---

## 💡 Why LocalStack?

- ✅ **Free S3 Testing**: Full S3 API compatibility
- ✅ **No AWS Costs**: Runs 100% locally
- ✅ **Fast Iteration**: No network latency
- ❌ **Lambda Limitations**: Free version doesn't support full Lambda execution (use local_test.py instead)

---

## 🎯 Summary

**For full end-to-end testing:**
1. Use LocalStack for S3 operations (upload/download)
2. Use `local_test.py` for Lambda simulation
3. Use `test_localstack.py` to combine both

**For real deployment:**
- Use AWS CDK: `cdk deploy`
- Test with real S3: `aws s3 cp samples/digit_5.png s3://your-bucket/`
