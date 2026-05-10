# Usage Guide - Serverless AI Inference

This guide covers both **local testing** (no AWS account needed) and **AWS deployment** (production).

---

## 📦 Prerequisites

### For Local Testing
- Python 3.11+
- pip

### For AWS Deployment
- AWS Account
- AWS CLI configured (`aws configure`)
- Node.js 18+ (for AWS CDK)
- Docker (for Lambda packaging)

---

## 🚀 Quick Start - Local Testing

### 1. Install Dependencies

```bash
cd serverless-ai-inference
pip install -r requirements.txt
```

### 2. Generate Sample Images

```bash
python generate_samples.py
```

This creates 10 test images (digits 0-9) in `samples/` directory.

### 3. Run Local Inference

```bash
python local_test.py
```

**Expected Output:**
```
============================================================
LOCAL INFERENCE TEST RESULTS
============================================================
Image: samples\digit_5.png
Predicted Digit: 5
Confidence: 90.21%
Inference Time: 1.01ms

Top 3 Predictions:
  1. Digit 5: 90.21%
  2. Digit 3: 8.15%
  3. Digit 6: 1.25%
============================================================
```

Results are saved to `outputs/result.json`

---

## 🐳 Local Testing with LocalStack (S3 Simulation)

LocalStack provides a local AWS cloud stack for testing.

### 1. Start LocalStack

```bash
docker-compose up -d localstack
```

This creates:
- Local S3 endpoint: `http://localhost:4566`
- Buckets: `mnist-inference-input`, `mnist-inference-output`

### 2. Upload Test Image

```bash
aws --endpoint-url=http://localhost:4566 s3 cp samples/digit_5.png s3://mnist-inference-input/
```

### 3. Verify Upload

```bash
aws --endpoint-url=http://localhost:4566 s3 ls s3://mnist-inference-input/
```

### 4. Stop LocalStack

```bash
docker-compose down
```

---

## ☁️ AWS Deployment (Production)

### Option 1: Manual Lambda Deployment

#### Step 1: Package Lambda Function

```bash
# Create deployment package
mkdir lambda_package
pip install -r requirements.txt -t lambda_package/
cp lambda_handler.py lambda_package/
cd lambda_package
zip -r ../lambda_function.zip .
cd ..
```

#### Step 2: Create S3 Buckets

```bash
aws s3 mb s3://your-mnist-inference-input
aws s3 mb s3://your-mnist-inference-output
```

#### Step 3: Upload Model to S3

```bash
aws s3 cp mnist-12.onnx s3://your-bucket-name/models/
```

#### Step 4: Create Lambda Layer (Model)

```bash
mkdir -p lambda_layer/opt/ml/model
cp mnist-12.onnx lambda_layer/opt/ml/model/
cd lambda_layer
zip -r ../model_layer.zip .
cd ..

aws lambda publish-layer-version \
  --layer-name mnist-model \
  --zip-file fileb://model_layer.zip \
  --compatible-runtimes python3.11
```

#### Step 5: Create Lambda Function

```bash
aws lambda create-function \
  --function-name mnist-serverless-inference \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 60 \
  --memory-size 512 \
  --layers arn:aws:lambda:REGION:ACCOUNT:layer:mnist-model:1
```

#### Step 6: Configure S3 Trigger

```bash
aws s3api put-bucket-notification-configuration \
  --bucket your-mnist-inference-input \
  --notification-configuration file://s3-notification.json
```

`s3-notification.json`:
```json
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "arn:aws:lambda:REGION:ACCOUNT:function:mnist-serverless-inference",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [{"Name": "suffix", "Value": ".png"}]
        }
      }
    }
  ]
}
```

---

### Option 2: AWS CDK Deployment (Recommended)

#### Step 1: Install CDK

```bash
npm install -g aws-cdk
pip install -r requirements-cdk.txt
```

#### Step 2: Bootstrap CDK (First Time Only)

```bash
cdk bootstrap aws://YOUR_ACCOUNT_ID/us-east-1
```

#### Step 3: Update Account ID

Edit `app.py`:
```python
env=Environment(
    account='YOUR_ACCOUNT_ID',  # Replace with your AWS account ID
    region='us-east-1'
)
```

#### Step 4: Deploy Stack

```bash
cdk deploy
```

Review changes and confirm. CDK will create:
- S3 input bucket
- S3 output bucket
- Lambda function
- Lambda layer with model
- IAM roles
- S3 event triggers

#### Step 5: Test Deployment

```bash
aws s3 cp samples/digit_7.png s3://mnist-inference-input/
```

#### Step 6: Check Results

```bash
aws s3 ls s3://mnist-inference-output/results/
aws s3 cp s3://mnist-inference-output/results/digit_7.json -
```

#### Step 7: View Logs

```bash
aws logs tail /aws/lambda/mnist-serverless-inference --follow
```

#### Step 8: Destroy Stack (Cleanup)

```bash
cdk destroy
```

---

## 📊 Example Output

### Lambda CloudWatch Logs
```
START RequestId: abc-123-def Version: $LATEST
[INFO] Received event: {"Records": [{"s3": {"bucket": {"name": "mnist-inference-input"}, ...}}]}
[INFO] Processing file: s3://mnist-inference-input/digit_5.png
[INFO] Loading ONNX model from /opt/ml/model/mnist-12.onnx
[INFO] Model loaded in 21.45ms
[INFO] Prediction: 5 (confidence: 90.21%)
[INFO] Results saved to s3://mnist-inference-output/results/digit_5.json
END RequestId: abc-123-def
REPORT Duration: 234.56 ms  Billed Duration: 235 ms  Memory: 512 MB  Max Memory Used: 187 MB
```

### S3 Output (result.json)
```json
{
  "predicted_digit": 5,
  "confidence": 0.9021,
  "probabilities": [0.0011, 0.0000, 0.0001, 0.0815, 0.0000, 0.9021, 0.0125, 0.0000, 0.0024, 0.0003],
  "inference_time_ms": 1.23,
  "bucket": "mnist-inference-input",
  "key": "digit_5.png",
  "timestamp": "2024-01-01 12:34:56"
}
```

---

## 🔍 Troubleshooting

### Local Testing Issues

**Error: `ModuleNotFoundError: No module named 'onnxruntime'`**
```bash
pip install onnxruntime Pillow numpy
```

**Error: `FileNotFoundError: mnist-12.onnx`**
- Ensure you're running from the `serverless-ai-inference` directory
- Model should be in the same directory as the script

### AWS Deployment Issues

**Error: `AccessDenied` when uploading to S3**
- Check IAM permissions for Lambda execution role
- Ensure bucket policies allow Lambda access

**Lambda times out**
- Increase timeout in `cdk_stack.py` (currently 60s)
- Increase memory allocation (currently 512MB)

**Cold start is slow (>5 seconds)**
- Consider provisioned concurrency for Lambda
- Or use Lambda SnapStart (Java only currently)

---

## 💰 Cost Considerations

### AWS Free Tier (First 12 months)
- **Lambda**: 1M requests/month + 400,000 GB-seconds compute
- **S3**: 5GB storage + 20,000 GET requests
- **CloudWatch**: 5GB logs

### Beyond Free Tier
- **Lambda**: ~$0.20 per 1M requests (512MB, 30ms avg)
- **S3**: ~$0.023/GB/month storage
- **Data Transfer**: First 100GB/month free

**Estimated cost for 10,000 inferences/month**: **$0.50 - $2.00**

---

## 📚 Next Steps

1. **Add API Gateway**: Expose HTTP endpoint for inference
2. **Batch Processing**: Trigger on S3 prefix for bulk inference
3. **Different Models**: Replace MNIST with custom ONNX models
4. **Monitoring**: Add CloudWatch dashboards and alarms
5. **CI/CD**: Automate deployment with GitHub Actions
6. **Multi-Region**: Deploy to multiple AWS regions for low latency

---

## 🎯 Key Takeaways

✅ **Event-Driven**: S3 upload automatically triggers inference
✅ **Serverless**: No servers to manage, auto-scaling
✅ **Cost-Effective**: Pay only for inference time
✅ **Portable**: ONNX model works anywhere
✅ **Local Testing**: Develop without AWS costs
✅ **Production-Ready**: CDK for infrastructure-as-code
