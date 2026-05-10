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