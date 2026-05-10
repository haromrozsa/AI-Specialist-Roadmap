#!/usr/bin/env python3
"""
AWS CDK App entry point for Serverless AI Infrastructure.

SECURITY NOTE: This file contains no credentials or secrets.
The account ID is a placeholder - you MUST replace it with your own AWS account ID.

To deploy:
    1. Update the account ID and region below
    2. cdk bootstrap  # First time only
    3. cdk deploy

To destroy:
    cdk destroy
"""

import os
from aws_cdk import App, Environment
from cdk_stack import ServerlessAIStack

app = App()

# Option 1: Use environment variables (recommended for security)
# aws_account = os.environ.get('CDK_DEFAULT_ACCOUNT')
# aws_region = os.environ.get('CDK_DEFAULT_REGION', 'us-east-1')

# Option 2: Hardcode for demo (replace with your values)
aws_account = '123456789012'  # ⚠️ PLACEHOLDER - Replace with your AWS account ID
aws_region = 'us-east-1'      # Replace with your preferred region

# Create the stack
ServerlessAIStack(
    app,
    'ServerlessAIStack',
    env=Environment(
        account=aws_account,
        region=aws_region
    ),
    description='Serverless AI inference using S3, Lambda, and ONNX Runtime'
)

app.synth()
