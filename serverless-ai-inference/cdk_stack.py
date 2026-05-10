"""
AWS CDK Infrastructure for Serverless AI Inference.

Creates:
- S3 buckets (input, output)
- Lambda function with ONNX runtime layer
- S3 event trigger
- IAM roles and permissions
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_s3_notifications as s3n,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct


class ServerlessAIStack(Stack):
    """CDK Stack for Serverless AI Inference system."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Buckets
        # Input bucket - receives uploaded images
        input_bucket = s3.Bucket(
            self,
            'MnistInputBucket',
            bucket_name='mnist-inference-input',
            removal_policy=RemovalPolicy.DESTROY,  # For dev/demo only
            auto_delete_objects=True,  # Cleanup on stack deletion
            versioned=False,
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # Output bucket - stores inference results
        output_bucket = s3.Bucket(
            self,
            'MnistOutputBucket',
            bucket_name='mnist-inference-output',
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=False,
            public_read_access=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # Lambda Layer for ONNX model
        model_layer = lambda_.LayerVersion(
            self,
            'MnistModelLayer',
            code=lambda_.Code.from_asset('lambda_layer'),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            description='MNIST ONNX model and dependencies',
        )

        # Lambda Function
        inference_function = lambda_.Function(
            self,
            'MnistInferenceFunction',
            function_name='mnist-serverless-inference',
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler='lambda_handler.lambda_handler',
            code=lambda_.Code.from_asset('lambda_code'),
            layers=[model_layer],
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                'OUTPUT_BUCKET': output_bucket.bucket_name,
                'LOG_LEVEL': 'INFO',
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # Grant permissions
        input_bucket.grant_read(inference_function)
        output_bucket.grant_write(inference_function)

        # Add S3 event trigger
        # Trigger Lambda when .png files are uploaded to input bucket
        input_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(inference_function),
            s3.NotificationKeyFilter(suffix='.png')
        )

        # CloudWatch Logs permissions (auto-created by CDK)
        # IAM role permissions for Lambda execution
        inference_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    'logs:CreateLogGroup',
                    'logs:CreateLogStream',
                    'logs:PutLogEvents',
                ],
                resources=['*'],
            )
        )

        # Output bucket ARNs for reference
        self.input_bucket_name = input_bucket.bucket_name
        self.output_bucket_name = output_bucket.bucket_name
        self.function_arn = inference_function.function_arn
