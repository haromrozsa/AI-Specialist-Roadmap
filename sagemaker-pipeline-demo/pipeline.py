"""
SageMaker Pipeline Definition for Iris Classification.

This pipeline demonstrates a simple ML workflow:
1. Training: Train Random Forest on Iris dataset
2. Evaluation: Evaluate model performance
3. Conditional Registration: Register model if accuracy meets threshold
"""
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pipeline(
    pipeline_name: str = "iris-classification-pipeline",
    region: str = "us-east-1",
    role: str = None,
    bucket: str = "sagemaker-pipeline-demo"
):
    """
    Create SageMaker Pipeline for Iris classification.

    Args:
        pipeline_name: Name of the pipeline
        region: AWS region
        role: IAM role ARN for SageMaker
        bucket: S3 bucket for artifacts

    Returns:
        SageMaker Pipeline object
    """
    # Pipeline parameters
    n_estimators = ParameterInteger(name="NEstimators", default_value=100)
    test_size = ParameterFloat(name="TestSize", default_value=0.2)
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.90)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.large")

    # Use provided role or get default
    if role is None:
        try:
            role = get_execution_role()
        except Exception:
            # For LocalStack, use dummy role
            role = "arn:aws:iam::000000000000:role/SageMakerRole"

    logger.info(f"Using role: {role}")

    # Step 1: Training
    sklearn_estimator = SKLearn(
        entry_point="src/train.py",
        role=role,
        instance_type=instance_type,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={
            "n-estimators": n_estimators,
            "test-size": test_size
        },
        output_path=f"s3://{bucket}/model-output"
    )

    training_step = TrainingStep(
        name="TrainIrisClassifier",
        estimator=sklearn_estimator,
        description="Train Random Forest on Iris dataset"
    )

    # Step 2: Evaluation
    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3"
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=sklearn_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/evaluation-output"
            )
        ],
        job_arguments=[
            "--accuracy-threshold", str(accuracy_threshold.default_value)
        ],
        description="Evaluate trained model"
    )

    # Property file for conditional logic
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_results.json"
    )

    evaluation_step.add_property_files(evaluation_report)

    # Step 3: Conditional Model Registration
    # Condition: accuracy >= threshold
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_report.json_path("metrics.accuracy"),
        right=accuracy_threshold
    )

    # Model registration step (executed only if condition is true)
    register_step = RegisterModel(
        name="RegisterIrisModel",
        estimator=sklearn_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="iris-classifier-group",
        approval_status="PendingManualApproval",
        description="Iris classification model"
    )

    # Conditional step
    condition_step = ConditionStep(
        name="CheckAccuracyCondition",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[],
        description="Register model only if accuracy meets threshold"
    )

    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[n_estimators, test_size, accuracy_threshold, instance_type],
        steps=[training_step, evaluation_step, condition_step],
        sagemaker_session=None  # Will use default session
    )

    return pipeline


def get_pipeline_definition(pipeline):
    """
    Get pipeline definition as JSON.

    Args:
        pipeline: SageMaker Pipeline object

    Returns:
        Pipeline definition as JSON string
    """
    import json
    definition = json.loads(pipeline.definition())
    return json.dumps(definition, indent=2)


if __name__ == "__main__":
    # Create pipeline
    pipeline = create_pipeline()

    # Print pipeline definition
    logger.info("Pipeline created successfully!")
    logger.info(f"Pipeline name: {pipeline.name}")
    logger.info(f"Steps: {[step.name for step in pipeline.steps]}")

    # Save pipeline definition
    definition = get_pipeline_definition(pipeline)
    with open("pipeline_definition.json", "w") as f:
        f.write(definition)
    logger.info("Pipeline definition saved to pipeline_definition.json")
