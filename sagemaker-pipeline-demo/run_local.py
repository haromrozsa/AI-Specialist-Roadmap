"""
Local execution script for SageMaker pipeline demonstration.

This script demonstrates the pipeline workflow locally without requiring
real AWS SageMaker infrastructure. It simulates the pipeline steps:
1. Training
2. Evaluation
3. Conditional Registration (simulated)
"""
import os
import json
import shutil
from pathlib import Path
import logging
from src.train import train_model
from src.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create output directories for pipeline artifacts."""
    dirs = {
        'model': Path('outputs/model'),
        'evaluation': Path('outputs/evaluation'),
        'logs': Path('outputs/logs')
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

    return dirs


def simulate_pipeline(n_estimators=100, test_size=0.2, accuracy_threshold=0.90):
    """
    Simulate SageMaker pipeline execution locally.

    Args:
        n_estimators: Number of trees in Random Forest
        test_size: Test split ratio
        accuracy_threshold: Minimum accuracy for model registration

    Returns:
        Pipeline execution summary
    """
    logger.info("=" * 80)
    logger.info("STARTING SAGEMAKER PIPELINE SIMULATION")
    logger.info("=" * 80)

    # Setup
    dirs = setup_directories()
    results = {
        'pipeline_name': 'iris-classification-pipeline',
        'steps': []
    }

    # Step 1: Training
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Training Model")
    logger.info("=" * 80)

    try:
        model, train_metadata = train_model(
            output_dir=str(dirs['model']),
            test_size=test_size,
            n_estimators=n_estimators
        )
        results['steps'].append({
            'step_name': 'TrainIrisClassifier',
            'status': 'SUCCESS',
            'metadata': train_metadata
        })
        logger.info("✓ Training step completed successfully")
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        results['steps'].append({
            'step_name': 'TrainIrisClassifier',
            'status': 'FAILED',
            'error': str(e)
        })
        return results

    # Step 2: Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Evaluating Model")
    logger.info("=" * 80)

    try:
        eval_results = evaluate_model(
            model_dir=str(dirs['model']),
            output_dir=str(dirs['evaluation']),
            accuracy_threshold=accuracy_threshold
        )
        results['steps'].append({
            'step_name': 'EvaluateModel',
            'status': 'SUCCESS',
            'metrics': eval_results['metrics'],
            'deploy_approved': eval_results['threshold']['deploy_approved']
        })
        logger.info("✓ Evaluation step completed successfully")
    except Exception as e:
        logger.error(f"✗ Evaluation step failed: {e}")
        results['steps'].append({
            'step_name': 'EvaluateModel',
            'status': 'FAILED',
            'error': str(e)
        })
        return results

    # Step 3: Conditional Registration (Simulated)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Conditional Model Registration")
    logger.info("=" * 80)

    accuracy = eval_results['metrics']['accuracy']
    deploy_approved = eval_results['threshold']['deploy_approved']

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Threshold: {accuracy_threshold}")
    logger.info(f"Condition met: {deploy_approved}")

    if deploy_approved:
        logger.info("✓ Model APPROVED for registration")
        logger.info("  → Model would be registered to SageMaker Model Registry")
        logger.info("  → Approval status: PendingManualApproval")
        results['steps'].append({
            'step_name': 'RegisterIrisModel',
            'status': 'SUCCESS',
            'action': 'Model registered (simulated)',
            'model_package_group': 'iris-classifier-group'
        })
    else:
        logger.info("✗ Model REJECTED - accuracy below threshold")
        logger.info("  → Model will NOT be registered")
        results['steps'].append({
            'step_name': 'CheckAccuracyCondition',
            'status': 'SKIPPED',
            'reason': f'Accuracy {accuracy:.4f} < threshold {accuracy_threshold}'
        })

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)

    logger.info(f"Total steps: {len(results['steps'])}")
    success_steps = sum(1 for s in results['steps'] if s['status'] == 'SUCCESS')
    logger.info(f"Successful steps: {success_steps}")
    logger.info(f"Pipeline status: {'SUCCESS' if success_steps >= 2 else 'FAILED'}")

    results['pipeline_status'] = 'SUCCESS' if success_steps >= 2 else 'FAILED'

    # Save pipeline results
    results_file = Path('outputs/pipeline_execution.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nPipeline results saved to {results_file}")

    logger.info("\nOutput files:")
    logger.info(f"  - Model: outputs/model/model.pkl")
    logger.info(f"  - Training metadata: outputs/model/training_metadata.json")
    logger.info(f"  - Evaluation results: outputs/evaluation/evaluation_results.json")
    logger.info(f"  - Pipeline summary: outputs/pipeline_execution.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SageMaker pipeline simulation locally")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--accuracy-threshold", type=float, default=0.90, help="Accuracy threshold")

    args = parser.parse_args()

    results = simulate_pipeline(
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        accuracy_threshold=args.accuracy_threshold
    )

    # Exit with appropriate code
    exit(0 if results['pipeline_status'] == 'SUCCESS' else 1)
