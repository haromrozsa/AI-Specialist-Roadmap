"""
Evaluation script for SageMaker pipeline demo.
Evaluates trained model and outputs metrics.
"""
import json
import joblib
import argparse
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model_dir: str = "/opt/ml/processing/model",
                   output_dir: str = "/opt/ml/processing/evaluation",
                   accuracy_threshold: float = 0.90):
    """
    Evaluate trained model and generate metrics.

    Args:
        model_dir: Directory containing trained model
        output_dir: Directory to save evaluation results
        accuracy_threshold: Minimum accuracy for deployment

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Loading model...")
    model_path = Path(model_dir) / "model.pkl"
    model = joblib.load(model_path)

    logger.info("Loading test data...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Determine if model passes threshold
    deploy_approved = accuracy >= accuracy_threshold
    logger.info(f"Deployment approved: {deploy_approved} (threshold: {accuracy_threshold})")

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)

    # Prepare evaluation results
    evaluation_results = {
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "threshold": {
            "accuracy_threshold": float(accuracy_threshold),
            "deploy_approved": bool(deploy_approved)
        },
        "classification_report": report,
        "test_samples": int(len(y_test))
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_file}")

    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)

    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        accuracy_threshold=args.accuracy_threshold
    )
