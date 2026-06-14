"""
Simple training script for SageMaker pipeline demo.
Trains a Random Forest classifier on Iris dataset.
"""
import json
import joblib
import argparse
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(output_dir: str = "/opt/ml/model", test_size: float = 0.2, n_estimators: int = 100):
    """
    Train Random Forest on Iris dataset.

    Args:
        output_dir: Directory to save trained model
        test_size: Fraction of data for testing
        n_estimators: Number of trees in random forest
    """
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=42
    )

    logger.info(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "model.pkl"
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")

    # Save training metadata
    metadata = {
        "n_estimators": n_estimators,
        "test_size": test_size,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(iris.target_names)
    }

    metadata_file = output_path / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")

    return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_estimators=args.n_estimators
    )
