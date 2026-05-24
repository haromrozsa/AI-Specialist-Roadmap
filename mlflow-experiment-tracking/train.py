"""
Simple MLflow experiment tracking demo with scikit-learn
Tracks parameters, metrics, dataset version, and input statistics
"""
import hashlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_dataset_version(X):
    """Generate dataset version hash"""
    data_hash = hashlib.md5(X.tobytes()).hexdigest()[:8]
    return f"v1_{data_hash}"


def calculate_data_stats(X, feature_names):
    """Calculate input data statistics"""
    df = pd.DataFrame(X, columns=feature_names)
    stats = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict()
    }
    return stats


def train_model(n_estimators=100, max_depth=None, min_samples_split=2):
    """Train model with MLflow tracking"""

    # Load dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log dataset version
        dataset_version = get_dataset_version(X)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dataset_name", "wine")

        # Log input statistics
        stats = calculate_data_stats(X_train, wine.feature_names)
        mlflow.log_param("n_samples_train", stats["n_samples"])
        mlflow.log_param("n_features", stats["n_features"])

        # Log feature statistics as metrics
        for feature, mean_val in stats["mean"].items():
            mlflow.log_metric(f"feature_mean_{feature}", mean_val)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\nRun completed:")
        print(f"  n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"  Dataset: {dataset_version}")


if __name__ == "__main__":
    # Set experiment name
    mlflow.set_experiment("wine-classification")

    print("=" * 60)
    print("MLflow Experiment Tracking Demo")
    print("=" * 60)

    # Run multiple experiments with different hyperparameters
    experiments = [
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 150, "max_depth": 8, "min_samples_split": 3}
    ]

    for i, params in enumerate(experiments, 1):
        print(f"\n[Experiment {i}/{len(experiments)}]")
        train_model(**params)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("View results with: mlflow ui")
    print("Then open: http://localhost:5000")
    print("=" * 60)
