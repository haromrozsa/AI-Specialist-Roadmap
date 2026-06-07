"""
Model training task for the ML pipeline.
Trains a Random Forest classifier on the Wine dataset.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os
import time


def train_model(**context):
    """
    Train a Random Forest classifier and save the model.
    """
    print("Starting model training...")
    start_time = time.time()

    output_dir = '/opt/airflow/outputs'

    # Load training data
    X_train = pd.read_csv(f'{output_dir}/X_train.csv')
    y_train = pd.read_csv(f'{output_dir}/y_train.csv')['target']

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Save model
    model_path = f'{output_dir}/wine_model.pkl'
    joblib.dump(model, model_path)

    # Training info
    training_info = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'training_time_seconds': round(training_time, 2),
        'n_samples_trained': len(X_train),
        'model_path': model_path
    }

    # Save training info
    with open(f'{output_dir}/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"Model trained in {training_time:.2f} seconds")
    print(f"Model saved to {model_path}")

    # Push to XCom
    context['ti'].xcom_push(key='training_info', value=training_info)

    return training_info
