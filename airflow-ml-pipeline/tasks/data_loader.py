"""
Data loading task for the ML pipeline.
Loads the Wine dataset and saves it for downstream tasks.
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import json
import os


def load_and_split_data(**context):
    """
    Load Wine dataset and split into train/test sets.
    Saves data to outputs directory and pushes metadata to XCom.
    """
    print("Loading Wine dataset...")
    wine = load_wine()

    # Create DataFrame
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='target')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create outputs directory
    output_dir = '/opt/airflow/outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    # Prepare metadata
    metadata = {
        'n_samples': len(wine.data),
        'n_features': len(wine.feature_names),
        'n_classes': len(wine.target_names),
        'class_names': wine.target_names.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_names': wine.feature_names
    }

    # Save metadata
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset loaded: {metadata['n_samples']} samples, {metadata['n_features']} features, {metadata['n_classes']} classes")
    print(f"Train size: {metadata['train_size']}, Test size: {metadata['test_size']}")

    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='metadata', value=metadata)

    return metadata
