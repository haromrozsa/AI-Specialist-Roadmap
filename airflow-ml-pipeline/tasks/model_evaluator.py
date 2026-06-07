"""
Model evaluation task for the ML pipeline.
Evaluates the trained model and generates predictions.
"""
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


def evaluate_model(**context):
    """
    Evaluate the trained model on test data.
    """
    print("Starting model evaluation...")

    output_dir = '/opt/airflow/outputs'

    # Load test data
    X_test = pd.read_csv(f'{output_dir}/X_test.csv')
    y_test = pd.read_csv(f'{output_dir}/y_test.csv')['target']

    # Load model
    model = joblib.load(f'{output_dir}/wine_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Get metadata for class names
    with open(f'{output_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    class_names = metadata['class_names']

    # Detailed classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Sample predictions (first 10)
    sample_predictions = []
    for i in range(min(10, len(y_test))):
        sample_predictions.append({
            'sample_index': i,
            'true_class': class_names[y_test.iloc[i]],
            'predicted_class': class_names[y_pred[i]],
            'confidence': round(float(np.max(y_proba[i])), 3),
            'correct': bool(y_test.iloc[i] == y_pred[i])
        })

    # Evaluation results
    evaluation_results = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'n_test_samples': len(y_test),
        'n_correct': int((y_test == y_pred).sum()),
        'n_incorrect': int((y_test != y_pred).sum()),
        'class_names': class_names,
        'classification_report': report,
        'sample_predictions': sample_predictions
    }

    # Save evaluation results
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Evaluation complete:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Correct predictions: {evaluation_results['n_correct']}/{evaluation_results['n_test_samples']}")

    # Push to XCom for LLM task
    context['ti'].xcom_push(key='evaluation_results', value=evaluation_results)

    return evaluation_results
