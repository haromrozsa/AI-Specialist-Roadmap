"""
ML Pipeline DAG with LLM Summary Generation

This DAG orchestrates a complete machine learning workflow:
1. Load Wine dataset and split into train/test
2. Train Random Forest classifier
3. Evaluate model performance
4. Generate LLM-based summary of results

This demonstrates workflow automation for ML pipelines with AI-generated insights.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add tasks directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tasks'))

# Import task functions
from data_loader import load_and_split_data
from model_trainer import train_model
from model_evaluator import evaluate_model
from llm_summarizer import generate_summary


# Default arguments for the DAG
default_args = {
    'owner': 'ai-specialist',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'ml_pipeline_with_llm',
    default_args=default_args,
    description='ML pipeline: data → train → evaluate → LLM summary',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['machine-learning', 'llm', 'wine-classification'],
)

# Task 1: Load and split data
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_and_split_data,
    dag=dag,
    doc_md="""
    ### Load Wine Dataset

    Loads the scikit-learn Wine dataset and splits it into training and test sets.

    **Outputs:**
    - X_train.csv, X_test.csv, y_train.csv, y_test.csv
    - metadata.json with dataset information
    """,
)

# Task 2: Train model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    doc_md="""
    ### Train Random Forest Classifier

    Trains a Random Forest model on the Wine dataset.

    **Model Configuration:**
    - n_estimators: 100
    - max_depth: 10
    - min_samples_split: 5

    **Outputs:**
    - wine_model.pkl (trained model)
    - training_info.json (training metadata)
    """,
)

# Task 3: Evaluate model
evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
    doc_md="""
    ### Evaluate Model Performance

    Evaluates the trained model on the test set.

    **Metrics Calculated:**
    - Accuracy, Precision, Recall, F1-Score
    - Per-class performance metrics
    - Sample predictions with confidence scores

    **Outputs:**
    - evaluation_results.json
    """,
)

# Task 4: Generate LLM summary
generate_summary_task = PythonOperator(
    task_id='generate_llm_summary',
    python_callable=generate_summary,
    dag=dag,
    execution_timeout=timedelta(minutes=10),  # LLM can take time
    doc_md="""
    ### Generate LLM Summary

    Uses Hugging Face Transformers (flan-t5-small) to generate a natural language
    summary of the ML pipeline results.

    **LLM Input:**
    - Model configuration and metrics
    - Performance results
    - Sample predictions

    **Outputs:**
    - llm_summary.json (structured output)
    - llm_summary.txt (human-readable text)
    """,
)

# Define task dependencies
# Data loading must complete before training
load_data_task >> train_model_task

# Training must complete before evaluation
train_model_task >> evaluate_model_task

# Evaluation must complete before LLM summary
evaluate_model_task >> generate_summary_task

# Visual representation:
# load_data → train_model → evaluate_model → generate_llm_summary
