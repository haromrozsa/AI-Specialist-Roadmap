"""
LLM summarizer task for the ML pipeline.
Generates natural language descriptions from model predictions and metrics.
"""
from transformers import pipeline
import json
import os


def generate_summary(**context):
    """
    Use an LLM to generate a human-readable summary of the ML pipeline results.
    """
    print("Initializing LLM for summary generation...")

    # Load evaluation results from XCom
    ti = context['ti']
    evaluation_results = ti.xcom_pull(key='evaluation_results', task_ids='evaluate_model')
    training_info = ti.xcom_pull(key='training_info', task_ids='train_model')
    metadata = ti.xcom_pull(key='metadata', task_ids='load_data')

    if not evaluation_results:
        raise ValueError("Evaluation results not found in XCom")

    # Initialize text generation pipeline
    print("Loading Hugging Face model (flan-t5-small)...")
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        device=-1  # CPU
    )

    # Prepare prompt with evaluation data
    prompt = f"""Summarize the following machine learning model results in 3-4 sentences for a technical audience:

Model: Random Forest with {training_info['n_estimators']} trees
Dataset: Wine classification with {metadata['n_classes']} classes ({', '.join(metadata['class_names'])})
Training: {metadata['train_size']} samples, Testing: {metadata['test_size']} samples

Performance Metrics:
- Accuracy: {evaluation_results['accuracy']:.2%}
- Precision: {evaluation_results['precision']:.2%}
- Recall: {evaluation_results['recall']:.2%}
- F1-Score: {evaluation_results['f1_score']:.2%}

Results: {evaluation_results['n_correct']} out of {evaluation_results['n_test_samples']} predictions were correct.

Sample Predictions:
{_format_sample_predictions(evaluation_results['sample_predictions'][:3])}

Write a concise summary explaining what the model does, how well it performed, and whether it's reliable."""

    print("Generating summary with LLM...")
    print(f"\nPrompt:\n{'-'*60}\n{prompt}\n{'-'*60}\n")

    # Generate summary
    summary = generator(prompt, max_length=256, do_sample=False)[0]['generated_text']

    print(f"\nLLM-Generated Summary:\n{'-'*60}\n{summary}\n{'-'*60}\n")

    # Also create a structured manual summary as fallback/comparison
    manual_summary = _create_manual_summary(evaluation_results, training_info, metadata)

    # Prepare output
    output = {
        'llm_summary': summary,
        'manual_summary': manual_summary,
        'prompt': prompt,
        'model_used': 'google/flan-t5-small',
        'timestamp': context['ts']
    }

    # Save to file
    output_dir = '/opt/airflow/outputs'
    with open(f'{output_dir}/llm_summary.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Also save as readable text
    with open(f'{output_dir}/llm_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ML PIPELINE SUMMARY (LLM-Generated)\n")
        f.write("="*80 + "\n\n")
        f.write(summary + "\n\n")
        f.write("="*80 + "\n")
        f.write("MANUAL SUMMARY (Structured)\n")
        f.write("="*80 + "\n\n")
        f.write(manual_summary + "\n")

    print(f"Summary saved to {output_dir}/llm_summary.txt")

    return output


def _format_sample_predictions(samples):
    """Format sample predictions for the prompt."""
    lines = []
    for s in samples:
        status = "✓" if s['correct'] else "✗"
        lines.append(f"{status} True: {s['true_class']}, Predicted: {s['predicted_class']} (confidence: {s['confidence']:.1%})")
    return "\n".join(lines)


def _create_manual_summary(evaluation_results, training_info, metadata):
    """Create a structured manual summary as comparison."""
    accuracy_pct = evaluation_results['accuracy'] * 100
    n_correct = evaluation_results['n_correct']
    n_total = evaluation_results['n_test_samples']

    # Determine performance level
    if accuracy_pct >= 95:
        performance = "excellent"
    elif accuracy_pct >= 85:
        performance = "strong"
    elif accuracy_pct >= 75:
        performance = "good"
    else:
        performance = "moderate"

    summary = f"""The Random Forest classifier achieved {performance} performance on the Wine classification task, correctly identifying {n_correct} out of {n_total} wine samples ({accuracy_pct:.1f}% accuracy). The model distinguishes between {metadata['n_classes']} wine classes ({', '.join(metadata['class_names'])}) based on {metadata['n_features']} chemical features.

Key metrics: Precision={evaluation_results['precision']:.1%}, Recall={evaluation_results['recall']:.1%}, F1-Score={evaluation_results['f1_score']:.1%}.

The model was trained on {metadata['train_size']} samples using {training_info['n_estimators']} decision trees, completing training in {training_info['training_time_seconds']} seconds. Based on the high accuracy and balanced precision/recall scores, this model is reliable for wine classification tasks."""

    return summary
