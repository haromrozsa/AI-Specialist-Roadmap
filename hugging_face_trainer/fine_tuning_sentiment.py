"""
Fine-tuning DistilBERT for Sentiment Analysis using Hugging Face Transformers
Dataset: IMDB Movie Reviews (subset for demonstration)
Simplified: Single method, no model saving/loading
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fine_tune_and_evaluate():
    """
    Complete fine-tuning pipeline in one method:
    1. Load dataset
    2. Load tokenizer and model
    3. Train
    4. Evaluate
    5. Test with custom examples
    """

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    TRAIN_SAMPLES = 2000
    TEST_SAMPLES = 500

    print("=" * 60)
    print("Fine-tuning DistilBERT for Sentiment Analysis")
    print("=" * 60)

    # =========================================================================
    # 1. LOAD DATASET
    # =========================================================================

    print("\n[1/5] Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(TEST_SAMPLES))
    print(f"      Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

    # =========================================================================
    # 2. LOAD TOKENIZER & MODEL
    # =========================================================================

    print(f"\n[2/5] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )

    # =========================================================================
    # 3. TOKENIZE DATASET
    # =========================================================================

    print("\n[3/5] Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # =========================================================================
    # 4. TRAIN
    # =========================================================================

    print("\n[4/5] Training...")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    training_args = TrainingArguments(
        output_dir="./tmp_training",
        eval_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # =========================================================================
    # 5. EVALUATE & TEST
    # =========================================================================

    print("\n[5/5] Final Evaluation")
    print("=" * 60)

    results = trainer.evaluate()

    print(f"\n  Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"  Precision: {results['eval_precision']:.4f}")
    print(f"  Recall:    {results['eval_recall']:.4f}")
    print(f"  F1 Score:  {results['eval_f1']:.4f}")

    # =========================================================================
    # TEST WITH CUSTOM EXAMPLES (using the trained model directly)
    # =========================================================================

    print("\n" + "=" * 60)
    print("Testing with Custom Examples")
    print("=" * 60)

    test_texts = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Terrible waste of time. The plot made no sense at all.",
        "It was okay, nothing special but not bad either.",
        "The acting was superb and the cinematography was breathtaking!",
        "I fell asleep halfway through. So boring and predictable."
    ]

    # Set model to evaluation mode
    model.eval()
    id2label = model.config.id2label

    print("\nPredictions:")
    print("-" * 60)

    for text in test_texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Inference (no gradient computation needed)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        label = id2label[predicted_class]
        emoji = "😊" if label == "POSITIVE" else "😞"

        print(f"\n{emoji} {label} ({confidence:.4f})")
        print(f"   \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

    print("\n" + "=" * 60)
    print("✅ Fine-tuning complete!")
    print("=" * 60)

    return model, tokenizer, results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    model, tokenizer, results = fine_tune_and_evaluate()