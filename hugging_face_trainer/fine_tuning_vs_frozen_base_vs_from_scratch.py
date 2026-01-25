"""
Three Approaches to Training with Hugging Face Transformers:
1. Fine-tuning (most common)
2. Training classification head only (frozen base)
3. Training from scratch (demonstration with small model)
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "distilbert-base-uncased"
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200


# ============================================================================
# SHARED FUNCTIONS
# ============================================================================

def load_data():
    """Load and prepare IMDB dataset subset"""
    print("Loading dataset...")
    dataset = load_dataset("imdb")
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(TEST_SAMPLES))
    return dataset


def tokenize_data(dataset, tokenizer):
    """Tokenize dataset"""
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
    
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    """Calculate accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def train_and_evaluate(model, tokenized_data, tokenizer, output_dir, epochs=2):
    """Train and evaluate model"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    return results


# ============================================================================
# APPROACH 1: FINE-TUNING (Most Common)
# ============================================================================

def approach_1_fine_tuning():
    """
    Fine-tuning: Load pre-trained weights, train ALL layers
    - Best for: Most use cases
    - Data needed: 1,000 - 100,000 samples
    - Time: Hours
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Fine-tuning (train all layers)")
    print("=" * 70)
    
    # Load data
    dataset = load_data()
    
    # Load tokenizer and model with PRE-TRAINED weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Tokenize
    tokenized_data = tokenize_data(dataset, tokenizer)
    
    # Train
    results = train_and_evaluate(
        model, tokenized_data, tokenizer, 
        output_dir="./output_fine_tuning"
    )
    
    print(f"\n✅ Fine-tuning Accuracy: {results['eval_accuracy']:.4f}")
    return results


# ============================================================================
# APPROACH 2: TRAIN CLASSIFICATION HEAD ONLY (Frozen Base)
# ============================================================================

def approach_2_frozen_base():
    """
    Frozen base: Load pre-trained weights, train ONLY classifier head
    - Best for: Limited data, quick experiments
    - Data needed: 100 - 1,000 samples
    - Time: Minutes
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Frozen Base (train classifier head only)")
    print("=" * 70)
    
    # Load data
    dataset = load_data()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    
    # ⭐ FREEZE the base model (DistilBERT layers)
    for param in model.distilbert.parameters():
        param.requires_grad = False
    
    # Only classifier head is trainable now
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Tokenize
    tokenized_data = tokenize_data(dataset, tokenizer)
    
    # Train (can use higher learning rate since only training small head)
    training_args = TrainingArguments(
        output_dir="./output_frozen",
        eval_strategy="epoch",
        learning_rate=1e-3,  # Higher LR for head only
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    print(f"\n✅ Frozen Base Accuracy: {results['eval_accuracy']:.4f}")
    return results


# ============================================================================
# APPROACH 3: TRAINING FROM SCRATCH
# ============================================================================

def approach_3_from_scratch():
    """
    From scratch: Random weights, train ALL layers
    - Best for: Custom domains, new languages, research
    - Data needed: Millions of samples
    - Time: Days/Weeks with many GPUs
    
    ⚠️ Note: This is a DEMONSTRATION with small data.
    Real from-scratch training needs MUCH more data and compute.
    """
    print("\n" + "=" * 70)
    print("APPROACH 3: Training from Scratch (random initialization)")
    print("=" * 70)
    print("⚠️  WARNING: This is a demonstration only!")
    print("    Real from-scratch training needs millions of samples.")
    
    # Load data
    dataset = load_data()
    
    # Load tokenizer (we still use pre-trained tokenizer for vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ⭐ Create model with RANDOM weights (not pre-trained)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        # Smaller model for demonstration
        n_layers=2,           # Reduced from 6
        n_heads=4,            # Reduced from 12
        dim=256,              # Reduced from 768
        hidden_dim=512,       # Reduced from 3072
    )
    
    # Initialize model with random weights
    model = AutoModelForSequenceClassification.from_config(config)
    
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (all random): {trainable_params:,}")
    
    # Tokenize
    tokenized_data = tokenize_data(dataset, tokenizer)
    
    # Train (needs more epochs and different hyperparameters)
    training_args = TrainingArguments(
        output_dir="./output_from_scratch",
        eval_strategy="epoch",
        learning_rate=5e-4,  # Higher LR for from-scratch
        per_device_train_batch_size=16,
        num_train_epochs=5,  # More epochs needed
        warmup_steps=100,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    print(f"\n✅ From Scratch Accuracy: {results['eval_accuracy']:.4f}")
    print("   (Expected to be lower due to limited data)")
    return results


# ============================================================================
# COMPARE ALL APPROACHES
# ============================================================================

def compare_all_approaches():
    """Run all three approaches and compare results"""
    print("\n" + "=" * 70)
    print("COMPARING ALL THREE TRAINING APPROACHES")
    print("=" * 70)
    
    results = {}
    
    # Run all approaches
    results["fine_tuning"] = approach_1_fine_tuning()
    results["frozen_base"] = approach_2_frozen_base()
    results["from_scratch"] = approach_3_from_scratch()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Accuracy Comparison")
    print("=" * 70)
    print(f"""
    ┌────────────────────────┬──────────────┬─────────────────────────┐
    │ Approach               │ Accuracy     │ Notes                   │
    ├────────────────────────┼──────────────┼─────────────────────────┤
    │ 1. Fine-tuning         │ {results['fine_tuning']['eval_accuracy']:.4f}       │ Best for most cases     │
    │ 2. Frozen Base         │ {results['frozen_base']['eval_accuracy']:.4f}       │ Fast, less data needed  │
    │ 3. From Scratch        │ {results['from_scratch']['eval_accuracy']:.4f}       │ Needs much more data    │
    └────────────────────────┴──────────────┴─────────────────────────┘
    """)
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run comparison of all approaches
    compare_all_approaches()
    
    # Or run individually:
    # approach_1_fine_tuning()
    # approach_2_frozen_base()
    # approach_3_from_scratch()
