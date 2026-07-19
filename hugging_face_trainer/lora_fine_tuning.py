"""
LoRA (Low-Rank Adaptation) Fine-tuning with PEFT:
A fourth approach to compare against fine-tuning / frozen base / from scratch.

Core idea: instead of learning the full weight update ΔW (768x768 = 589,824
numbers per attention projection), learn a low-rank factorization ΔW ≈ B @ A
where B is (768 x r) and A is (r x 768) with r=8. That is 12,288 numbers
instead of 589,824 -- about 2% -- while the original weight W stays frozen.

    h = W·x  +  (alpha/r) · B·A·x
        ↑                   ↑
      frozen            trainable

Requires: pip install peft
"""

import os
import copy

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "distilbert-base-uncased"
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200

# LoRA hyperparameters
LORA_R = 8              # Rank of the update matrices. Higher = more capacity.
LORA_ALPHA = 16         # Scaling factor. Update is multiplied by alpha/r.
LORA_DROPOUT = 0.05     # Dropout applied to the LoRA path only

# Which weight matrices get adapters. The LoRA paper found the attention
# query and value projections give the best accuracy-per-parameter.
# In DistilBERT these are named q_lin and v_lin.
TARGET_MODULES = ["q_lin", "v_lin"]

# AutoModelForSequenceClassification adds a RANDOMLY INITIALIZED head on top of
# DistilBERT: pre_classifier (Linear 768->768) then classifier (Linear 768->2).
# These must be trained and saved too -- otherwise pre_classifier would stay
# frozen at its random initialization and cripple accuracy.
MODULES_TO_SAVE = ["pre_classifier", "classifier"]


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


def count_params(model):
    """Return (trainable, total) parameter counts"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def directory_size_bytes(path):
    """Total size on disk of every file under path"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


# ============================================================================
# APPROACH 4: LoRA (Low-Rank Adaptation)
# ============================================================================

def approach_4_lora():
    """
    LoRA: Freeze the whole base model, inject small trainable low-rank
    matrices into the attention projections.
    - Best for: Getting close to full fine-tuning accuracy on limited hardware
    - Data needed: 1,000 - 100,000 samples
    - Time: Similar to fine-tuning per step, but far less optimizer memory
    """
    print("\n" + "=" * 70)
    print("APPROACH 4: LoRA (low-rank adapters on attention projections)")
    print("=" * 70)

    dataset = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    base_trainable, base_total = count_params(model)
    print(f"Base model parameters: {base_total:,} (all trainable before LoRA)")

    # ⭐ Wrap the model with LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,
    )
    model = get_peft_model(model, lora_config)

    # PEFT's own summary: trainable vs total, and the percentage
    model.print_trainable_parameters()

    # Break the number down so it is clear where the trainable params live.
    # B is initialized to ZEROS, so B@A = 0 at step 0 and the model starts
    # numerically identical to the pre-trained one -- no perturbation.
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    head_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n
    )
    print(f"  ├─ LoRA adapter parameters : {lora_params:,}")
    print(f"  └─ Classification head      : {head_params:,}")
    print(f"     (head dominates here only because DistilBERT is small)")

    tokenized_data = tokenize_data(dataset, tokenizer)

    # LoRA tolerates a much higher learning rate than full fine-tuning (2e-5),
    # because the frozen pre-trained weights cannot be catastrophically
    # overwritten -- only the small adapter moves.
    training_args = TrainingArguments(
        output_dir="./output_lora",
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        num_train_epochs=3,
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

    print(f"\n✅ LoRA Accuracy: {results['eval_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # What actually has to be shipped
    # ------------------------------------------------------------------
    adapter_dir = "./output_lora/adapter"
    model.save_pretrained(adapter_dir)
    adapter_bytes = directory_size_bytes(adapter_dir)

    print("\n" + "-" * 70)
    print("ARTIFACT SIZE")
    print("-" * 70)
    print(f"Saved adapter: {adapter_dir}")
    print(f"Adapter size on disk : {adapter_bytes / 1024:.1f} KB")
    print(f"Full model would be  : ~{base_total * 4 / 1024 ** 2:.0f} MB (fp32)")
    print("PEFT save_pretrained() writes ONLY the adapter -- the frozen base")
    print("model is referenced by name and re-downloaded from the Hub.")

    # ------------------------------------------------------------------
    # Merging: LoRA is a training-time trick, not a permanent architecture
    # ------------------------------------------------------------------
    demo_merge(model, tokenizer)

    return results


def demo_merge(peft_model, tokenizer):
    """
    Show that W + (alpha/r)·B·A can be folded back into W, producing an
    ordinary model with ZERO extra inference latency.
    """
    print("\n" + "-" * 70)
    print("MERGING ADAPTER INTO BASE WEIGHTS")
    print("-" * 70)

    samples = [
        "An absolute masterpiece, I was moved to tears.",
        "Dull, predictable and far too long. A waste of time.",
    ]
    batch = tokenizer(samples, return_tensors="pt", padding=True, truncation=True)

    peft_model.eval()
    with torch.no_grad():
        before = peft_model(**batch).logits

    # merge_and_unload() is destructive, so work on a copy
    merged = copy.deepcopy(peft_model).merge_and_unload()
    merged.eval()
    with torch.no_grad():
        after = merged(**batch).logits

    max_diff = (before - after).abs().max().item()

    print(f"Logits before merge : {before.tolist()}")
    print(f"Logits after merge  : {after.tolist()}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Type after merge: {type(merged).__name__} (plain transformers model)")

    if max_diff < 1e-4:
        print("✅ Predictions identical -- the adapter is now baked into W.")
    else:
        print("⚠️  Unexpected drift after merging; check the LoRA configuration.")


# ============================================================================
# BASELINE: FULL FINE-TUNING (for an apples-to-apples comparison)
# ============================================================================

def baseline_full_fine_tuning():
    """Train all ~67M parameters, same data and evaluation as the LoRA run"""
    print("\n" + "=" * 70)
    print("BASELINE: Full fine-tuning (train all layers)")
    print("=" * 70)

    dataset = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    trainable, total = count_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    tokenized_data = tokenize_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./output_full_ft",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
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

    print(f"\n✅ Full Fine-tuning Accuracy: {results['eval_accuracy']:.4f}")
    return results, trainable


# ============================================================================
# COMPARE LoRA vs FULL FINE-TUNING
# ============================================================================

def compare_lora_vs_full():
    """Run both approaches and print the accuracy / parameter tradeoff"""
    print("\n" + "=" * 70)
    print("COMPARING LoRA AGAINST FULL FINE-TUNING")
    print("=" * 70)

    full_results, full_params = baseline_full_fine_tuning()
    lora_results = approach_4_lora()

    full_acc = full_results["eval_accuracy"]
    lora_acc = lora_results["eval_accuracy"]

    print("\n" + "=" * 70)
    print("SUMMARY: LoRA vs Full Fine-tuning")
    print("=" * 70)
    print(f"""
    ┌────────────────────────┬──────────────┬─────────────────────────┐
    │ Approach               │ Accuracy     │ Notes                   │
    ├────────────────────────┼──────────────┼─────────────────────────┤
    │ Full fine-tuning       │ {full_acc:.4f}       │ All weights updated     │
    │ LoRA (r={LORA_R})             │ {lora_acc:.4f}       │ Base frozen, ~MB adapter│
    └────────────────────────┴──────────────┴─────────────────────────┘

    Accuracy delta (LoRA - full): {lora_acc - full_acc:+.4f}
    """)

    return {"full_fine_tuning": full_results, "lora": lora_results}


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run both and compare
    compare_lora_vs_full()

    # Or run LoRA on its own:
    # approach_4_lora()
