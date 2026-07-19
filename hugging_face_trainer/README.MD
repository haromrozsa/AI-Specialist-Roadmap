# Hugging Face Training

# Session Summary: Fine-tuning and Training Approaches with Hugging Face Transformers

## What I Did During the Session

1. **Built a Complete Fine-tuning Pipeline**:
    - Created for sentiment analysis on IMDB movie reviews `fine_tuning_sentiment.py`
    - Used the Hugging Face `Trainer` API for streamlined training workflow
    - Implemented custom metrics calculation (accuracy, precision, recall, F1 score)
    - Used `DataCollatorWithPadding` for efficient dynamic padding during training
    - Tested the fine-tuned model with custom examples using direct inference

2. **Compared Three Training Approaches**:
    - Created to benchmark different strategies `fine_tuning_vs_frozen_base_vs_from_scratch.py`
    - **Fine-tuning**: Training all layers with pre-trained weights (~66M trainable parameters)
    - **Frozen Base**: Freezing DistilBERT layers, training only the classifier head (~592K trainable parameters: `pre_classifier` 590,592 + `classifier` 1,538)
    - **From Scratch**: Initializing random weights using `AutoConfig` and `from_config()` method

3. **Explored Model Architecture and Weight Management**:
    - Investigated how `AutoModelForSequenceClassification` adds classification layers on top of base models
    - Learned to freeze layers using `param.requires_grad = False`
    - Understood the difference between `from_pretrained()` (loads weights) vs `from_config()` (random initialization)
    - Explored model structure: base model (`model.distilbert`) vs classifier head (`model.classifier`, `model.pre_classifier`)

4. **Implemented Custom Inference Without Saving/Loading**:
    - Used `torch.no_grad()` for efficient inference
    - Applied `torch.nn.functional.softmax()` to convert logits to probabilities
    - Extracted predictions using `torch.argmax()` and mapped to labels via `model.config.id2label`

5. **Added LoRA as a Fourth Training Approach**:
    - Created `lora_fine_tuning.py` using the `peft` library (`pip install peft`)
    - Wrapped the model with `get_peft_model()` and a `LoraConfig` (`r=8`, `lora_alpha=16`, `task_type=SEQ_CLS`)
    - Targeted the attention query/value projections (`q_lin`, `v_lin`) — the projections the LoRA paper found most effective
    - Used `modules_to_save=["pre_classifier", "classifier"]` so the randomly initialized head is trained and saved alongside the adapter
    - Measured the saved adapter size on disk versus the full model
    - Demonstrated `merge_and_unload()` folding the adapter back into the base weights and verified predictions are unchanged
    - Ran full fine-tuning as a baseline in the same script for an apples-to-apples accuracy comparison

## What I Learned
1. **Fine-tuning vs Transfer Learning Approaches**:
    - **Fine-tuning** (train all layers): Best accuracy, requires 1K-100K samples, hours of training
    - **Frozen base** (train head only): Faster, works with limited data (100-1K samples), slightly lower accuracy
    - **From scratch** (random weights): Requires millions of samples and significant compute, lowest accuracy with small datasets
    - The choice depends on available data, compute resources, and accuracy requirements

2. **Model Architecture Understanding**:
    - `AutoModelForSequenceClassification` automatically adds a classification head on top of the base model
    - The head consists of: `pre_classifier` (Linear 768→768) → ReLU → Dropout → `classifier` (Linear 768→num_labels)
    - Base model parameters can be frozen independently while keeping the head trainable
    - `model.parameters()` iterates over ALL parameters; `model.distilbert.parameters()` only the base

3. **Pre-trained Weights vs Random Initialization**:
    - `from_pretrained()`: Downloads config AND pre-trained weights (model "knows" language)
    - `from_config()`: Uses config for architecture but initializes weights randomly (model "knows" nothing)
    - `AutoConfig.from_pretrained()` only downloads the architecture definition, not the weights
    - Pre-trained weights encode knowledge learned from millions of texts (Wikipedia, books, web)

4. **Hugging Face Trainer API Best Practices**:
    - `TrainingArguments` controls all training hyperparameters (learning rate, batch size, epochs, etc.)
    - `eval_strategy="epoch"` evaluates after each epoch for progress monitoring
    - `save_strategy="no"` disables checkpointing for faster experimentation
    - `DataCollatorWithPadding` is more efficient than pre-padding all sequences to max length
    - Custom `compute_metrics` function enables tracking of any evaluation metrics

5. **Hyperparameter Differences by Training Approach**:
    - Fine-tuning: Small learning rate (2e-5) to preserve pre-trained knowledge
    - Frozen base: Higher learning rate (1e-3) acceptable since only training small head
    - From scratch: Higher learning rate (5e-4), more epochs, warmup steps needed
    - Weight decay (0.01) helps regularization in all approaches

6. **LoRA and Parameter-Efficient Fine-tuning (PEFT)**:
    - The insight is that the *update* `ΔW`, not the weight matrix itself, is empirically low-rank — so it can be factorized as `ΔW ≈ B @ A` with `B` of shape (768 × r) and `A` of shape (r × 768)
    - The forward pass becomes `h = W·x + (alpha/r)·B·A·x`, where `W` stays frozen and only `A` and `B` receive gradients
    - For one 768×768 projection at `r=8` this is 12,288 trainable numbers instead of 768×768 = 589,824 — roughly 2% (LoRA adapts the weight matrix only, not the bias)
    - `B` is initialized to **zeros** and `A` to small random values, so `B@A = 0` at step 0: training starts from the pre-trained model exactly, which is why LoRA is stable without warmup tricks
    - `r` controls capacity (4–8 is ample for classification, 16–64 for generative tasks); `lora_alpha` scales the adapter's influence, with `alpha = 2r` a common convention
    - Adapting only `q_lin` and `v_lin` across 6 layers gives 12 modules × 12,288 ≈ **147K LoRA parameters** (~0.22% of the ~67M total)

7. **Where the Trainable Parameters Actually Live**:
    - Total trainable in the LoRA run is ~740K: ~147K of adapters plus ~592K of classification head
    - The head *dominates* because DistilBERT is small and its head is proportionally large — on a 7B-parameter LLM the ratio inverts completely and the adapters are the only meaningful cost
    - `task_type=SEQ_CLS` alone is not enough: without `modules_to_save`, `pre_classifier` stays frozen at its **random** initialization and accuracy collapses

8. **Artifact Size Is Not the Reason to Choose LoRA**:
    - Frozen base only changes the head, so in principle it ships ~592K parameters (~2.3 MB) — *smaller* than a LoRA adapter
    - The real difference is **capacity**: frozen base learns only a new linear readout over features that never move, while LoRA injects trainable deltas into attention across all 6 layers, so internal representations genuinely shift toward the task
    - Tooling obscures this: `Trainer.save_model()` serializes all ~255 MB regardless of what was frozen, whereas PEFT's `save_pretrained()` writes only the adapter by default

9. **LoRA Is a Training-Time Trick, Not a Permanent Architecture**:
    - `merge_and_unload()` folds `W + (alpha/r)·B·A` back into `W`, returning an ordinary transformers model with **zero** added inference latency
    - Keeping adapters unmerged instead allows hot-swapping many task-specific adapters against one shared frozen base — the standard way to serve many fine-tunes without many model copies
    - LoRA tolerates a much higher learning rate (2e-4) than full fine-tuning (2e-5), because frozen pre-trained weights cannot be catastrophically overwritten
    - Combined with 4-bit quantization this becomes **QLoRA**, the technique that makes fine-tuning 7B–70B models feasible on a single GPU

## Approach Comparison

All rows describe `distilbert-base-uncased` (~67M parameters) except where noted.

| Approach | Trainable params | Base weights changed? | Adapts internal representations? |
|---|---|---|---|
| From scratch | ~67M in principle; **~9.1M as implemented** (reduced config: 2 layers, dim 256) | n/a (all random) | Yes |
| Full fine-tuning | ~67M | Yes | Yes |
| Frozen base | ~592K | No | **No** |
| LoRA (`r=8`) | ~740K (147K adapter + 592K head) | No | Yes |

⚠️ **The from-scratch row is not a like-for-like comparison.** Training a full DistilBERT from random initialization would use the same ~67M parameters as full fine-tuning — architecture is independent of how the weights are initialized. `approach_3_from_scratch()` deliberately shrinks the model (`n_layers=2`, `dim=256`, `hidden_dim=512`) so the demonstration finishes in reasonable time on CPU. Its weak accuracy therefore has **two** causes — random initialization *and* reduced capacity — which should not be conflated. The other three rows all train the same ~67M model, so they *are* directly comparable to each other.

## Running It

```bash
cd hugging_face_trainer
pip install peft
python lora_fine_tuning.py
```

Runs full fine-tuning as a baseline, then the LoRA run, and prints an accuracy comparison. CPU-friendly with DistilBERT on a 1,000-sample IMDB subset.

This session provided deep understanding of how transfer learning works in practice, the tradeoffs between different training strategies, and hands-on experience with the Hugging Face Trainer API for model fine-tuning.