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
    - **Frozen Base**: Freezing DistilBERT layers, training only the classifier head (~1.5K trainable parameters)
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

This session provided deep understanding of how transfer learning works in practice, the tradeoffs between different training strategies, and hands-on experience with the Hugging Face Trainer API for model fine-tuning.