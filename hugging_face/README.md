# Hugging Face – Getting Started

# Session Summary: Basic Inference with Hugging Face Transformers

## What I Did During the Session

1. **Set Up Hugging Face Transformers Library**
   - Installed the `transformers` library in the conda environment
   - Configured TensorFlow as the backend framework

2. **Implemented Basic Inference Pipeline**
   - Created `basic_inference.py` for sentiment analysis
   - Loaded a pre-trained tokenizer using `AutoTokenizer.from_pretrained()`
   - Loaded a pre-trained model using `TFAutoModelForSequenceClassification.from_pretrained()`
   - Used the DistilBERT model fine-tuned on SST-2 dataset for sentiment classification

3. **Processed Text Input**
   - Tokenized input text with appropriate parameters (padding, truncation)
   - Converted text to TensorFlow tensors for model consumption

4. **Ran Model Inference**
   - Executed the model on tokenized inputs
   - Applied softmax to convert logits to probabilities
   - Extracted predicted class and confidence scores

5. **Analyzed and Displayed Results**
   - Retrieved label mappings from model configuration
   - Printed prediction results with confidence scores for each class

## What I Learned

1. **Hugging Face Ecosystem**:
   - The `transformers` library provides easy access to thousands of pre-trained models
   - Models can be loaded with just a few lines of code using the AutoClass API
   - Both PyTorch and TensorFlow backends are supported

2. **Tokenizers**:
   - Tokenizers convert raw text into numerical representations
   - The `AutoTokenizer` automatically selects the correct tokenizer for a given model
   - Important parameters include `return_tensors`, `padding`, and `truncation`

3. **Pre-trained Models**:
   - Models are identified by their Hugging Face Hub name (e.g., `distilbert-base-uncased-finetuned-sst-2-english`)
   - TensorFlow models use the `TF` prefix (e.g., `TFAutoModelForSequenceClassification`)
   - Model configurations contain useful metadata like label mappings (`id2label`)

4. **Inference Pipeline**:
   - Basic inference follows a consistent pattern: load tokenizer → load model → tokenize input → run inference → process output
   - Output logits need to be converted to probabilities using softmax
   - `tf.argmax()` is used to get the predicted class index

This session provided an introduction to using Hugging Face Transformers for NLP tasks, specifically implementing a sentiment analysis pipeline using a pre-trained DistilBERT model with TensorFlow.
