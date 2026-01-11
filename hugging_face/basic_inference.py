from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


def main():
    # Define the model to use (sentiment analysis model as example)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # 1. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load model (TF version)
    print("Loading model...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

    # 3. Prepare input text
    text = "I love using Hugging Face transformers! It's amazing."

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)

    # 4. Run inference
    print("Running inference...")
    outputs = model(**inputs)

    # 5. Print output
    # Get the predicted class
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]

    # Get label names
    labels = model.config.id2label

    print("\n--- Results ---")
    print(f"Input text: {text}")
    print(f"Predicted class: {labels[predicted_class]}")
    print(f"Confidence scores:")
    print(f"  NEGATIVE: {predictions[0][0]:.4f}")
    print(f"  POSITIVE: {predictions[0][1]:.4f}")


if __name__ == "__main__":
    main()