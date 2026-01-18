"""
Basic TensorFlow Pipeline for Sentiment Analysis using Hugging Face Transformers
"""

from transformers import pipeline


def main():
    # Initialize a sentiment analysis pipeline with TensorFlow
    print("Initializing pipeline...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Sample texts for inference
    texts = [
        "I love using Hugging Face transformers! It's amazing.",
        "This movie was terrible and boring.",
        "The weather is okay today."
    ]

    # Run inference
    print("Running inference...")
    results = classifier(texts)

    # Print results
    print("\n--- Results ---")
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Label: {result['label']}, Score: {result['score']:.4f}")
        print()


if __name__ == "__main__":
    main()