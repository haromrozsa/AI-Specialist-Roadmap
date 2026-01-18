from transformers import pipeline
import time

def measure_inference(pipe, inputs, task_name, model_name):
    """Measure inference time and return results with timing."""
    start_time = time.time()
    results = pipe(inputs)
    elapsed_time = time.time() - start_time

    return {
        "task": task_name,
        "model": model_name,
        "results": results,
        "time_seconds": elapsed_time,
        "inputs_count": len(inputs) if isinstance(inputs, list) else 1
    }


def compare_sentiment_models():
    """Compare different sentiment analysis models."""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS COMPARISON")
    print("=" * 60)

    # Test texts
    test_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "This was a terrible experience. I'm very disappointed.",
        "The service was okay, nothing special but not bad either.",
        "Incredible quality and fast shipping. Highly recommend!",
        "I regret buying this. Complete waste of money."
    ]

    # Models to compare
    sentiment_models = [
        ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT (small, fast)"),
        ("nlptown/bert-base-multilingual-uncased-sentiment", "BERT Multilingual (5-star rating)")
    ]

    results_summary = []

    for model_name, description in sentiment_models:
        print(f"\n--- Model: {description} ---")
        print(f"    ({model_name})")

        # Load pipeline
        print("    Loading model...")
        load_start = time.time()
        classifier = pipeline("sentiment-analysis", model=model_name)
        load_time = time.time() - load_start
        print(f"    Model loaded in {load_time:.2f}s")

        # Run inference
        metrics = measure_inference(classifier, test_texts, "sentiment", model_name)

        print(f"\n    Inference time: {metrics['time_seconds']:.3f}s for {metrics['inputs_count']} texts")
        print(f"    Avg time per text: {metrics['time_seconds'] / metrics['inputs_count'] * 1000:.1f}ms")

        print("\n    Results:")
        for text, result in zip(test_texts, metrics['results']):
            print(f"      \"{text[:50]}...\"")
            print(f"        → {result['label']} (confidence: {result['score']:.4f})")

        results_summary.append({
            "model": description,
            "load_time": load_time,
            "inference_time": metrics['time_seconds'],
            "avg_per_text": metrics['time_seconds'] / metrics['inputs_count']
        })

        # Clean up to free memory
        del classifier

    return results_summary


def compare_text_generation_models():
    """Compare different text generation models."""
    print("\n" + "=" * 60)
    print("TEXT GENERATION COMPARISON")
    print("=" * 60)

    # Prompts to test
    prompts = [
        "The future of artificial intelligence is",
        "In a world where robots"
    ]

    # Models to compare (smaller models for faster comparison)
    generation_models = [
        ("distilgpt2", "DistilGPT-2 (small, 82M params)"),
        ("gpt2", "GPT-2 (medium, 124M params)")
    ]

    results_summary = []

    for model_name, description in generation_models:
        print(f"\n--- Model: {description} ---")
        print(f"    ({model_name})")

        # Load pipeline
        print("    Loading model...")
        load_start = time.time()
        generator = pipeline(
            "text-generation",
            model=model_name,
            pad_token_id=50256
        )
        load_time = time.time() - load_start
        print(f"    Model loaded in {load_time:.2f}s")

        # Run inference
        print("\n    Generating text...")
        start_time = time.time()

        all_results = []
        for prompt in prompts:
            result = generator(
                prompt,
                max_length=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                truncation=True
            )
            all_results.append(result)

        inference_time = time.time() - start_time

        print(f"    Inference time: {inference_time:.3f}s for {len(prompts)} prompts")
        print(f"    Avg time per prompt: {inference_time / len(prompts):.3f}s")

        print("\n    Generated Texts:")
        for prompt, result in zip(prompts, all_results):
            print(f"\n      Prompt: \"{prompt}\"")
            generated = result[0]['generated_text']
            print(f"      Output: \"{generated}\"")

        results_summary.append({
            "model": description,
            "load_time": load_time,
            "inference_time": inference_time,
            "avg_per_prompt": inference_time / len(prompts)
        })

        # Clean up
        del generator

    return results_summary


def print_tradeoff_analysis(sentiment_results, generation_results):
    """Print a summary of tradeoffs between models."""
    print("\n" + "=" * 60)
    print("TRADEOFF ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n📊 SENTIMENT ANALYSIS MODELS:")
    print("-" * 50)
    print(f"{'Model':<35} {'Load(s)':<10} {'Avg/text(ms)':<12}")
    print("-" * 50)
    for r in sentiment_results:
        print(f"{r['model']:<35} {r['load_time']:<10.2f} {r['avg_per_text']*1000:<12.1f}")

    print("\n📊 TEXT GENERATION MODELS:")
    print("-" * 50)
    print(f"{'Model':<35} {'Load(s)':<10} {'Avg/prompt(s)':<12}")
    print("-" * 50)
    for r in generation_results:
        print(f"{r['model']:<35} {r['load_time']:<10.2f} {r['avg_per_prompt']:<12.3f}")

    print("\n💡 KEY TRADEOFFS:")
    print("-" * 50)
    print("""
    SENTIMENT ANALYSIS:
    • DistilBERT: Faster, binary classification (pos/neg)
      Best for: Speed-critical applications, English only

    • BERT Multilingual: Slower, 5-star granularity
      Best for: Nuanced sentiment, multilingual support

    TEXT GENERATION:
    • DistilGPT-2: ~40% faster, smaller memory footprint
      Best for: Resource-constrained environments, quick prototypes

    • GPT-2: Better text quality, more coherent outputs
      Best for: When quality matters more than speed

    GENERAL RECOMMENDATIONS:
    • Development/Testing: Use smaller models (DistilBERT, DistilGPT-2)
    • Production: Balance based on your latency/quality requirements
    • Batch processing: Larger models become more efficient at scale
    """)


def main():
    print("🤗 Hugging Face Pipelines - Model Comparison Demo")
    print("=" * 60)
    print("This script compares different models for common NLP tasks")
    print("measuring load time, inference speed, and output quality.")

    # Run sentiment analysis comparison
    sentiment_results = compare_sentiment_models()

    # Run text generation comparison
    generation_results = compare_text_generation_models()

    # Print tradeoff analysis
    print_tradeoff_analysis(sentiment_results, generation_results)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()