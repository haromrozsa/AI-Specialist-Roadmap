#!/usr/bin/env python3
"""
Train a CNN on MNIST and export to ONNX.

Outputs:
- SavedModel:  onnx/artifacts/saved_model/
- ONNX model:  onnx/artifacts/mnist_cnn.onnx
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf


def build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def export_to_onnx(model: tf.keras.Model, onnx_path: str, opset: int = 13):
    try:
        import tf2onnx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Missing dependency: tf2onnx.\n"
            "Install it in your environment (condavenv) and re-run.\n"
            "Example (one of these, depending on your setup):\n"
            "  condavenv install tf2onnx onnx\n"
            "or\n"
            "  condavenv install -c conda-forge tf2onnx onnx\n"
        ) from e

    # Define input signature for stable export
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

    _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=onnx_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train MNIST CNN and export to ONNX")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--artifacts-dir", type=str, default=os.path.join("onnx", "artifacts"))
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_mnist_cnn()
    model.summary()

    print("\nTraining...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")

    # Save SavedModel (useful as an intermediate artifact)
    saved_model_dir = os.path.join(args.artifacts_dir, "saved_model")
    print(f"\nSaving TensorFlow SavedModel to: {saved_model_dir}")
    model.save(saved_model_dir)

    # Export to ONNX
    onnx_path = os.path.join(args.artifacts_dir, "mnist_cnn.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    export_to_onnx(model, onnx_path, opset=args.opset)

    print("\nDone.")
    print(f"Artifacts written under: {args.artifacts_dir}")
    print(f"ONNX model: {onnx_path}")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    _ = history  # silence linters


if __name__ == "__main__":
    main()
