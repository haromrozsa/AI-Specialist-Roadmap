"""
Post-training INT8 quantization benchmark for an ONNX CNN.

Quantization stores weights as 8-bit integers instead of 32-bit floats. The
mapping uses two numbers per tensor:

    scale = (max - min) / 255
    q     = round(x / scale) + zero_point      # float -> int8
    x'    = (q - zero_point) * scale           # int8  -> float (approximate)

x' is not exactly x. That rounding error is the entire cost of quantization: it
is lossy compression of the weights, and accuracy drops slightly as a result.

This script measures what you actually get in exchange -- file size, inference
latency, and top-1 accuracy -- for the FP32 baseline and two INT8 variants.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.datasets import fetch_openml

HERE = Path(__file__).resolve().parent

# The FP32 model produced by onnx/MnistCnnTrainerExportOnnx.py. It is a
# TensorFlow export, so its input layout is NHWC: (N, 28, 28, 1).
DEFAULT_MODEL = HERE.parent / "onnx" / "onnx" / "artifacts" / "mnist_cnn.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark INT8 quantization of an ONNX MNIST CNN."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the FP32 ONNX model (default: the repo's mnist_cnn.onnx).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=HERE / "artifacts",
        help="Directory for generated quantized models.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="Training images used to calibrate activation ranges (static only).",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=200,
        help="Timed single-image inferences per model.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
        help="Untimed inferences discarded before measuring.",
    )
    return parser.parse_args()


def resolve_model(path: Path) -> Path:
    """Fail early and actionably if the FP32 model has not been generated."""
    if not path.is_file():
        raise SystemExit(
            "FP32 model not found: {}\n\n"
            "Generate it first:\n"
            "    cd onnx && python MnistCnnTrainerExportOnnx.py\n\n"
            "Or point at a different model with --model.".format(path)
        )
    return path


def load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def model_io(session: ort.InferenceSession) -> tuple[str, list, str]:
    """
    Read tensor names and shape from the session instead of hardcoding them, so
    a re-export with different naming or batch dimensions still works.
    """
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    return inp.name, inp.shape, out.name


def to_model_layout(flat: np.ndarray, input_shape: list) -> np.ndarray:
    """
    Reshape flat 784-value rows into the model's expected layout.

    The batch dimension may be dynamic (reported as a string or None), so it is
    always replaced with -1 and inferred by NumPy.
    """
    trailing = [int(d) for d in input_shape[1:]]
    return flat.reshape(-1, *trailing)


def load_mnist(input_shape: list):
    """
    Load MNIST via scikit-learn rather than keras.datasets, so this demo does
    not depend on TensorFlow just to read data.

    Uses the canonical split: first 60,000 train, last 10,000 test. Calibration
    later draws only from the train portion -- calibrating on test data leaks
    test information into the model and invalidates the reported accuracy.
    """
    print("Loading MNIST (first run downloads ~15 MB, cached afterwards)...")
    bunch = fetch_openml("mnist_784", version=1, as_frame=False)

    # Scale to [0, 1] float32 to match how the CNN was trained.
    X = bunch.data.astype(np.float32) / 255.0
    y = bunch.target.astype(np.int64)

    X = to_model_layout(X, input_shape)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def batch_size_for(input_shape: list, preferred: int) -> int:
    """Honour a fixed batch dimension if the model has one; otherwise batch."""
    batch_dim = input_shape[0]
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return preferred


def evaluate_accuracy(session, input_name, output_name, X, y, batch_size) -> float:
    correct = 0
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size]
        logits = session.run([output_name], {input_name: chunk})[0]
        correct += int((logits.argmax(axis=1) == y[start:start + batch_size]).sum())
    return correct / len(X)


@dataclass
class Result:
    label: str
    path: Path
    size_kb: float
    accuracy: float
    median_ms: float
    p95_ms: float


def measure_latency(session, input_name, output_name, sample, warmup, runs):
    """
    Time single-image inference.

    Reports median and p95 rather than mean: latency distributions are
    right-skewed, and a mean is distorted by occasional scheduler noise.
    """
    feed = {input_name: sample}

    for _ in range(warmup):
        session.run([output_name], feed)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run([output_name], feed)
        timings.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(timings)), float(np.percentile(timings, 95))


def benchmark(label, path, input_name, output_name, input_shape,
              X_test, y_test, args) -> Result:
    session = load_session(path)
    batch = batch_size_for(input_shape, 256)

    accuracy = evaluate_accuracy(session, input_name, output_name,
                                 X_test, y_test, batch)
    median_ms, p95_ms = measure_latency(
        session, input_name, output_name, X_test[:1],
        args.warmup_runs, args.latency_runs,
    )

    return Result(
        label=label,
        path=path,
        size_kb=path.stat().st_size / 1024,
        accuracy=accuracy,
        median_ms=median_ms,
        p95_ms=p95_ms,
    )


def print_table(results: list) -> None:
    baseline = results[0]

    header = (f"{'Model':<18}{'Size (KB)':>11}{'vs FP32':>9}"
              f"{'Accuracy':>10}{'Delta':>8}{'Median ms':>11}{'p95 ms':>9}")
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        size_ratio = baseline.size_kb / r.size_kb
        acc_delta = (r.accuracy - baseline.accuracy) * 100
        print(f"{r.label:<18}{r.size_kb:>11.1f}{size_ratio:>8.2f}x"
              f"{r.accuracy * 100:>9.2f}%{acc_delta:>+7.2f}%"
              f"{r.median_ms:>11.3f}{r.p95_ms:>9.3f}")


def preprocess_graph(model_path: Path, artifacts: Path) -> Path:
    """
    Run ONNX Runtime's symbolic shape inference before quantizing.

    This is a documented prerequisite, not an optimisation: skipping it is the
    usual cause of otherwise cryptic quantization failures.
    """
    prep_path = artifacts / "mnist_cnn.prep.onnx"
    quant_pre_process(str(model_path), str(prep_path))
    return prep_path


def make_dynamic(prep_path: Path, artifacts: Path) -> Path:
    """
    Dynamic quantization: weights are converted offline, but activation ranges
    are measured at runtime on every single inference.

    That runtime measurement is why it needs no calibration data -- and why it
    mainly pays off on Transformer/RNN matmuls rather than convolutions.
    """
    out_path = artifacts / "mnist_cnn.int8_dynamic.onnx"
    quantize_dynamic(
        str(prep_path),
        str(out_path),
        weight_type=QuantType.QUInt8,
    )
    return out_path


class MnistCalibrationReader(CalibrationDataReader):
    """
    Feeds representative images through the graph so activation ranges can be
    recorded once, ahead of time.

    Calibration exists because integer arithmetic needs BOTH operands as
    integers. In y = W @ x, the weights W are known after training, but x --
    the previous layer's output -- depends on the input image. Without a range
    for x you cannot pick its scale, so W must be dequantized back to float:
    you keep the smaller file and lose the speedup entirely.

    Samples come from the TRAIN split only. Calibrating on test data leaks test
    information into the model and makes the reported accuracy meaningless.
    """

    def __init__(self, data: np.ndarray, input_name: str, batch_size: int = 32):
        self.input_name = input_name
        self.batches = [
            data[i:i + batch_size] for i in range(0, len(data), batch_size)
        ]
        self.index = 0

    def get_next(self):
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index]
        self.index += 1
        return {self.input_name: batch}

    def rewind(self):
        self.index = 0


def make_static(prep_path: Path, artifacts: Path, reader) -> Path:
    """
    Static quantization: calibrated activation ranges are baked into the graph,
    so inference is pure integer arithmetic with no per-run range scan.

    per_channel=True gives each convolution filter its own scale, which
    noticeably reduces accuracy loss on conv layers versus one scale per tensor.
    """
    out_path = artifacts / "mnist_cnn.int8_static.onnx"
    quantize_static(
        str(prep_path),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return out_path


def try_build(label: str, builder):
    """
    Quantization can fail on unsupported operator combinations. Report and carry
    on so the remaining variants still produce a comparable row.
    """
    try:
        return builder()
    except Exception as exc:
        print(f"[warn] {label} quantization failed: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    args = parse_args()
    model_path = resolve_model(args.model)
    args.artifacts.mkdir(parents=True, exist_ok=True)

    session = load_session(model_path)
    input_name, input_shape, output_name = model_io(session)

    print(f"Model        : {model_path}")
    print(f"Input        : {input_name} {input_shape}")
    print(f"Output       : {output_name}")

    X_train, y_train, X_test, y_test = load_mnist(input_shape)
    print(f"Train / test : {len(X_train)} / {len(X_test)} images")

    results = [
        benchmark("FP32 baseline", model_path, input_name, output_name,
                  input_shape, X_test, y_test, args)
    ]

    prep_path = preprocess_graph(model_path, args.artifacts)

    dynamic_path = try_build(
        "dynamic", lambda: make_dynamic(prep_path, args.artifacts)
    )
    if dynamic_path is not None:
        results.append(
            benchmark("INT8 dynamic", dynamic_path, input_name, output_name,
                      input_shape, X_test, y_test, args)
        )

    calibration_data = X_train[:args.calibration_samples]
    print(f"Calibrating on {len(calibration_data)} training images...")

    static_path = try_build(
        "static",
        lambda: make_static(
            prep_path,
            args.artifacts,
            MnistCalibrationReader(calibration_data, input_name),
        ),
    )
    if static_path is not None:
        results.append(
            benchmark("INT8 static", static_path, input_name, output_name,
                      input_shape, X_test, y_test, args)
        )

    print_table(results)


if __name__ == "__main__":
    main()
