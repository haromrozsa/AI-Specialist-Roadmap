"""
MCP Inference Server
====================

A minimal **Model Context Protocol (MCP)** server that exposes a trained ONNX
digit classifier as tools an LLM client (Claude Desktop, an agent, etc.) can
call. It demonstrates the core idea of MCP: a model or capability becomes a
*tool* behind a standard protocol, so any MCP-aware client can discover and use
it without bespoke glue code.

The model is the same scikit-learn ``LogisticRegression`` used in the
``fastapi-onnx`` demo, trained on the sklearn *digits* dataset (8x8 images,
64 features, pixel values 0-16, 10 classes 0-9).

Two tools are exposed:

* ``get_sample(index)``   -> returns a real digits sample to classify
* ``classify_digit(...)`` -> runs the ONNX model on a 64-feature vector

Run it directly for a real client to attach over stdio::

    python server.py

Or drive it with the included ``client_demo.py`` (no API key needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_digits

from mcp.server.fastmcp import FastMCP

# --- Constants describing the model / dataset ---------------------------------
EXPECTED_FEATURES = 64  # flattened 8x8 image
N_CLASSES = 10          # digits 0-9
PIXEL_MAX = 16.0        # digits pixels are 0..16; the model was trained on /16

# --- Load assets ONCE at import time ------------------------------------------
# Loading the ONNX session and the dataset on every tool call would be wasteful;
# an MCP server is a long-lived process, so we do it here at startup.
_DIGITS = load_digits()
_X = _DIGITS.data                      # shape (1797, 64), float, values 0..16
_Y = _DIGITS.target                    # shape (1797,), int labels 0..9

_MODEL_PATH = Path(__file__).parent / "model.onnx"
_session = ort.InferenceSession(str(_MODEL_PATH))
_input_name = _session.get_inputs()[0].name  # 'float_input' for this model

# The FastMCP object is the server. The name is what clients see.
mcp = FastMCP("digit-inference")


@mcp.tool()
def get_sample(index: int) -> dict:
    """Return one handwritten-digit sample from the sklearn digits dataset.

    Gives an agent something concrete to classify: fetch a sample here, then
    pass its ``features`` straight into ``classify_digit``.

    Args:
        index: Row in the dataset, 0 to 1796 inclusive.

    Returns:
        dict with ``index``, ``features`` (64 pixel values 0-16), and the
        ``true_label`` so you can check the prediction.
    """
    if not 0 <= index < len(_X):
        raise ValueError(f"index must be in [0, {len(_X) - 1}], got {index}")

    return {
        "index": index,
        "features": _X[index].tolist(),
        "true_label": int(_Y[index]),
    }


@mcp.tool()
def classify_digit(features: list[float]) -> dict:
    """Classify a single 8x8 digit given its 64 pixel values.

    Args:
        features: Exactly 64 pixel values in the range 0-16 (as returned by
            ``get_sample``). They are normalized by /16 to match training.

    Returns:
        dict with the predicted digit (``prediction``), the model's
        ``confidence`` in it (0-1), and the full ``probabilities`` list.
    """
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"expected {EXPECTED_FEATURES} features, got {len(features)}"
        )

    # Same preprocessing the model was trained with: scale pixels to [0, 1].
    arr = (np.array(features, dtype=np.float32) / PIXEL_MAX).reshape(1, -1)

    # None => return every output. skl2onnx classifiers emit two:
    # [0] the predicted label, [1] the class probabilities.
    outputs = _session.run(None, {_input_name: arr})
    probs = _extract_probabilities(outputs)

    prediction = int(np.argmax(probs))
    return {
        "prediction": prediction,
        "confidence": float(probs[prediction]),
        "probabilities": [float(p) for p in probs],
    }


def _extract_probabilities(outputs) -> list[float]:
    """Pull an ordered probability list (class 0..9) out of the ONNX outputs.

    ML-correctness gotcha worth knowing: skl2onnx exports the probability output
    as a *ZipMap* -- onnxruntime hands it back as a Python list (one entry per
    input row) of ``{class_int: probability}`` dicts, NOT a plain array. So we
    read the dict for our single row and order it by class. (Some export
    configs disable ZipMap and return an ndarray instead; we handle both.)
    """
    prob_out = outputs[1] if len(outputs) > 1 else outputs[0]
    row = prob_out[0]  # first (and only) input row
    if isinstance(row, dict):
        return [float(row[c]) for c in range(N_CLASSES)]
    return [float(p) for p in row]


if __name__ == "__main__":
    # stdio is the standard MCP transport: the client launches this script as a
    # subprocess and talks JSON-RPC over stdin/stdout.
    mcp.run(transport="stdio")
