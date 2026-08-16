# Model Optimization – Post-Training INT8 Quantization (ONNX)

# Session Summary: Quantize an ONNX CNN and Measure the Real Tradeoff

Quantizes the MNIST CNN from [`onnx/`](../onnx) to INT8 using both dynamic and
static post-training quantization, then benchmarks size, accuracy, and latency
against the FP32 baseline.

## What I Did During the Session

1. **Implemented `quantize_benchmark.py`**
   - Reads the model's input name and shape from the ONNX Runtime session rather
     than hardcoding them
   - Loads MNIST via `sklearn.datasets.fetch_openml` so the demo needs no
     TensorFlow just to read data
   - Produces two INT8 variants and benchmarks all three models in one run

2. **Applied graph pre-processing before quantizing**
   - `onnxruntime.quantization.shape_inference.quant_pre_process()` runs symbolic
     shape inference — a documented prerequisite, not an optimization

3. **Dynamic quantization** — `quantize_dynamic(weight_type=QuantType.QUInt8)`

4. **Static quantization** — `quantize_static(quant_format=QuantFormat.QDQ,
   per_channel=True)` with a custom `CalibrationDataReader` feeding 200 training
   images in batches of 32

5. **Benchmarked size, accuracy, and latency** across all three models

## Results

Measured on the repository's own `mnist_cnn.onnx` (10,000 test images, 200 timed
single-image inferences after 20 warmup runs):

```
Model               Size (KB)  vs FP32  Accuracy   Delta  Median ms   p95 ms
----------------------------------------------------------------------------
FP32 baseline           367.7    1.00x    98.88%  +0.00%      0.101    0.132
INT8 dynamic            103.2    3.56x    98.90%  +0.02%      0.334    0.410
INT8 static             109.4    3.36x    98.89%  +0.01%      0.130    0.172
```

Environment: Python 3.8.8, onnxruntime 1.19.2, onnx 1.17.0, CPU execution provider.

Sizes and accuracy reproduce exactly across runs. Latency does not — repeated runs
put FP32 at 0.10–0.13 ms and static INT8 at 0.12–0.13 ms, so treat the two as
comparable rather than reading a precise ratio off a single run. The dynamic result
(~0.33 ms) sits well outside that band and is a genuine difference.

**Size:** both variants shrink the model by ~3.5x, close to the theoretical 4x
from 32-bit → 8-bit. The gap exists because not every tensor is quantized and
the graph structure itself carries fixed overhead.

**Accuracy:** effectively unchanged. All deltas are within ±0.02%, which on 10,000
images is one or two samples — noise, not a real improvement. The honest reading is
that INT8 cost this model no measurable accuracy.

**Latency:** this is where the interesting result is, and it is not the flattering
one. Dynamic quantization came out **3.3x slower** than FP32. Static came out
roughly at parity (0.130 vs 0.101 ms), and **2.6x faster than dynamic**. See the
caveat section below.

## What I Learned

1. **What quantization actually is**

   Weights are stored as 8-bit integers instead of 32-bit floats, using two
   numbers per tensor:

   ```
   scale = (max - min) / 255
   q     = round(x / scale) + zero_point      # float -> int8
   x'    = (q - zero_point) * scale           # int8  -> float (approximate)
   ```

   `x'` is not exactly `x`. That rounding error is the entire cost — quantization
   is lossy compression of the weights.

2. **Weights need no calibration. Activations do.**

   This was the part I had backwards initially. Quantizing weights is trivial:
   they are fixed after training, so you convert them offline and you are done.

   The problem is the *other* operand. Every layer is essentially `y = W @ x`,
   and integer arithmetic needs **both** operands as integers:

   | | Weights `W` | Activations `x` |
   |---|---|---|
   | Known before inference? | Yes | No — depends on the input image |
   | Needs calibration? | **No** | **Yes** |

   Without a range for `x` you cannot pick its scale, so `W` has to be
   dequantized back to float to multiply with a float `x`. You keep the 4x
   smaller file and lose the speedup entirely. Calibration exists solely to
   learn those activation ranges ahead of time.

3. **Dynamic vs. static is exactly that distinction**

   - **Dynamic** measures activation ranges at runtime, on every single
     inference. No calibration data needed, but you pay that scan every time.
   - **Static** pushes calibration samples through once, records the ranges, and
     bakes them into the graph. Inference is then pure integer arithmetic.

   The measured 2.6x gap between dynamic and static is that per-inference scan.
   Dynamic quantization is generally worth it for Transformer/RNN matmuls; for
   convolutions, static is the one that actually accelerates anything.

4. **Calibration data must come from the training split**

   Calibration estimates the activation distribution the model will meet in
   production. Drawing those samples from the test set leaks test information
   into the model, and the accuracy you then report is no longer measuring
   generalization. This script slices `X_train[:200]` and never touches
   `X_test` for calibration.

5. **INT8 is not automatically faster — and why it wasn't here**

   Two things have to be true for INT8 to win on latency, and neither is
   guaranteed:

   - **The CPU needs integer acceleration.** On x86 that means VNNI. Without it,
     INT8 ops get emulated and the arithmetic advantage evaporates.
   - **The model has to be big enough for arithmetic to dominate.** This CNN runs
     in ~0.1 ms. At that scale, the fixed cost of quantize/dequantize nodes that
     QDQ inserts around each operator is a large fraction of total runtime, so
     INT8 overhead swamps INT8 savings.

   This is why static landed at parity rather than ahead, and why dynamic landed
   well behind. On a larger model — or on hardware with VNNI — the ordering would
   be expected to change. The lesson is that quantization's payoff is
   **hardware- and size-dependent**, and the only way to know is to measure it on
   the target rather than assume the textbook result.

6. **`quant_pre_process` matters**

   Skipping the symbolic shape inference pass is the usual cause of otherwise
   cryptic quantization failures. It is a prerequisite, not a tuning knob.

7. **`per_channel=True` for convolutions**

   Giving each conv filter its own scale, instead of one scale for the whole
   weight tensor, reduces accuracy loss — filters with very different value
   ranges otherwise get squeezed into a single shared range.

## How to Run

The FP32 model comes from [`onnx/MnistCnnTrainerExportOnnx.py`](../onnx). Generate
it first if `onnx/onnx/artifacts/mnist_cnn.onnx` does not exist:

```bash
cd onnx && python MnistCnnTrainerExportOnnx.py
```

Then:

```bash
cd onnx-quantization
pip install -r requirements.txt
python quantize_benchmark.py
```

The first run downloads MNIST (~15 MB) via scikit-learn and caches it.

Options:

```bash
python quantize_benchmark.py --model path/to/other.onnx   # different model
python quantize_benchmark.py --calibration-samples 500    # more calibration data
python quantize_benchmark.py --latency-runs 1000          # steadier timings
```

Quantized models are written to `onnx-quantization/artifacts/` (git-ignored —
they are reproducible from the script).

## Files

| File | Purpose |
|---|---|
| `quantize_benchmark.py` | Quantization + benchmark script |
| `requirements.txt` | onnx, onnxruntime, numpy, scikit-learn |
| `artifacts/` | Generated quantized models (git-ignored) |
