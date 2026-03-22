# FastAPI ONNX Inference API

# Session Summary: A complete ML inference API using FastAPI and ONNX Runtime, with a scikit-learn model trained on the digits dataset and exported to ONNX format.

## What I Did During the Session

1. **Trained a Machine Learning Model**:
   - Used scikit-learn's `load_digits` dataset (handwritten digit images, 64 features each)
   - Trained a `LogisticRegression` classifier with `max_iter=10000` for convergence
   - Model predicts digits 0-9 from 8x8 pixel grayscale images (flattened to 64 features)

2. **Exported Model to ONNX Format**:
   - Used `skl2onnx` library to convert the scikit-learn model to ONNX
   - Defined input type as `FloatTensorType([None, 64])` for flexible batch sizes
   - Saved the model as `model.onnx` for portable inference

3. **Built a FastAPI Inference API**:
   - Created a `/predict` POST endpoint that accepts JSON input data
   - Loaded the ONNX model using `onnxruntime.InferenceSession`
   - Implemented input normalization (dividing by 16.0) to match training preprocessing

4. **Debugged Runtime Errors**:
   - Fixed `AttributeError: 'str' object has no attribute 'read'` by using `Request` object for JSON parsing
   - Fixed `INVALID_ARGUMENT: Got: 68 Expected: 64` by validating input dimensions
   - Fixed `axis 1 is out of bounds for array of dimension 1` by reshaping 1D outputs to 2D
   - Added proper HTTP error responses with `HTTPException`

5. **Implemented Input Validation**:
   - Added checks to ensure each sample has exactly 64 features
   - Supported multiple input formats: `{"data": [[...]]}` and `[[...]]`
   - Auto-wrapped single samples into batch format for consistent processing

---

## What I Learned

1. **ONNX Model Export**:
   - **ONNX (Open Neural Network Exchange)** is a portable format for ML models
   - `skl2onnx` converts scikit-learn models to ONNX for production deployment
   - `FloatTensorType([None, n_features])` defines dynamic batch size with fixed feature count
   - ONNX models can be served without scikit-learn dependency using `onnxruntime`

2. **FastAPI Request Handling**:
   - Using `data: list` as a parameter causes FastAPI to expect form data, not JSON
   - Use `Request` object with `await request.json()` for flexible JSON parsing
   - `HTTPException` provides proper HTTP status codes instead of returning errors with 200 OK
   - Always set `Content-Type: application/json` header when sending JSON requests

3. **ONNX Runtime Inference**:
   - `InferenceSession` loads and runs ONNX models efficiently
   - `session.get_inputs()[0].name` retrieves the expected input tensor name
   - Input data must match the exact shape defined during model export
   - Output shape may vary (1D for single sample, 2D for batch) — always handle both cases

4. **Debugging ML API Errors**:
   - Dimension mismatch errors indicate input shape doesn't match model expectations
   - `axis out of bounds` errors mean array has fewer dimensions than expected
   - Use `ndim` check and `reshape()` to normalize array dimensions before processing
   - Validate input data early and return clear error messages to API consumers

5. **ML API Best Practices**:
   - Validate input dimensions before inference to provide clear error messages
   - Normalize/preprocess input data the same way as during training (e.g., `/16.0`)
   - Return both predictions and raw model output for debugging and confidence analysis
   - Use `np.argmax()` to convert probability outputs to class predictions

---

This session provided hands-on experience with end-to-end ML model deployment, from training and ONNX export to building a production-ready FastAPI inference API with proper error handling and input validation.