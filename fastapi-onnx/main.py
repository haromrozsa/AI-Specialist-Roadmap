from fastapi import FastAPI, HTTPException, Request
import onnxruntime as ort
import numpy as np

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("model.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Get expected input shape from the model
expected_shape = session.get_inputs()[0].shape
expected_features = expected_shape[1] if len(expected_shape) > 1 else 64


@app.get("/")
def root():
    return {"message": "AI API is running"}


@app.post("/predict")
async def predict(request: Request):
    try:
        # Try to parse JSON from the request body
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON. Send JSON with Content-Type: application/json"
            )

        # Handle different input formats
        if isinstance(body, dict) and "data" in body:
            data = body["data"]
        elif isinstance(body, list):
            data = body
        else:
            raise HTTPException(
                status_code=400,
                detail="Expected JSON format: {\"data\": [[...], [...]]} or [[...], [...]]"
            )

        # Ensure data is 2D (wrap single sample in a list)
        if data and not isinstance(data[0], list):
            data = [data]

        # Validate input dimensions
        for i, sample in enumerate(data):
            if len(sample) != expected_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample {i} has {len(sample)} features, but model expects {expected_features}"
                )

        # Convert input to numpy + normalize
        input_array = np.array(data, dtype=np.float32) / 16.0

        # Run inference
        result = session.run([output_name], {input_name: input_array})

        # Get raw output and ensure it's 2D for consistent processing
        raw_output = result[0]
        if raw_output.ndim == 1:
            raw_output = raw_output.reshape(1, -1)

        # Get predicted class (argmax)
        prediction = np.argmax(raw_output, axis=1)

        return {
            "prediction": prediction.tolist(),
            "raw_output": raw_output.tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🚀 Run WITHOUT external uvicorn command
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)