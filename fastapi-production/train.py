from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load dataset
X, y = load_digits(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X, y)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Model exported to model.onnx")