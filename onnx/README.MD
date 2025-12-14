# ML Fundamentals – CNN to ONNX (MNIST)

# Session Summary: Train a CNN on MNIST and Export to ONNX

## What I Did During the Session

1. **Introduced ONNX and ONNX Runtime**
    - Explained what ONNX is (a portable, framework-neutral model format)
    - Explained the role of ONNX inference engines and how ONNX Runtime executes optimized graphs on CPU/GPU

2. **Discussed Local ONNX Inference**
    - Explained ONNX inference on local ONNX Runtime
    - Clarified the difference between CPU vs. GPU execution (via different runtime backends)

3. **Explored Deployment to Spring Boot**
    - Explained how to run ONNX models inside Spring Boot using ONNX Runtime Java
    - Explained the typical approach: ship the `.onnx` model, load it once at startup, and run inference in a service layer

4. **Implemented “Train CNN on MNIST; Export to ONNX”**
    - Implemented `MnistCnnTrainerExportOnnx.py` in the `onnx/` folder
    - Trained a CNN on MNIST, evaluated test accuracy, and saved a TensorFlow SavedModel artifact
    - Exported the trained model to `mnist_cnn.onnx` using `tf2onnx` with a stable input signature and configurable opset

5. **Covered Dependency Installation (pip)**
    - Documented how to install ONNX export dependencies via pip: `tf2onnx` and `onnx`
    - Mentioned ONNX Runtime as the typical runtime for executing ONNX models after export

## What I Learned

1. **ONNX vs. ONNX Runtime**
    - ONNX is the model interchange format; ONNX Runtime is a common engine used to run ONNX models efficiently

2. **Export Requirements**
    - Exporting Keras/TensorFlow models to ONNX typically requires a converter library (e.g., `tf2onnx`)
    - Defining an explicit input signature improves export stability (shape + dtype + input name)

3. **Practical Deployment Considerations**
    - Local inference and server-side inference are both feasible with ONNX Runtime
    - ONNX models can be used from Java/Spring Boot, but input preprocessing and tensor shapes/dtypes must match what the model expects