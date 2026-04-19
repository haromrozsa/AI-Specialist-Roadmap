# ONNX + Java (MNIST Image Classification)

## Session Summary: Built a Java console application that runs inference on a pre-trained MNIST ONNX model using ONNX Runtime, demonstrating that AI inference can be integrated directly into JVM-based applications without Python.

## What I Did During the Session

1. **Planned the Implementation Strategy**:
    - Evaluated candidate models: MNIST, MobileNetV2, SqueezeNet, and Fashion-MNIST
    - Chose MNIST (`mnist-12.onnx` from the ONNX Model Zoo) for its tiny size (~26 KB), trivial preprocessing, and clean demo output
    - Designed a three-class structure: `ModelLoader`, `ImageClassifier`, and `OnnxInferenceApp`
    - Set up Maven with ONNX Runtime `1.18.0`, Lombok `1.18.42`, SLF4J + Logback for logging
    - Added the `maven-shade-plugin` to produce a runnable fat JAR

2. **Implemented the ONNX Inference Pipeline**:
    - Built `ModelLoader` to create an `OrtEnvironment` and `OrtSession` with optimization level `ALL_OPT`
    - Implemented `ImageClassifier` with image preprocessing (resize to 28×28, grayscale, normalize to `[0, 1]`)
    - Shaped the input tensor as `[1, 1, 28, 28]` to match the model's `Input3` input
    - Ran inference via `session.run(...)` and decoded the `Plus214_Output_0` logits with a custom `softmax` + `argmax`
    - Returned predictions as a clean `ClassificationResult` record (digit, confidence, probabilities, inference time)

3. **Added Professional Logging with Lombok + SLF4J**:
    - Annotated classes with `@Slf4j`, `@RequiredArgsConstructor`, `@Getter`, and `@UtilityClass`
    - Configured `logback.xml` with a readable pattern (`HH:mm:ss.SSS [thread] LEVEL logger - msg`)
    - Silenced noisy `ai.onnxruntime` internal logs at `WARN` level
    - Resolved a `TypeTag :: UNKNOWN` error by upgrading Lombok to a JDK 25–compatible version
    - Replaced all `System.out.println` calls with structured `log.info` / `log.debug` / `log.error` messages

4. **Hardened the App Against Path Traversal Vulnerabilities**:
    - Created a dedicated `PathValidator` utility with canonicalization and base-directory enforcement
    - Used `Path.toRealPath()` to resolve symlinks and `..` segments before validation
    - Enforced a file-extension allow-list (`onnx` for models, `png/jpg/jpeg/bmp` for images)
    - Rejected paths escaping the `models/` and `samples/` base directories with clear `SecurityException` messages
    - Fixed a default-path duplication bug where the base directory was accidentally prepended twice

5. **Prepared Model and Sample Assets**:
    - Downloaded `mnist-12.onnx` from the official ONNX Model Zoo on GitHub
    - Wrote a Python helper script (`sample_generator.py`) using Pillow to generate a clean 28×28 digit image with a white glyph on a black background
    - Documented multiple ways to obtain MNIST samples (ONNX Model Zoo, Kaggle, Hugging Face, local generation)
    - Verified the end-to-end pipeline: model loading → preprocessing → inference → top-3 prediction output

## What I Learned

1. **ONNX Runtime in Java**:
    - ONNX Runtime ships an official Java binding (`com.microsoft.onnxruntime:onnxruntime`) that runs on CPU out of the box
    - `OrtEnvironment` is a singleton-like global environment, while `OrtSession` is the per-model inference context
    - Input tensors are created via `OnnxTensor.createTensor(env, nd-array)` and must match the model's exact shape and dtype
    - Model metadata (`session.getInputNames()` / `getOutputNames()`) is invaluable when adapting code to a new model
    - Both tensors and sessions implement `AutoCloseable` — using try-with-resources prevents native memory leaks

2. **Image Preprocessing in Pure Java**:
    - `ImageIO.read(...)` + `BufferedImage` is enough for simple grayscale classification tasks — no OpenCV required
    - MNIST models expect **white digits on a black background**; inverted images silently produce wrong predictions
    - Normalization (`pixel / 255.0f`) and tensor shape (`[N, C, H, W]`) must exactly match the model's training conditions
    - Downscaling with `Image.SCALE_SMOOTH` preserves digit shapes much better than nearest-neighbor resizing
    - Softmax must be computed with a `max` subtraction trick to avoid `Float.POSITIVE_INFINITY` from `Math.exp(...)`

3. **Lombok + Modern JDK Compatibility**:
    - Lombok relies on internal `com.sun.tools.javac.*` APIs, so each new JDK release can break older Lombok versions
    - The error `ExceptionInInitializerError: TypeTag :: UNKNOWN` is the classic "Lombok too old for this JDK" signature
    - Lombok `1.18.42+` adds official JDK 25 support; always bump Lombok when upgrading the JDK
    - `@Slf4j` generates a `private static final Logger log` field, which is cleaner than manual SLF4J boilerplate
    - `@UtilityClass` automatically makes the class `final`, adds a private constructor, and makes all members `static`

4. **Path Traversal and Secure File Handling**:
    - Any code that accepts a user-provided file path is potentially vulnerable — even a local CLI tool, because static analyzers will flag it
    - Canonicalization with `Path.toRealPath()` is essential to resolve symlinks and `..` segments before any `startsWith` check
    - An allow-list base directory + allow-list of extensions stops the vast majority of path traversal attacks
    - Returning a clean `SecurityException` with a clear message helps both debugging and incident forensics
    - When defaults use `baseDir.resolve(...)`, do **not** also prepend the base inside the validator, or you get duplicated paths

5. **AI Deployment Beyond Python**:
    - ONNX is the portable bridge between training frameworks (PyTorch, TensorFlow, scikit-learn) and any runtime target
    - Running ONNX models in Java removes Python as a deployment dependency — huge for enterprise JVM environments
    - The same `.onnx` file can be served from FastAPI in Python **and** from a Java CLI/service, identical outputs
    - Inference on CPU for a small model (MNIST) is effectively free (~3–5 ms), making Java + ONNX viable for latency-sensitive systems
    - Clean separation of loading, preprocessing, inference, and result decoding is the foundation for scaling this pattern to REST APIs, batch jobs, or embedded systems

This session provided hands-on experience with running AI inference in Java using ONNX Runtime, integrating professional logging and Lombok, hardening file-input handling against path traversal, and producing a clean, reproducible portfolio artifact demonstrating that AI is not limited to Python.