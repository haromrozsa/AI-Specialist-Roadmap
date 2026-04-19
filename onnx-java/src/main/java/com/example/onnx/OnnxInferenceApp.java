package com.example.onnx;

import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;

@Slf4j
public class OnnxInferenceApp {

    private static final Path MODELS_DIR  = Paths.get("onnx-java/models");
    private static final Path SAMPLES_DIR = Paths.get("onnx-java/samples");

    private static final Path DEFAULT_MODEL = Paths.get("mnist-12.onnx");
    private static final Path DEFAULT_IMAGE = Paths.get("digit.png");

    private static final Set<String> ALLOWED_MODEL_EXT = Set.of("onnx");
    private static final Set<String> ALLOWED_IMAGE_EXT = Set.of("png", "jpg", "jpeg", "bmp");

    static void main(String[] args) {
        Path rawModelPath = args.length > 0 ? Paths.get(args[0]) : DEFAULT_MODEL;
        Path rawImagePath = args.length > 1 ? Paths.get(args[1]) : DEFAULT_IMAGE;
        
        log.info("=== ONNX + Java: MNIST Inference Demo ===");

        try {
            Path modelPath = PathValidator.validate(
                    rawModelPath, MODELS_DIR, ALLOWED_MODEL_EXT);
            Path imagePath = PathValidator.validate(
                    rawImagePath, SAMPLES_DIR, ALLOWED_IMAGE_EXT);

            log.info("Model : {}", modelPath);
            log.info("Image : {}", imagePath);

            runInference(modelPath, imagePath);

        } catch (SecurityException e) {
            log.error("Rejected input: {}", e.getMessage());
            System.exit(2);
        } catch (Exception e) {
            log.error("Inference failed: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

    private static void runInference(Path modelPath, Path imagePath) throws Exception {
        ModelLoader loader = new ModelLoader();
        try (OrtSession session = loader.loadModel(modelPath)) {
            ImageClassifier classifier =
                    new ImageClassifier(loader.getEnvironment(), session);

            ImageClassifier.ClassificationResult result =
                    classifier.classify(imagePath);

            logResult(result);
        }
    }

    private static void logResult(ImageClassifier.ClassificationResult result) {
        log.info("Predicted digit : {}", result.predictedDigit());
        log.info("Confidence      : {}", String.format("%.4f", result.confidence()));
        log.info("Inference time  : {} ms", result.inferenceTimeMs());

        float[] probs = result.probabilities();
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < probs.length; i++) indices.add(i);
        indices.sort(Comparator.comparingDouble(i -> -probs[i]));

        log.info("Top-3 predictions:");
        for (int i = 0; i < 3; i++) {
            int digit = indices.get(i);
            log.info("  {}. digit {}  -  {}", i + 1, digit,
                    String.format("%.4f", probs[digit]));
        }
    }
}