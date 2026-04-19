package com.example.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/**
 * Runs inference using the MNIST ONNX model.
 *
 * Expected model: mnist-12.onnx from the ONNX Model Zoo
 *  - Input:  "Input3",             shape [1, 1, 28, 28], float32
 *  - Output: "Plus214_Output_0",   shape [1, 10], float32 (logits)
 */
@Slf4j
@RequiredArgsConstructor
public class ImageClassifier {

    private static final int IMG_SIZE = 28;
    private static final String INPUT_NAME = "Input3";

    private final OrtEnvironment environment;
    private final OrtSession session;

    public ClassificationResult classify(Path imagePath)
            throws IOException, OrtException {

        log.info("Running classification on image: {}", imagePath.toAbsolutePath());

        float[][][][] input = preprocess(imagePath);

        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, input)) {
            Map<String, OnnxTensor> inputs =
                    Collections.singletonMap(INPUT_NAME, tensor);

            long start = System.nanoTime();
            try (OrtSession.Result result = session.run(inputs)) {
                long elapsedMs = (System.nanoTime() - start) / 1_000_000;

                float[][] logits = (float[][]) result.get(0).getValue();
                float[] probs = softmax(logits[0]);

                int predicted = argmax(probs);

                log.debug("Inference finished in {} ms. Predicted digit: {} (confidence={})",
                        elapsedMs, predicted, probs[predicted]);

                return new ClassificationResult(
                        predicted, probs[predicted], probs, elapsedMs);
            }
        }
    }

    /**
     * Loads an image, converts to 28x28 grayscale, normalizes to [0, 1],
     * and shapes it as [1, 1, 28, 28] for the MNIST model.
     */
    private float[][][][] preprocess(Path imagePath) throws IOException {
        BufferedImage original = ImageIO.read(imagePath.toFile());
        if (original == null) {
            log.error("Could not read image: {}", imagePath.toAbsolutePath());
            throw new IOException(
                    "Could not read image: " + imagePath.toAbsolutePath());
        }

        log.debug("Original image size: {}x{}", original.getWidth(), original.getHeight());

        BufferedImage resized = new BufferedImage(
                IMG_SIZE, IMG_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.drawImage(original.getScaledInstance(
                IMG_SIZE, IMG_SIZE, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();

        float[][][][] input = new float[1][1][IMG_SIZE][IMG_SIZE];

        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int gray = resized.getRaster().getSample(x, y, 0);
                // MNIST expects white digit on black background.
                // If your sample has black digit on white, invert:
                //   float value = (255 - gray) / 255.0f;
                float value = gray / 255.0f;
                input[0][0][y][x] = value;
            }
        }

        log.debug("Image preprocessed to {}x{} grayscale tensor", IMG_SIZE, IMG_SIZE);
        return input;
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;

        float sum = 0f;
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < exps.length; i++) exps[i] /= sum;
        return exps;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) best = i;
        }
        return best;
    }

    public record ClassificationResult(
            int predictedDigit,
            float confidence,
            float[] probabilities,
            long inferenceTimeMs) {
    }
}