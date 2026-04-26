package com.ai.specialist.mlapi.util;

import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Base64;

@Slf4j
@UtilityClass
public class ImagePreprocessor {

    private static final int MNIST_WIDTH = 28;
    private static final int MNIST_HEIGHT = 28;

    /**
     * Preprocess image from byte array to MNIST format [1, 1, 28, 28]
     */
    public static float[][][][] preprocessImage(byte[] imageBytes) throws IOException {
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (image == null) {
            throw new IOException("Failed to decode image");
        }
        return processBufferedImage(image);
    }

    /**
     * Preprocess image from Base64 string to MNIST format [1, 1, 28, 28]
     */
    public static float[][][][] preprocessBase64Image(String base64Image) throws IOException {
        // Remove data:image prefix if present
        String base64Data = base64Image;
        if (base64Image.contains(",")) {
            base64Data = base64Image.split(",")[1];
        }

        byte[] imageBytes = Base64.getDecoder().decode(base64Data);
        return preprocessImage(imageBytes);
    }

    /**
     * Preprocess image from InputStream to MNIST format [1, 1, 28, 28]
     */
    public static float[][][][] preprocessImage(InputStream inputStream) throws IOException {
        BufferedImage image = ImageIO.read(inputStream);
        if (image == null) {
            throw new IOException("Failed to decode image");
        }
        return processBufferedImage(image);
    }

    private static float[][][][] processBufferedImage(BufferedImage image) {
        log.debug("Processing image: {}x{}", image.getWidth(), image.getHeight());

        // Resize to 28x28
        BufferedImage resized = resizeImage(image, MNIST_WIDTH, MNIST_HEIGHT);

        // Convert to grayscale
        BufferedImage grayscale = convertToGrayscale(resized);

        // Normalize and convert to tensor [1, 1, 28, 28]
        float[][][][] tensor = new float[1][1][MNIST_HEIGHT][MNIST_WIDTH];

        for (int y = 0; y < MNIST_HEIGHT; y++) {
            for (int x = 0; x < MNIST_WIDTH; x++) {
                int rgb = grayscale.getRGB(x, y);
                int gray = rgb & 0xFF; // Extract gray value
                // Normalize to [0, 1]
                tensor[0][0][y][x] = gray / 255.0f;
            }
        }

        log.debug("Image preprocessed successfully");
        return tensor;
    }

    private static BufferedImage resizeImage(BufferedImage original, int targetWidth, int targetHeight) {
        Image tmp = original.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return resized;
    }

    private static BufferedImage convertToGrayscale(BufferedImage original) {
        if (original.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return original;
        }

        BufferedImage grayscale = new BufferedImage(
                original.getWidth(),
                original.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D g2d = grayscale.createGraphics();
        g2d.drawImage(original, 0, 0, null);
        g2d.dispose();

        return grayscale;
    }

    /**
     * Convert raw feature array to tensor format [1, 1, 28, 28]
     * Expects 784 features (28x28)
     */
    public static float[][][][] featuresToTensor(float[] features) {
        if (features.length != MNIST_WIDTH * MNIST_HEIGHT) {
            throw new IllegalArgumentException(
                    "Expected 784 features (28x28), got " + features.length
            );
        }

        float[][][][] tensor = new float[1][1][MNIST_HEIGHT][MNIST_WIDTH];
        int index = 0;

        for (int y = 0; y < MNIST_HEIGHT; y++) {
            for (int x = 0; x < MNIST_WIDTH; x++) {
                tensor[0][0][y][x] = features[index++];
            }
        }

        return tensor;
    }

    /**
     * Apply softmax to raw logits
     */
    public static float[] softmax(float[] logits) {
        float[] probabilities = new float[logits.length];
        float max = Float.NEGATIVE_INFINITY;

        // Find max for numerical stability
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - max);
            sum += probabilities[i];
        }

        // Normalize
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    /**
     * Get index of maximum value (argmax)
     */
    public static int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
