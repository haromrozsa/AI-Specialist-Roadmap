package com.ai.specialist.mlapi.service;

import ai.onnxruntime.*;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

@Slf4j
@Service
public class ModelService {

    @Value("${model.path:models/mnist-12.onnx}")
    private String modelPath;

    private OrtEnvironment environment;
    private OrtSession session;
    private String inputName;
    private String outputName;

    @PostConstruct
    public void initialize() throws OrtException, IOException {
        log.info("Initializing ONNX Runtime environment...");
        environment = OrtEnvironment.getEnvironment();

        log.info("Loading MNIST model from: {}", modelPath);

        // Load model from resources or file system
        byte[] modelBytes = loadModelBytes();

        session = environment.createSession(modelBytes, new OrtSession.SessionOptions());

        // Get input/output names
        inputName = session.getInputNames().iterator().next();
        outputName = session.getOutputNames().iterator().next();

        log.info("Model loaded successfully");
        log.info("Input name: {}, Output name: {}", inputName, outputName);
        log.info("Model metadata: {}", session.getMetadata());
    }

    private byte[] loadModelBytes() throws IOException {
        // Try loading from classpath first
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(modelPath)) {
            if (is != null) {
                log.info("Loading model from classpath: {}", modelPath);
                return is.readAllBytes();
            }
        }

        // Try loading from file system
        Path path = Path.of(modelPath);
        if (Files.exists(path)) {
            log.info("Loading model from file system: {}", path.toAbsolutePath());
            return Files.readAllBytes(path);
        }

        throw new IOException("Model file not found: " + modelPath);
    }

    public float[] predict(float[][][][] inputTensor) throws OrtException {
        long startTime = System.currentTimeMillis();

        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, inputTensor)) {
            try (OrtSession.Result result = session.run(java.util.Map.of(inputName, tensor))) {
                float[][] outputArray = (float[][]) result.get(0).getValue();
                long inferenceTime = System.currentTimeMillis() - startTime;
                log.debug("Inference completed in {} ms", inferenceTime);
                return outputArray[0];
            }
        }
    }

    public String getInputName() {
        return inputName;
    }

    public String getOutputName() {
        return outputName;
    }

    public OrtSession.SessionOptions getSessionOptions() throws OrtException {
        return new OrtSession.SessionOptions();
    }

    @PreDestroy
    public void cleanup() throws OrtException {
        if (session != null) {
            log.info("Closing ONNX session...");
            session.close();
        }
        if (environment != null) {
            log.info("Closing ONNX environment...");
            environment.close();
        }
    }
}
