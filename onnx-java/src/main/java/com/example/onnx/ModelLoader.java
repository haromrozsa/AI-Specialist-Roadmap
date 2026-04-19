package com.example.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;

@Slf4j
public class ModelLoader {

    @Getter
    private final OrtEnvironment environment;

    public ModelLoader() {
        this.environment = OrtEnvironment.getEnvironment();
        log.debug("ONNX Runtime environment initialized");
    }

    public OrtSession loadModel(Path modelPath) throws OrtException, IOException {
        Path absolute = modelPath.toAbsolutePath().normalize();
        log.info("Loading ONNX model from: {}", absolute);

        if (!Files.isRegularFile(absolute, LinkOption.NOFOLLOW_LINKS)) {
            log.error("Model file not found or not a regular file: {}", absolute);
            throw new IllegalArgumentException(
                    "Model file not found or not a regular file: " + absolute);
        }

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT);

        OrtSession session = environment.createSession(absolute.toString(), options);
        log.info("Model loaded successfully. Inputs: {}, Outputs: {}",
                session.getInputNames(), session.getOutputNames());

        return session;
    }
}