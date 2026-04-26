package com.ai.specialist.mlapi.controller;

import com.ai.specialist.mlapi.dto.Base64PredictionRequest;
import com.ai.specialist.mlapi.dto.PredictionRequest;
import com.ai.specialist.mlapi.dto.PredictionResponse;
import com.ai.specialist.mlapi.service.ModelService;
import com.ai.specialist.mlapi.util.ImagePreprocessor;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Tag(name = "Prediction", description = "MNIST digit prediction endpoints")
public class PredictionController {

    private final ModelService modelService;

    @PostMapping("/predict")
    @Operation(
            summary = "Predict digit from raw features",
            description = "Accepts a flattened array of 784 pixel values (28x28) normalized to [0, 1]",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Successful prediction",
                            content = @Content(schema = @Schema(implementation = PredictionResponse.class))),
                    @ApiResponse(responseCode = "400", description = "Invalid input")
            }
    )
    public PredictionResponse predictFromFeatures(
            @Valid @RequestBody PredictionRequest request) throws Exception {

        log.info("Received prediction request with {} features", request.getFeatures().length);
        long startTime = System.currentTimeMillis();

        // Convert features to tensor
        float[][][][] tensor = ImagePreprocessor.featuresToTensor(request.getFeatures());

        // Predict
        float[] logits = modelService.predict(tensor);
        float[] probabilities = ImagePreprocessor.softmax(logits);
        int prediction = ImagePreprocessor.argmax(probabilities);
        float confidence = probabilities[prediction];

        long inferenceTime = System.currentTimeMillis() - startTime;

        log.info("Prediction: {}, Confidence: {:.4f}, Time: {} ms",
                prediction, confidence, inferenceTime);

        return PredictionResponse.builder()
                .prediction(prediction)
                .confidence(confidence)
                .probabilities(probabilities)
                .inferenceTimeMs(inferenceTime)
                .build();
    }

    @PostMapping("/predict/image")
    @Operation(
            summary = "Predict digit from uploaded image",
            description = "Accepts an image file (PNG, JPG, etc.) and predicts the digit",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Successful prediction",
                            content = @Content(schema = @Schema(implementation = PredictionResponse.class))),
                    @ApiResponse(responseCode = "400", description = "Invalid image file")
            }
    )
    public PredictionResponse predictFromImage(
            @Parameter(description = "Image file containing a handwritten digit")
            @RequestParam("file") MultipartFile file) throws Exception {

        log.info("Received image upload: {}, Size: {} bytes",
                file.getOriginalFilename(), file.getSize());

        long startTime = System.currentTimeMillis();

        // Preprocess image
        float[][][][] tensor = ImagePreprocessor.preprocessImage(file.getInputStream());

        // Predict
        float[] logits = modelService.predict(tensor);
        float[] probabilities = ImagePreprocessor.softmax(logits);
        int prediction = ImagePreprocessor.argmax(probabilities);
        float confidence = probabilities[prediction];

        long inferenceTime = System.currentTimeMillis() - startTime;

        log.info("Prediction from image: {}, Confidence: {:.4f}, Time: {} ms",
                prediction, confidence, inferenceTime);

        return PredictionResponse.builder()
                .prediction(prediction)
                .confidence(confidence)
                .probabilities(probabilities)
                .inferenceTimeMs(inferenceTime)
                .build();
    }

    @PostMapping("/predict/base64")
    @Operation(
            summary = "Predict digit from Base64-encoded image",
            description = "Accepts a Base64-encoded image string and predicts the digit",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Successful prediction",
                            content = @Content(schema = @Schema(implementation = PredictionResponse.class))),
                    @ApiResponse(responseCode = "400", description = "Invalid Base64 image")
            }
    )
    public PredictionResponse predictFromBase64(
            @Valid @RequestBody Base64PredictionRequest request) throws Exception {

        log.info("Received Base64 image prediction request");
        long startTime = System.currentTimeMillis();

        // Preprocess Base64 image
        float[][][][] tensor = ImagePreprocessor.preprocessBase64Image(request.getImage());

        // Predict
        float[] logits = modelService.predict(tensor);
        float[] probabilities = ImagePreprocessor.softmax(logits);
        int prediction = ImagePreprocessor.argmax(probabilities);
        float confidence = probabilities[prediction];

        long inferenceTime = System.currentTimeMillis() - startTime;

        log.info("Prediction from Base64: {}, Confidence: {:.4f}, Time: {} ms",
                prediction, confidence, inferenceTime);

        return PredictionResponse.builder()
                .prediction(prediction)
                .confidence(confidence)
                .probabilities(probabilities)
                .inferenceTimeMs(inferenceTime)
                .build();
    }
}
