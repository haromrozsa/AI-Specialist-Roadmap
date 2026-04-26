package com.ai.specialist.mlapi.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "Prediction response with digit classification result")
public class PredictionResponse {

    @Schema(description = "Predicted digit (0-9)", example = "7")
    private int prediction;

    @Schema(description = "Confidence score for the prediction", example = "0.9876")
    private float confidence;

    @Schema(description = "Probability distribution across all 10 digits",
            example = "[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.9876, 0.009, 0.001]")
    private float[] probabilities;

    @Schema(description = "Inference time in milliseconds", example = "5")
    private long inferenceTimeMs;
}
