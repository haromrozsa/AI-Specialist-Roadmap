package com.ai.specialist.mlapi.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
@Schema(description = "Prediction request with raw features (784 values for 28x28 MNIST)")
public class PredictionRequest {

    @NotNull(message = "Features array cannot be null")
    @Size(min = 784, max = 784, message = "Features array must contain exactly 784 values (28x28)")
    @Schema(description = "Flattened array of 784 pixel values (28x28) normalized to [0, 1]",
            example = "[0.0, 0.1, 0.2, ..., 0.9]")
    private float[] features;
}
