package com.ai.specialist.mlapi.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
@Schema(description = "Prediction request with Base64-encoded image")
public class Base64PredictionRequest {

    @NotBlank(message = "Base64 image string cannot be blank")
    @Schema(description = "Base64-encoded image string (with or without data:image prefix)",
            example = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA...")
    private String image;
}
