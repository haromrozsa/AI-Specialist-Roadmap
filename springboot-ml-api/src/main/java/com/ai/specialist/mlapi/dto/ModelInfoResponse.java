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
@Schema(description = "Model information response")
public class ModelInfoResponse {

    @Schema(description = "Model name", example = "MNIST Digit Classifier")
    private String modelName;

    @Schema(description = "Model version", example = "1.0")
    private String version;

    @Schema(description = "Model input name", example = "Input3")
    private String inputName;

    @Schema(description = "Model output name", example = "Plus214_Output_0")
    private String outputName;

    @Schema(description = "Expected input shape", example = "[1, 1, 28, 28]")
    private String inputShape;

    @Schema(description = "Number of output classes", example = "10")
    private int outputClasses;

    @Schema(description = "Model description", example = "Pre-trained MNIST model from ONNX Model Zoo")
    private String description;
}
