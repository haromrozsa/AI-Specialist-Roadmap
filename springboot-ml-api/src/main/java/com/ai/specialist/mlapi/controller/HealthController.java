package com.ai.specialist.mlapi.controller;

import com.ai.specialist.mlapi.dto.HealthResponse;
import com.ai.specialist.mlapi.dto.ModelInfoResponse;
import com.ai.specialist.mlapi.service.ModelService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Tag(name = "Health & Info", description = "Health check and model information endpoints")
public class HealthController {

    private final ModelService modelService;

    @Value("${spring.application.name:Spring Boot ML API}")
    private String applicationName;

    @Value("${app.version:1.0.0}")
    private String version;

    @GetMapping("/health")
    @Operation(
            summary = "Health check",
            description = "Returns the health status of the application and model",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Service is healthy",
                            content = @Content(schema = @Schema(implementation = HealthResponse.class)))
            }
    )
    public HealthResponse health() {
        String modelStatus = (modelService != null) ? "LOADED" : "NOT_LOADED";

        return HealthResponse.builder()
                .status("UP")
                .modelStatus(modelStatus)
                .version(version)
                .timestamp(Instant.now().toString())
                .build();
    }

    @GetMapping("/model/info")
    @Operation(
            summary = "Get model information",
            description = "Returns detailed information about the loaded MNIST model",
            responses = {
                    @ApiResponse(responseCode = "200", description = "Model information",
                            content = @Content(schema = @Schema(implementation = ModelInfoResponse.class)))
            }
    )
    public ModelInfoResponse modelInfo() {
        return ModelInfoResponse.builder()
                .modelName("MNIST Digit Classifier")
                .version("1.0")
                .inputName(modelService.getInputName())
                .outputName(modelService.getOutputName())
                .inputShape("[1, 1, 28, 28]")
                .outputClasses(10)
                .description("Pre-trained MNIST model from ONNX Model Zoo for handwritten digit classification")
                .build();
    }
}
