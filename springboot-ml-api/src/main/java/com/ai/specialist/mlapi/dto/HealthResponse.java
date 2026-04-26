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
@Schema(description = "Health check response")
public class HealthResponse {

    @Schema(description = "Service status", example = "UP")
    private String status;

    @Schema(description = "Model loading status", example = "LOADED")
    private String modelStatus;

    @Schema(description = "Application version", example = "1.0.0")
    private String version;

    @Schema(description = "Timestamp of the health check", example = "2026-04-26T13:30:00Z")
    private String timestamp;
}
