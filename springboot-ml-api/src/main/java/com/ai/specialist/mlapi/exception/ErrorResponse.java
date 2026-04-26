package com.ai.specialist.mlapi.exception;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "Error response")
public class ErrorResponse {

    @Schema(description = "Timestamp of the error", example = "2026-04-26T13:30:00Z")
    private String timestamp;

    @Schema(description = "HTTP status code", example = "400")
    private int status;

    @Schema(description = "Error type", example = "Validation Failed")
    private String error;

    @Schema(description = "Error message", example = "Invalid input parameters")
    private String message;

    @Schema(description = "Additional error details")
    private Map<String, String> details;
}
