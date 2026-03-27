from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    data: List[List[float]] = Field(
        ...,
        description="2D array of input features. Each inner list is one sample with 64 features.",
        min_length=1
    )

    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")

        expected_features = 64  # digits dataset has 64 features
        for i, sample in enumerate(v):
            if len(sample) != expected_features:
                raise ValueError(
                    f"Sample {i} has {len(sample)} features, expected {expected_features}"
                )
            # Validate feature values (digits dataset uses 0-16 range)
            for j, val in enumerate(sample):
                if not (0 <= val <= 16):
                    raise ValueError(
                        f"Sample {i}, feature {j}: value {val} out of range [0, 16]"
                    )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [[0.0] * 64]
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: List[int] = Field(..., description="Predicted digit classes (0-9)")
    confidence: List[float] = Field(..., description="Confidence scores for predictions")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str