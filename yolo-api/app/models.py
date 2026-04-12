from pydantic import BaseModel
from typing import List, Optional


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    """Response model for /detect endpoint"""
    success: bool
    image_name: str
    image_size: dict
    detections_count: int
    detections: List[Detection]
    inference_time_ms: float


class EvaluationMetrics(BaseModel):
    """Evaluation metrics response"""
    total_images: int
    total_ground_truth: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    iou_threshold: float