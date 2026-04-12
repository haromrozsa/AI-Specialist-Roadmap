"""
YOLO Model Evaluation Script
Calculates Precision, Recall, and F1-Score on a test dataset
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import cv2
from ultralytics import YOLO


@dataclass
class BBox:
    """Bounding box with class"""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float = 1.0


def calculate_iou(box1: BBox, box2: BBox) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    # Calculate intersection coordinates
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


def load_ground_truth(annotations_path: Path) -> Dict[str, List[BBox]]:
    """
    Load ground truth annotations from JSON files

    Expected format per file:
    {
        "image_name": "image1.jpg",
        "annotations": [
            {"class": "person", "bbox": [x1, y1, x2, y2]},
            ...
        ]
    }
    """
    ground_truth = {}

    for json_file in annotations_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        image_name = data.get("image_name", json_file.stem + ".jpg")
        boxes = []

        for ann in data.get("annotations", []):
            bbox = ann["bbox"]
            boxes.append(BBox(
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[2],
                y2=bbox[3],
                class_name=ann["class"]
            ))

        ground_truth[image_name] = boxes

    return ground_truth


def run_detection(model: YOLO, image_path: Path, confidence: float = 0.25) -> List[BBox]:
    """Run YOLO detection on a single image"""
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    results = model(image, conf=confidence, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append(BBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_name=results[0].names[class_id],
            confidence=conf
        ))

    return detections


def evaluate_detections(
        predictions: List[BBox],
        ground_truth: List[BBox],
        iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Evaluate predictions against ground truth

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    tp = 0
    matched_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            # Check if same class
            if pred.class_name.lower() != gt.class_name.lower():
                continue

            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn


def run_evaluation(
        model_path: str = "yolov8n.pt",
        test_images_path: str = "test_images",
        annotations_path: str = "evaluation/annotations",
        confidence: float = 0.25,
        iou_threshold: float = 0.5
) -> Dict:
    """
    Run full evaluation on test dataset
    """
    print("=" * 60)
    print("YOLO Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # Load paths
    test_path = Path(test_images_path)
    ann_path = Path(annotations_path)

    # Load ground truth
    print(f"📋 Loading annotations from: {ann_path}")
    ground_truth = load_ground_truth(ann_path)
    print(f"   Found annotations for {len(ground_truth)} images")

    # Get test images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = [f for f in test_path.iterdir()
                   if f.suffix.lower() in image_extensions]
    print(f"🖼️  Found {len(test_images)} test images")

    # Run evaluation
    print(f"\n🔍 Running evaluation (IoU threshold: {iou_threshold})...")
    print("-" * 60)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    results_per_image = []

    for image_path in sorted(test_images):
        image_name = image_path.name

        # Get ground truth for this image
        gt_boxes = ground_truth.get(image_name, [])

        # Run detection
        pred_boxes = run_detection(model, image_path, confidence)

        # Evaluate
        tp, fp, fn = evaluate_detections(pred_boxes, gt_boxes, iou_threshold)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        result = {
            "image": image_name,
            "ground_truth": len(gt_boxes),
            "predictions": len(pred_boxes),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        results_per_image.append(result)

        status = "✅" if fp == 0 and fn == 0 else "⚠️"
        print(f"  {status} {image_name}: GT={len(gt_boxes)}, "
              f"Pred={len(pred_boxes)}, TP={tp}, FP={fp}, FN={fn}")

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n📈 Dataset Statistics:")
    print(f"   • Total images evaluated: {len(test_images)}")
    print(f"   • Total ground truth objects: {total_gt}")
    print(f"   • Total predictions: {total_pred}")

    print(f"\n🎯 Detection Results:")
    print(f"   • True Positives (TP):  {total_tp}")
    print(f"   • False Positives (FP): {total_fp}")
    print(f"   • False Negatives (FN): {total_fn}")

    print(f"\n📏 Metrics (IoU ≥ {iou_threshold}):")
    print(f"   • Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"   • Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"   • F1-Score:  {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("=" * 60)

    # Create output
    evaluation_results = {
        "model": model_path,
        "confidence_threshold": confidence,
        "iou_threshold": iou_threshold,
        "dataset": {
            "total_images": len(test_images),
            "total_ground_truth": total_gt,
            "total_predictions": total_pred
        },
        "results": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        },
        "per_image_results": results_per_image
    }

    # Save results
    output_path = Path("outputs/evaluation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")

    return evaluation_results


if __name__ == "__main__":
    run_evaluation()