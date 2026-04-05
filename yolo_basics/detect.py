"""
Week 15 - YOLO Basics: Object Detection
Script to detect objects in multiple images using YOLOv8
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2


def create_output_directory(output_path):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"✓ Created output directory: {output_path}")
    else:
        print(f"✓ Output directory exists: {output_path}")


def get_image_files(input_path):
    """Get all image files from the input directory"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if not os.path.exists(input_path):
        print(f"✗ Error: Input path '{input_path}' does not exist!")
        return []

    image_files = []
    for file in os.listdir(input_path):
        if Path(file).suffix.lower() in supported_formats:
            image_files.append(os.path.join(input_path, file))

    return sorted(image_files)


def detect_objects(model, image_path, output_path, confidence_threshold=0.25):
    """
    Detect objects in a single image and save annotated result

    Args:
        model: YOLO model instance
        image_path: Path to input image
        output_path: Path to save annotated image
        confidence_threshold: Minimum confidence for detection
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Failed to read image: {image_path}")
            return False

        # Perform detection
        results = model(image, conf=confidence_threshold, verbose=False)

        # Get annotated image
        annotated_image = results[0].plot()

        # Generate output filename
        input_filename = Path(image_path).stem
        output_filename = f"{input_filename}_detected.jpg"
        output_file_path = os.path.join(output_path, output_filename)

        # Save annotated image
        cv2.imwrite(output_file_path, annotated_image)

        # Get detection details
        detections = results[0].boxes
        num_objects = len(detections)

        # Get detected classes
        detected_classes = []
        if num_objects > 0:
            class_names = results[0].names
            for box in detections:
                class_id = int(box.cls[0])
                detected_classes.append(class_names[class_id])

        print(f"  ✓ Processed: {Path(image_path).name}")
        print(f"    - Objects detected: {num_objects}")
        if num_objects > 0:
            print(f"    - Classes: {', '.join(set(detected_classes))}")
        print(f"    - Saved to: {output_filename}")

        return True

    except Exception as e:
        print(f"  ✗ Error processing {image_path}: {str(e)}")
        return False


def main():
    """Main function to run object detection on multiple images"""

    print("=" * 60)
    print("YOLO Object Detection - Week 15")
    print("=" * 60)

    # Configuration
    input_folder = "input_images"
    output_folder = "output_images"
    model_name = "yolov8n.pt"  # Nano model
    confidence_threshold = 0.25

    # Allow command-line arguments (optional)
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]

    print(f"\nConfiguration:")
    print(f"  - Model: {model_name}")
    print(f"  - Input folder: {input_folder}")
    print(f"  - Output folder: {output_folder}")
    print(f"  - Confidence threshold: {confidence_threshold}")
    print()

    # Create output directory
    create_output_directory(output_folder)

    # Get image files
    print(f"\nScanning for images in '{input_folder}'...")
    image_files = get_image_files(input_folder)

    if not image_files:
        print(f"✗ No images found in '{input_folder}'")
        print(f"  Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        return

    print(f"✓ Found {len(image_files)} image(s)")
    print()

    # Load YOLO model
    print(f"Loading YOLO model ({model_name})...")
    try:
        model = YOLO(model_name)
        print(f"✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return

    # Process each image
    print("Processing images...")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        if detect_objects(model, image_path, output_folder, confidence_threshold):
            successful += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  - Total images: {len(image_files)}")
    print(f"  - Successfully processed: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Results saved in: {output_folder}")
    print("=" * 60)
    print("\n✓ Detection completed!")


if __name__ == "__main__":
    main()