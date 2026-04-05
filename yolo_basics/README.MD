# YOLO Basics (Object Detection)

## Session Summary: Implemented a YOLOv8-based object detection script that processes multiple images, detects objects, and saves annotated results with bounding boxes and labels.

## What I Did During the Session

1. **Planned the Implementation Strategy**:
    - Evaluated three approaches: Ultralytics YOLOv8, OpenCV DNN module, and PyTorch YOLOv5
    - Chose Ultralytics YOLOv8 for its simplicity, modern API, and excellent documentation
    - Selected YOLOv8n (nano model) for lightweight, fast experimentation
    - Designed folder structure with separate input/output directories

2. **Implemented Batch Image Processing ()`detect.py`**:
    - Created `get_image_files()` to scan directories for images with multiple format support (.jpg, .png, .bmp, .tiff, .webp)
    - Built `create_output_directory()` to automatically create output folders if they don't exist
    - Added path validation to check if input directories exist before processing
    - Sorted image files for consistent processing order

3. **Built Object Detection Pipeline**:
    - Loaded pre-trained YOLOv8n model using `YOLO(model_name)` from Ultralytics
    - Created `detect_objects()` function to handle single image detection
    - Used `model(image, conf=confidence_threshold)` for inference with configurable confidence (default 0.25)
    - Generated annotated images with bounding boxes using `results[0].plot()`
    - Saved detected images with descriptive filenames (e.g., `image_detected.jpg`)

4. **Added Detection Statistics and Reporting**:
    - Extracted detection details from `results[0].boxes` to count detected objects
    - Retrieved class names from the model and displayed unique detected classes per image
    - Printed per-image statistics: filename, object count, and detected class names
    - Created summary report showing total images, successful/failed processing counts

5. **Implemented Command-Line Interface**:
    - Added `sys.argv` parsing to accept custom input/output folder paths
    - Configured default folders (`input_images`, `output_images`) for basic usage
    - Displayed configuration details (model, folders, confidence threshold) before processing
    - Made the script flexible for different use cases without code modification

6. **Added Error Handling and User Feedback**:
    - Wrapped image reading in try-except blocks to handle corrupted/invalid files
    - Validated image loading with `cv2.imread()` null checks
    - Caught model loading exceptions and displayed clear error messages
    - Provided visual feedback with checkmarks (✓) and crosses (✗) for operation status
    - Tracked successful and failed image processing counts

## What I Learned
1. **YOLO Object Detection Fundamentals**:
    - YOLO (You Only Look Once) processes the entire image in a single forward pass for real-time detection
    - YOLOv8 comes in different sizes (nano, small, medium, large, extra-large) with speed-accuracy tradeoffs
    - The nano model (`yolov8n.pt`) is ~6MB and suitable for learning and experimentation
    - YOLO models trained on COCO dataset can detect 80 different object classes
    - Confidence threshold filters detections—0.25 balances finding objects vs. avoiding false positives

2. **Ultralytics YOLOv8 API**:
    - `YOLO(model_name)` automatically downloads pre-trained weights on first run
    - Inference is as simple as calling `model(image, conf=threshold)` with OpenCV-loaded images
    - Results contain `boxes` (detections), `names` (class labels), and visualization methods
    - `results[0].plot()` automatically draws annotated bounding boxes with labels and confidence scores
    - Setting `verbose=False` suppresses console output during inference for cleaner logs

3. **Working with Pre-trained Models**:
    - Pre-trained models eliminate the need for dataset collection and training
    - Models can be used immediately after installation for practical applications
    - The first run downloads model weights from the internet automatically
    - Pre-trained weights encode knowledge from large datasets (COCO has 330K images)
    - Transfer learning allows starting with pre-trained models and fine-tuning for specific tasks

4. **Batch Image Processing Patterns**:
    - Use `Path(file).suffix` to filter files by extension reliably across platforms
    - Process images in a loop with indexed progress indicators (`[1/5]`, `[2/5]`, etc.)
    - Separate concerns: directory scanning, single-image processing, and batch orchestration
    - Track success/failure counts for meaningful summary reports
    - Generate output filenames programmatically using input filename stems

5. **Computer Vision with OpenCV**:
    - `cv2.imread()` loads images as NumPy arrays in BGR format
    - Always check if `cv2.imread()` returns `None` to detect invalid/corrupted files
    - `cv2.imwrite()` saves images with automatic format detection from file extension
    - OpenCV integrates seamlessly with deep learning frameworks like YOLO
    - Image preprocessing (resizing, normalization) is often handled internally by modern frameworks

6. **Script Design for CLI Tools**:
    - Provide sensible defaults while allowing customization via command-line arguments
    - Display configuration details before execution for transparency
    - Use clear visual separators (lines, sections) to organize console output
    - Print progress indicators for long-running tasks to reassure users
    - Include summary statistics to give context about overall execution

This session provided hands-on experience with modern object detection using YOLO, working with pre-trained models, batch processing workflows, and building user-friendly command-line tools for computer vision tasks.