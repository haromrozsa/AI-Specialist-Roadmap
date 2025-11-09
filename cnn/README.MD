# ML Fundamentals – CNN Intro

# Session Summary: CNN Architecture and MNIST Classification

## What I Did During the Session

1. **Learned About CNN Architecture**
   - Discussed the concept of filters in convolutional layers
   - Understood how multiple filters create feature maps
   - Explored how filters detect different patterns in input data
   - Clarified the relationship between filter count and output channels

2. **Implemented a Simple CNN for MNIST Classification**
   - Created a basic CNN architecture with multiple convolutional layers
   - Built a model with the following structure:
     - First convolutional layer: 32 filters (3×3), followed by max pooling
     - Second convolutional layer: 64 filters (3×3), followed by max pooling
     - Third convolutional layer: 64 filters (3×3)
     - Flattening layer followed by dense layers (64 neurons and 10 output classes)

3. **Prepared the MNIST Dataset**
   - Loaded the MNIST handwritten digit dataset
   - Preprocessed the images (reshaped and normalized)
   - Converted labels to categorical format for multi-class classification

4. **Trained and Evaluated the CNN Model**
   - Implemented model training with appropriate hyperparameters
   - Used callbacks to save the best model during training
   - Evaluated model performance on the test set
   - Visualized training history (accuracy and loss curves)

5. **Implemented Visualization Techniques**
   - Created visualizations of sample MNIST images
   - Generated and analyzed feature maps from convolutional layers
   - Created a confusion matrix to evaluate per-class performance
   - Displayed examples with predictions to understand model behavior

6. **Explored Advanced CNN Architecture**
   - Defined a deeper CNN with additional layers for experimentation
   - Implemented architectural improvements like batch normalization and dropout
   - Considered various optimization strategies for better performance

## What I Learned

1. **CNN Architecture Fundamentals**:
   - Filters in Conv2D layers specify the number of output feature maps
   - Each filter learns to detect specific patterns in the input data
   - Filter depth matches input channels automatically
   - Filters enable hierarchical feature learning in CNNs

2. **Convolutional Layer Mechanics**:
   - Multiple filters create multiple feature maps in each layer
   - Early layers detect simple features (edges, textures)
   - Deeper layers combine simple features into complex patterns
   - Filter weights are learned during model training

3. **Model Building Techniques**:
   - Proper CNN architecture design for image classification
   - Importance of filter count selection at each layer
   - Effective use of pooling layers to reduce dimensionality
   - Transition from convolutional to dense layers via flattening

4. **Practical Implementation Skills**:
   - Loading and preprocessing image data for CNNs
   - Configuring and training CNN models effectively
   - Using callbacks for model checkpointing
   - Evaluating model performance on image classification tasks

5. **Visualization and Analysis**:
   - Techniques for visualizing CNN feature maps
   - Creation and interpretation of confusion matrices
   - Analysis of per-class accuracy
   - Visualization of model predictions on test examples

6. **Advanced Concepts**:
   - Architectural enhancements like batch normalization and dropout
   - Deeper network design considerations
   - Strategies for improving CNN performance

This session provided a comprehensive understanding of CNN architecture, particularly focusing on filters in convolutional layers and their role in feature extraction. I implemented and analyzed a practical CNN model for the classic MNIST digit classification task and explored advanced techniques for improving CNN performance.