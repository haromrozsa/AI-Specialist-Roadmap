import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize images
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert labels to categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(f"Training data shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")


def build_simple_cnn():
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


# Create model
model = build_simple_cnn()

# Display some examples from the dataset
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(train_labels[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Create checkpoint to save the best model
checkpoint = ModelCheckpoint('best_mnist_model.h5',
                            monitor='val_accuracy',
                            save_best_only=True,
                            mode='max')

# Train the model
history = model.fit(train_images, train_labels,
                   epochs=10,
                   batch_size=64,
                   validation_split=0.2,
                   callbacks=[checkpoint])

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Load best saved model
best_model = models.load_model('best_mnist_model.h5')

# Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Make predictions
predictions = best_model.predict(test_images)

# Display some examples with predictions
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')

    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])

    color = 'green' if predicted_label == true_label else 'red'

    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Define a model that outputs feature maps after each convolution
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Choose an image to visualize
img = test_images[0:1]
activations = activation_model.predict(img)

# Display the original image
plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1)
plt.imshow(img[0].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display feature maps from each convolutional layer
for i, layer_activation in enumerate(activations):
    n_features = layer_activation.shape[-1]

    # We'll visualize the first 8 features or all if less than 8
    n_to_display = min(8, n_features)

    plt.subplot(1, 4, i + 2)

    # Choose a representative feature map
    feature_map = layer_activation[0, :, :, 0]

    plt.imshow(feature_map, cmap='viridis')
    plt.title(f'Conv Layer {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get predictions for the test set
y_pred = best_model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate per-class accuracy
class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
for i, acc in enumerate(class_accuracy):
    print(f'Accuracy for digit {i}: {acc:.4f}')


    def build_deeper_cnn():
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classification block
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()
        return model

    # You can uncomment and run this to experiment with a deeper model:
    # deeper_model = build_deeper_cnn()
    # deeper_history = deeper_model.fit(train_images, train_labels,
    #                                   epochs=15,
    #                                   batch_size=64,
    #                                   validation_split=0.2)

    # Further experiments:
    # 1. Try different optimizers: SGD, RMSProp, etc.
    # 2. Add data augmentation
    # 3. Adjust learning rate
    # 4. Implement early stopping