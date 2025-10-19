import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow logging to avoid the metric registration error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Clear any previous TensorFlow session
tf.keras.backend.clear_session()

# Load the MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data preprocessing
print("Preprocessing data...")
# Flatten the images
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

# Convert to float and normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Print dataset shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Build the Sequential model - Feedforward Neural Network
print("Building model...")
model = Sequential([
    # Input layer with 784 nodes (28*28 pixels)
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    # Output layer with 10 nodes (for digits 0-9)
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, name="adam_unique"),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train the model
print("Training model...")
batch_size = 128
epochs = 15

start_time = time.time()

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test)
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate the model
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()


# Function to display example predictions
def plot_example_predictions(X_data, y_true, num_examples=10):
    # Get predictions
    y_pred_probs = model.predict(X_data[:num_examples])
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_true[:num_examples], axis=1)

    # Plot
    plt.figure(figsize=(15, 8))
    for i in range(num_examples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_data[i].reshape(28, 28), cmap='gray')
        title_color = 'green' if y_pred_classes[i] == y_true_classes[i] else 'red'
        plt.title(f"Pred: {y_pred_classes[i]}\nTrue: {y_true_classes[i]}",
                  color=title_color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Show some example predictions
print("Displaying example predictions...")
plot_example_predictions(X_test, y_test)

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Add text annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()


# Function to display misclassified examples
def plot_misclassified_examples(X_data, y_true, num_examples=10):
    # Get predictions
    X_reshape = X_data.reshape(X_data.shape[0], 784)  # Ensure proper shape
    y_pred_probs = model.predict(X_reshape)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Find misclassified examples
    misclassified = np.where(y_pred_classes != y_true_classes)[0]

    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return

    # Take a subset of misclassified examples
    num_examples = min(num_examples, len(misclassified))
    indices = misclassified[:num_examples]

    # Plot
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_data[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {y_pred_classes[idx]}\nTrue: {y_true_classes[idx]}",
                  color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Show some misclassified examples
print("Displaying misclassified examples...")
X_test_2d = X_test.reshape(X_test.shape[0], 28, 28)  # Reshape for visualization
plot_misclassified_examples(X_test_2d, y_test)

print("Done!")
