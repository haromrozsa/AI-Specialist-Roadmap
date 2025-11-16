import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import os


# Load and preprocess CIFAR-10 dataset
def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


# Visualize sample images from the dataset
def visualize_dataset_samples(train_images, train_labels, num_samples=25):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels are one-hot encoded vectors,
        # so we need to convert back to the label index
        label_idx = np.argmax(train_labels[i])
        plt.xlabel(class_names[label_idx])
    plt.tight_layout()
    plt.show()


# Build CNN model for CIFAR-10 classification
def build_cnn_model():
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            input_shape=(32, 32, 3), name='conv2d_1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_4'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_5'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_6'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels, epochs=20, batch_size=64):
    # Create callbacks for model training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]

    # Train the model
    history = model.fit(
        train_images, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    return history


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

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


# Visualize model predictions
def visualize_predictions(model, test_images, test_labels, num_samples=15):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Get model predictions
    predictions = model.predict(test_images[:num_samples])
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels[:num_samples], axis=1)

    # Plot images with predictions
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(3, 5, i + 1)
        plt.imshow(test_images[i])
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        plt.title(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}",
                  color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Visualize convolutional layers feature maps
def visualize_feature_maps(model, test_images, layer_name, image_index=0):
    # Create a model that outputs feature maps for the given layer
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    if not layer_outputs:
        print(f"Layer {layer_name} not found in the model.")
        return

    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    # Get feature maps for a sample image
    img = np.expand_dims(test_images[image_index], axis=0)
    activations = activation_model.predict(img)

    # Plot the feature maps
    feature_maps = activations[0]
    n_features = min(16, feature_maps.shape[-1])  # Display up to 16 feature maps
    size = int(np.ceil(np.sqrt(n_features)))

    plt.figure(figsize=(12, 12))
    for i in range(n_features):
        plt.subplot(size, size, i + 1)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# Visualize filters from convolutional layers
def visualize_filters(model, layer_name, max_filters=16):
    # Get the layer by name
    layer = None
    for l in model.layers:
        if l.name == layer_name:
            layer = l
            break

    if layer is None:
        print(f"Layer {layer_name} not found in the model.")
        return

    # Get filters
    filters, biases = layer.get_weights()

    # Normalize filter values for better visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)

    # Number of filters to display
    n_filters = min(max_filters, filters.shape[3])
    size = int(np.ceil(np.sqrt(n_filters)))

    # Create figure with subplots
    plt.figure(figsize=(12, 12))
    for i in range(n_filters):
        plt.subplot(size, size, i + 1)

        # For RGB filters (3 channels)
        filter_img = filters[:, :, :, i]

        # If filter has depth > 1, take average across channels
        if filter_img.shape[2] > 1:
            plt.imshow(np.mean(filter_img, axis=2), cmap='viridis')
        else:
            plt.imshow(filter_img[:, :, 0], cmap='viridis')

        plt.axis('off')

    plt.suptitle(f'Filters from {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# Save the model
def save_model(model, filename='cifar10_cnn_model.h5'):
    model.save(filename)
    print(f"Model saved as {filename}")


# Main function to run the entire process
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing CIFAR-10 dataset...")
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # Visualize dataset samples
    print("Visualizing sample images from the dataset...")
    visualize_dataset_samples(train_images, train_labels)

    # Build the model
    print("Building CNN model...")
    model = build_cnn_model()
    model.summary()

    # Train the model
    print("Training the model...")
    history = train_model(model, train_images, train_labels, test_images, test_labels)

    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)

    # Visualize predictions
    print("Visualizing model predictions...")
    visualize_predictions(model, test_images, test_labels)

    # Visualize feature maps from different convolutional layers
    print("Visualizing feature maps...")
    # Choose a sample image
    sample_idx = 7
    # Visualize feature maps from different layers
    visualize_feature_maps(model, test_images, 'conv2d_1', sample_idx)
    visualize_feature_maps(model, test_images, 'conv2d_3', sample_idx)
    visualize_feature_maps(model, test_images, 'conv2d_5', sample_idx)

    # Visualize filters from convolutional layers
    print("Visualizing filters...")
    visualize_filters(model, 'conv2d_1')
    visualize_filters(model, 'conv2d_3')
    visualize_filters(model, 'conv2d_5')

    # Save the model
    save_model(model)

    print("CNN CIFAR-10 Classification complete!")


if __name__ == "__main__":
    main()
