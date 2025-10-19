import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

# Load the MNIST dataset
print("Loading MNIST dataset...")
# Remove the parser parameter
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.astype('float32')
y = y.astype('int')

# Convert to numpy array if needed
if hasattr(X, 'values'):
    X = X.values
if hasattr(y, 'values'):
    y = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the MLPClassifier
print("Training MLP Classifier...")
start_time = time.time()

# Define the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),  # Two hidden layers with 100 neurons each
    activation='relu',  # ReLU activation function
    solver='adam',  # Adam optimizer
    alpha=0.0001,  # L2 penalty (regularization term)
    batch_size=256,  # Size of minibatches for stochastic optimizers
    learning_rate='adaptive',  # Learning rate schedule for weight updates
    max_iter=30,  # Maximum number of iterations
    verbose=True,  # Print progress messages to stdout
    random_state=42  # Random state for reproducibility
)

# Train the model
mlp.fit(X_train, y_train)

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Plot some example predictions
def plot_example_predictions(X_test, y_test, y_pred, indices=None):
    if indices is None:
        # Choose random samples
        indices = np.random.choice(len(X_test), 9, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot some examples
print("Displaying example predictions...")
plot_example_predictions(X_test, y_test, y_pred)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

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
