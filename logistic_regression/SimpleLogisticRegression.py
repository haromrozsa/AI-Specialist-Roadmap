# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset for binary classification
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='Class 0', alpha=0.5)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='Class 1', alpha=0.5)
plt.title('Binary Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Create a logistic regression model
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Print model coefficients
print("Model coefficients:")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Visualize the decision boundary
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Function to plot the decision boundary of a classifier
    """
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Step size in the mesh
    h = 0.01

    # Generate a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict class for each point in mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and data points
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', label='Class 0', edgecolors='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='x', label='Class 1', edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot decision boundary for training data
plot_decision_boundary(X_train, y_train, model, "Training Data with Decision Boundary")

# Plot decision boundary for test data
plot_decision_boundary(X_test, y_test, model, "Test Data with Decision Boundary")