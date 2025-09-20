import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SimpleLogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False):
        """
        Initialize the logistic regression model.

        Parameters:
        -----------
        learning_rate : float, default=0.01
            The step size for gradient descent
        num_iterations : int, default=1000
            Number of iterations for gradient descent
        fit_intercept : bool, default=True
            Whether to add a bias/intercept term
        verbose : bool, default=False
            Whether to print training progress
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.weights = None
        self.cost_history = []

    def sigmoid(self, z):
        """
        Compute the sigmoid function.

        Parameters:
        -----------
        z : array-like
            Input values

        Returns:
        --------
        array-like
            Sigmoid of the inputs
        """
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)

        Returns:
        --------
        self : object
            Returns self
        """
        # Add intercept if specified
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        # Gradient descent
        m = X.shape[0]  # number of samples

        for iteration in range(self.num_iterations):
            # Calculate predictions
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)

            # Calculate error
            error = predictions - y

            # Calculate gradient
            gradient = np.dot(X.T, error) / m

            # Update weights
            self.weights -= self.learning_rate * gradient

            # Calculate cost
            cost = -1 / m * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
            self.cost_history.append(cost)

            # Print progress if verbose is True
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        array-like, shape (n_samples,)
            Class 1 probabilities
        """
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        threshold : float, default=0.5
            Threshold for positive class

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate the accuracy of the model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True class labels

        Returns:
        --------
        float
            Accuracy of the model
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Generate synthetic data for classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='Class 0', alpha=0.5)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='Class 1', alpha=0.5)
plt.title('Synthetic Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Train the logistic regression model
model = SimpleLogisticRegression(learning_rate=0.1, num_iterations=2000, verbose=True)
model.fit(X_train, y_train)

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(model.cost_history)
plt.title('Cost History During Training')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
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