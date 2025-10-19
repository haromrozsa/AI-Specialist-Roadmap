from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize the Perceptron model
clf = Perceptron(
    tol=1e-3,  # Tolerance for stopping criterion
    random_state=42,  # For reproducibility
    max_iter=1000,  # Maximum number of passes over the training data
    eta0=1.0  # Learning rate
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Access model parameters
print(f"Weights: {clf.coef_}")
print(f"Bias: {clf.intercept_}")
