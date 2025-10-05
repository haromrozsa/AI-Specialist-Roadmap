import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Keep all existing functions and classes as they are

# Add this new section for SGDRegressor implementation
def sgd_regressor_example():
    """Demonstrate gradient descent using scikit-learn's SGDRegressor"""
    print("\nSGDRegressor Example:")

    # Generate synthetic data (similar to our custom implementation)
    n_samples = 100
    np.random.seed(42)  # For reproducibility

    # True parameters: intercept=2.5, slope=3.7
    true_intercept = 2.5
    true_slope = 3.7

    # Generate random x values
    x = np.random.rand(n_samples, 1) * 10

    # Generate y with some noise
    y = true_intercept + true_slope * x.flatten() + 1.0 * np.random.randn(n_samples)

    # Split the data into training and test sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"Training data: {x_train.shape[0]} samples")
    print(f"Test data: {x_test.shape[0]} samples")

    # Create a pipeline with standardization and SGDRegressor
    # Using 'squared_loss' instead of 'squared_error' for compatibility
    sgd_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDRegressor(loss='squared_loss', penalty=None, learning_rate='constant',
                             eta0=0.01, max_iter=1000, tol=1e-3, random_state=42))
    ])

    # Train the model and keep track of the loss at each epoch
    train_losses = []
    test_losses = []
    # Using 'squared_loss' instead of 'squared_error'
    sgd = SGDRegressor(loss='squared_loss', penalty=None, learning_rate='constant',
                       eta0=0.01, max_iter=1, tol=None, random_state=42,
                       warm_start=True)  # warm_start allows us to continue training

    # Fit with warm_start to track loss per epoch
    for i in range(200):  # 200 epochs
        sgd.fit(x_train, y_train)

        # Calculate training loss
        y_train_pred = sgd.predict(x_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_losses.append(train_loss)

        # Calculate test loss
        y_test_pred = sgd.predict(x_test)
        test_loss = mean_squared_error(y_test, y_test_pred)
        test_losses.append(test_loss)

    # Get the final parameters
    final_slope = sgd.coef_[0]
    final_intercept = sgd.intercept_[0]

    print(f"True parameters: intercept={true_intercept}, slope={true_slope}")
    print(f"SGDRegressor learned parameters: intercept={final_intercept:.4f}, slope={final_slope:.4f}")
    print(f"Final Training MSE: {train_losses[-1]:.4f}")
    print(f"Final Test MSE: {test_losses[-1]:.4f}")

    # Plot the results
    plt.figure(figsize=(15, 12))  # Increased figure height for three plots

    # Plot 1: Data and fitted line
    plt.subplot(2, 2, 1)
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    plt.scatter(x_test, y_test, alpha=0.6, color='green', label='Test Data')

    # Sort x for line plotting to ensure proper line rendering
    x_sorted = np.sort(np.vstack([x_train, x_test]), axis=0)
    plt.plot(x_sorted, sgd.predict(x_sorted), 'r-',
             label=f'Fitted Line: y = {final_intercept:.2f} + {final_slope:.2f}x')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('SGDRegressor Result')
    plt.legend()
    plt.grid(True)

    # Plot 2: Loss history
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Test Loss History')
    plt.legend()
    plt.grid(True)

    # Plot 3: Learning curve - Loss vs Dataset Size
    plt.subplot(2, 1, 2)  # Use the entire bottom row for the learning curve

    # Define dataset sizes to evaluate (from 2 samples up to full training set)
    train_sizes = np.linspace(2, len(x_train), 20, dtype=int)
    train_size_losses = []
    test_size_losses = []

    # Initialize a new SGD model for this experiment
    size_sgd = SGDRegressor(loss='squared_loss', penalty=None, learning_rate='constant',
                            eta0=0.01, max_iter=200, tol=1e-3, random_state=42)

    # For each training size, train the model and evaluate
    for size in train_sizes:
        # Take the first 'size' samples from the training set
        X_subset = x_train[:size]
        y_subset = y_train[:size]

        # Fit the model
        size_sgd.fit(X_subset, y_subset)

        # Calculate training and test errors
        train_pred = size_sgd.predict(X_subset)
        test_pred = size_sgd.predict(x_test)

        train_error = mean_squared_error(y_subset, train_pred)
        test_error = mean_squared_error(y_test, test_pred)

        train_size_losses.append(train_error)
        test_size_losses.append(test_error)

    # Plot the learning curve
    plt.plot(train_sizes, train_size_losses, 'o-', label='Training Loss')
    plt.plot(train_sizes, test_size_losses, 'o-', label='Test Loss')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve: Loss vs Training Set Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Also fit our pipeline version (with scaling) for comparison
    sgd_reg.fit(x_train, y_train)
    pipeline_slope = sgd_reg.named_steps['sgd'].coef_[0]
    pipeline_intercept = sgd_reg.named_steps['sgd'].intercept_[0]

    print("\nWith preprocessing (scaling):")
    print(f"Pipeline SGDRegressor parameters: intercept={pipeline_intercept:.4f}, slope={pipeline_slope:.4f}")
    y_train_pred_pipeline = sgd_reg.predict(x_train)
    y_test_pred_pipeline = sgd_reg.predict(x_test)
    pipeline_train_mse = mean_squared_error(y_train, y_train_pred_pipeline)
    pipeline_test_mse = mean_squared_error(y_test, y_test_pred_pipeline)
    print(f"Pipeline Training MSE: {pipeline_train_mse:.4f}")
    print(f"Pipeline Test MSE: {pipeline_test_mse:.4f}")

    # Compare with a single-pass standard linear regression for reference
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("\nFor comparison - Closed-form solution (LinearRegression):")
    print(f"LinearRegression parameters: intercept={lr.intercept_:.4f}, slope={lr.coef_[0]:.4f}")
    lr_train_mse = mean_squared_error(y_train, lr.predict(x_train))
    lr_test_mse = mean_squared_error(y_test, lr.predict(x_test))
    print(f"LinearRegression Training MSE: {lr_train_mse:.4f}")
    print(f"LinearRegression Test MSE: {lr_test_mse:.4f}")


# Add this line to the main section at the end
if __name__ == "__main__":
    print("Running simple gradient descent example...")

    # Then run the SGDRegressor example
    sgd_regressor_example()
