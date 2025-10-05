import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(gradient_func, initial_guess, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
    """
    Performs gradient descent optimization on a function.

    Args:
        gradient_func: Function that computes the gradient of the objective function
        initial_guess: Starting point for optimization
        learning_rate: Step size for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for the norm of the gradient

    Returns:
        x: The optimized parameters
        trajectory: List of points visited during optimization
    """
    x = initial_guess
    trajectory = [x.copy()]

    for i in range(max_iterations):
        # Compute gradient
        grad = gradient_func(x)

        # Check for convergence
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i} iterations!")
            break

        # Update parameters
        x = x - learning_rate * grad

        # Store current position
        trajectory.append(x.copy())

    return x, trajectory


# Example 1: Minimize f(x) = x^2 + 2y^2
def objective_function(x):
    """Simple quadratic function: f(x) = x^2 + 2y^2"""
    return x[0] ** 2 + 2 * x[1] ** 2


def gradient_function(x):
    """Gradient of the objective function: ∇f(x) = [2x, 4y]"""
    return np.array([2 * x[0], 4 * x[1]])


# Visualization
def plot_contour_and_trajectory(trajectory):
    # Create a grid of points
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + 2 * Y ** 2  # Our objective function

    # Convert trajectory to arrays for plotting
    trajectory_array = np.array(trajectory)

    # Create contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar(contour, label='f(x, y)')

    # Plot trajectory
    plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'bo-', linewidth=1, markersize=4,
             label='Gradient Descent Path')
    plt.plot(trajectory_array[0, 0], trajectory_array[0, 1], 'go', markersize=8, label='Initial Point')
    plt.plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 'ro', markersize=8, label='Final Point')

    plt.title('Gradient Descent Optimization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Example 2: Linear Regression with Mean Squared Error
class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.params = None
        self.trajectory = None

    def generate_sample_data(self, n_samples=100, noise=0.5):
        """Generate synthetic data for linear regression"""
        # True parameters: intercept=2.5, slope=3.7
        true_params = np.array([2.5, 3.7])

        # Generate random x values
        x = np.random.rand(n_samples, 1) * 10

        # Add a column of 1s for the intercept
        X = np.hstack((np.ones((n_samples, 1)), x))

        # Generate y with some noise
        y = X.dot(true_params) + noise * np.random.randn(n_samples)

        self.X = X
        self.y = y
        return X, y

    def mse_loss(self, params):
        """Mean Squared Error loss function"""
        predictions = self.X.dot(params)
        return np.mean((predictions - self.y) ** 2)

    def mse_gradient(self, params):
        """Gradient of the MSE loss function"""
        predictions = self.X.dot(params)
        errors = predictions - self.y
        gradient = 2 * self.X.T.dot(errors) / len(self.y)
        return gradient

    def fit(self, learning_rate=0.01, max_iterations=1000):
        """Fit the linear regression model using gradient descent"""
        # Initialize parameters (intercept and slope)
        initial_params = np.zeros(self.X.shape[1])

        # Run gradient descent
        self.params, self.trajectory = gradient_descent(
            lambda params: self.mse_gradient(params),
            initial_params,
            learning_rate=learning_rate,
            max_iterations=max_iterations
        )

        return self.params

    def predict(self, X_new):
        """Make predictions with the trained model"""
        # Add intercept term if X_new doesn't have it
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_new.shape[1] == 1:
            X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))

        return X_new.dot(self.params)

    def plot_regression_results(self):
        """Plot the data, fitted line and loss history"""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot the data and fitted line
        x_values = self.X[:, 1]
        y_pred = self.predict(self.X)

        ax1.scatter(x_values, self.y, alpha=0.6, label='Data')
        ax1.plot(x_values, y_pred, 'r-', label=f'Fitted Line: y = {self.params[0]:.2f} + {self.params[1]:.2f}x')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Linear Regression Result')
        ax1.legend()
        ax1.grid(True)

        # Plot the loss history
        loss_history = [self.mse_loss(params) for params in self.trajectory]
        ax2.plot(loss_history)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Loss History')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot the trajectory in parameter space
        trajectory_array = np.array(self.trajectory)
        plt.figure(figsize=(8, 6))
        plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'bo-', linewidth=1, markersize=4)
        plt.plot(trajectory_array[0, 0], trajectory_array[0, 1], 'go', markersize=8, label='Initial Parameters')
        plt.plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 'ro', markersize=8, label='Final Parameters')
        plt.xlabel('Intercept')
        plt.ylabel('Slope')
        plt.title('Parameter Trajectory During Training')
        plt.legend()
        plt.grid(True)
        plt.show()


# Run gradient descent on quadratic function
initial_point = np.array([5.0, 5.0])  # Starting far from minimum at (0,0)
learning_rate = 0.1
optimal_x, trajectory = gradient_descent(gradient_function, initial_point, learning_rate, max_iterations=50)

# Print results
print(f"Quadratic Function Example:")
print(f"Starting point: {initial_point}")
print(f"Optimal point found: {optimal_x}")
print(f"Objective function value at optimal point: {objective_function(optimal_x)}")

# Plot the results
plot_contour_and_trajectory(trajectory)

# Example: Linear Regression using Gradient Descent
print("\nLinear Regression Example:")
# Create and fit the linear regression model
lr_model = LinearRegression()
X, y = lr_model.generate_sample_data(n_samples=100, noise=1.0)
params = lr_model.fit(learning_rate=0.01, max_iterations=200)

print(f"True parameters: intercept=2.5, slope=3.7")
print(f"Learned parameters: intercept={params[0]:.4f}, slope={params[1]:.4f}")
print(f"Final MSE: {lr_model.mse_loss(params):.4f}")

# Plot the regression results
lr_model.plot_regression_results()

# Example: Using the gradient descent to find the minimum
if __name__ == "__main__":
    print("Running simple gradient descent example...")

    # Try different learning rates
    for lr in [0.01, 0.1, 0.5]:
        print(f"\nTesting learning rate: {lr}")
        optimal_x, trajectory = gradient_descent(gradient_function, initial_point, learning_rate=lr)
        print(f"Iterations required: {len(trajectory)}")
        print(f"Final point: {optimal_x}")
