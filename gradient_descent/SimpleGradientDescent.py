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


# Example: Minimize f(x) = x^2 + 2y^2
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


# Run gradient descent
initial_point = np.array([5.0, 5.0])  # Starting far from minimum at (0,0)
learning_rate = 0.1
optimal_x, trajectory = gradient_descent(gradient_function, initial_point, learning_rate, max_iterations=50)

# Print results
print(f"Starting point: {initial_point}")
print(f"Optimal point found: {optimal_x}")
print(f"Objective function value at optimal point: {objective_function(optimal_x)}")

# Plot the results
# plot_contour_and_trajectory(trajectory)

# Example: Using the gradient descent to find the minimum
if __name__ == "__main__":
    print("Running simple gradient descent example...")

    # Try different learning rates
    for lr in [0.01, 0.1, 0.4]:
        print(f"\nTesting learning rate: {lr}")
        optimal_x, trajectory = gradient_descent(gradient_function, initial_point, learning_rate=lr)
        plot_contour_and_trajectory(trajectory)
        print(f"Iterations required: {len(trajectory)}")
        print(f"Final point: {optimal_x}")
