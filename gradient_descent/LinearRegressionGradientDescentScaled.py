import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def gradient_descent(objective_func, gradient_func, initial_point, learning_rate=0.1, n_iterations=100, tolerance=1e-6):
    """
    Perform gradient descent optimization.

    Parameters:
    -----------
    objective_func : callable
        Function to minimize.
    gradient_func : callable
        Function that computes the gradient of the objective.
    initial_point : array-like
        Starting point for optimization.
    learning_rate : float, default=0.1
        Step size for parameter updates.
    n_iterations : int, default=100
        Maximum number of iterations.
    tolerance : float, default=1e-6
        Convergence threshold.

    Returns:
    --------
    optimal_x : ndarray
        Optimal parameters found.
    trajectory : list of ndarrays
        History of parameter values.
    """
    # Initialize at the starting point
    current_x = np.array(initial_point, dtype=float)
    trajectory = [current_x.copy()]

    # Perform gradient descent
    for i in range(n_iterations):
        # Compute gradient
        gradient = gradient_func(current_x)

        # Update parameters
        new_x = current_x - learning_rate * gradient

        # Store new parameters
        trajectory.append(new_x.copy())

        # Check for convergence
        if np.linalg.norm(new_x - current_x) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            break

        # Update current parameters
        current_x = new_x

    return current_x, trajectory


def objective_function(x):
    """
    Example objective function (e.g., MSE for a specific dataset).

    Parameters:
    -----------
    x : array-like
        Parameters.

    Returns:
    --------
    obj_value : float
        Objective function value.
    """
    return (x[0] ** 2 + x[1] ** 2)  # Simple quadratic function for illustration


def gradient_function(x):
    """
    Gradient of the example objective function.

    Parameters:
    -----------
    x : array-like
        Parameters.

    Returns:
    --------
    gradient : ndarray
        Gradient of the objective at x.
    """
    return np.array([2 * x[0], 2 * x[1]])  # Gradient of the quadratic function


def plot_contour_and_trajectory(objective_func, trajectory, title="Gradient Descent Optimization"):
    """
    Plot the contour of the objective function and the optimization trajectory.

    Parameters:
    -----------
    objective_func : callable
        Objective function to visualize.
    trajectory : list of ndarrays
        History of parameter values.
    title : str
        Plot title.
    """
    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)

    # Create a grid of points
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute objective function values on the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func([X[i, j], Y[i, j]])

    # Create subplots: 2D contour and 3D surface
    fig = plt.figure(figsize=(15, 6))

    # 2D contour plot with trajectory
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=1, markersize=8)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    ax1.set_title('Contour Plot with Optimization Path')
    ax1.set_xlabel('Parameter 1')
    ax1.set_ylabel('Parameter 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax1)

    # 3D surface plot with trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, rstride=5, cstride=5)
    ax2.plot(trajectory[:, 0], trajectory[:, 1],
             [objective_func([p[0], p[1]]) for p in trajectory],
             'r.-', linewidth=2, markersize=8)
    ax2.set_title('3D Surface Plot with Optimization Path')
    ax2.set_xlabel('Parameter 1')
    ax2.set_ylabel('Parameter 2')
    ax2.set_zlabel('Objective Function')
    plt.colorbar(surface, ax=ax2, shrink=0.7)

    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


class LinearRegression:
    """Linear Regression implementation using Gradient Descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, use_scaling=False):
        """
        Initialize Linear Regression model.

        Parameters:
        -----------
        learning_rate : float, default=0.01
            The learning rate for gradient descent.
        n_iterations : int, default=1000
            Number of iterations for gradient descent.
        use_scaling : bool, default=False
            Whether to use feature scaling.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.use_scaling = use_scaling
        self.X = None
        self.y = None
        self.params = None
        self.trajectory = None
        self.scaler = StandardScaler() if use_scaling else None

    def generate_sample_data(self, n_samples=100, n_features=1, noise=0.1, seed=42):
        """
        Generate sample data for regression.

        Parameters:
        -----------
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=1
            Number of features to generate.
        noise : float, default=0.1
            Standard deviation of the Gaussian noise.
        seed : int, default=42
            Random seed for reproducibility.

        Returns:
        --------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.
        """
        np.random.seed(seed)

        # Generate random feature matrix
        X = np.random.randn(n_samples, n_features)

        # Generate random coefficients
        true_coefficients = np.random.randn(n_features)

        # Generate target with noise
        y = X.dot(true_coefficients) + noise * np.random.randn(n_samples)

        return X, y

    def mse_loss(self, y_true, y_pred):
        """
        Calculate Mean Squared Error loss.

        Parameters:
        -----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.

        Returns:
        --------
        mse : float
            Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def mse_gradient(self, X, y_true, y_pred):
        """
        Calculate the gradient of MSE with respect to model parameters.

        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.

        Returns:
        --------
        gradient : ndarray
            Gradient vector.
        """
        n_samples = X.shape[0]
        error = y_pred - y_true
        return 2 * X.T.dot(error) / n_samples

    def fit(self, X, y):
        """
        Fit the linear regression model to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        --------
        self : Returns fitted instance of self.
        """
        # Store original data
        self.X = np.array(X)
        self.y = np.array(y)

        # Ensure X is 2D
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        # Apply feature scaling if enabled
        X_train = self.X
        if self.use_scaling:
            X_train = self.scaler.fit_transform(self.X)

        # Initialize parameters
        n_features = X_train.shape[1]
        self.params = np.zeros(n_features)
        self.trajectory = [self.params.copy()]

        # Gradient descent optimization
        for i in range(self.n_iterations):
            # Calculate predictions with current parameters
            y_pred = X_train.dot(self.params)

            # Calculate gradient
            gradient = self.mse_gradient(X_train, self.y, y_pred)

            # Update parameters
            self.params = self.params - self.learning_rate * gradient

            # Store parameters history
            self.trajectory.append(self.params.copy())

        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict on.

        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values.
        """
        X_new = np.array(X)

        # Ensure X is 2D
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)

        # Apply feature scaling if enabled
        if self.use_scaling and self.scaler is not None:
            X_new = self.scaler.transform(X_new)

        # Check if shapes are compatible
        if X_new.shape[1] != len(self.params):
            raise ValueError(f"X has {X_new.shape[1]} features, but model was trained with {len(self.params)} features")

        return X_new.dot(self.params)

    def plot_regression_results(self, title="Linear Regression Results", xlabel="Feature", ylabel="Target"):
        """
        Plot the regression results.

        Parameters:
        -----------
        title : str
            Plot title.
        xlabel : str
            Label for x-axis.
        ylabel : str
            Label for y-axis.
        """
        plt.figure(figsize=(12, 6))

        # For 1D data
        if self.X.shape[1] == 1:
            # Get the feature values
            x_values = self.X[:, 0]

            # Create scatter plot of original data
            plt.scatter(x_values, self.y, color='blue', alpha=0.5, label='Data')

            # Sort x for smooth line plotting
            x_sort_idx = np.argsort(x_values)
            x_sorted = x_values[x_sort_idx]

            # Create array for line plotting
            X_line = x_sorted.reshape(-1, 1)
            y_line = self.predict(X_line)

            # Plot the regression line
            plt.plot(x_sorted, y_line, 'r-', linewidth=2, label='Prediction')

            # Add a text with model info
            if self.use_scaling:
                model_info = f"y = {self.params[0]:.4f}*x_scaled"
                plt.text(0.05, 0.95, "Using Feature Scaling", transform=plt.gca().transAxes,
                         fontsize=10, va='top', bbox=dict(boxstyle='round', alpha=0.1))
            else:
                model_info = f"y = {self.params[0]:.4f}*x"

            plt.text(0.05, 0.9, model_info, transform=plt.gca().transAxes,
                     fontsize=10, va='top', bbox=dict(boxstyle='round', alpha=0.1))

        # For higher dimensions
        else:
            plt.text(0.5, 0.5, f"Cannot visualize {self.X.shape[1]}-dimensional data",
                     ha='center', va='center', fontsize=14)
            if self.use_scaling:
                plt.text(0.5, 0.4, "Using Feature Scaling", ha='center', va='center', fontsize=12)

            # Print model parameters
            param_text = "Model Parameters:\n"
            for i, p in enumerate(self.params):
                param_text += f"w{i} = {p:.4f}\n"
            plt.text(0.5, 0.3, param_text, ha='center', va='center', fontsize=10)

        # Add labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # If we have multiple iterations, plot the loss curve
        if len(self.trajectory) > 1:
            self._plot_convergence()

            # For 2D parameter space, plot the gradient descent optimization
            if len(self.params) == 2:
                self.plot_gradient_descent_optimization()

    def _plot_convergence(self):
        """Plot the convergence of parameters during training."""
        plt.figure(figsize=(12, 5))

        # Plot all parameters convergence
        for i, param in enumerate(np.array(self.trajectory).T):
            plt.plot(param, label=f'Parameter {i}')

        plt.title('Parameter Convergence Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_gradient_descent_optimization(self):
        """
        Plot the gradient descent optimization path for 2D parameter space.
        """
        # Only works for 2 parameters
        if len(self.params) != 2:
            print("Gradient descent visualization only available for 2D parameter space")
            return

        # We need to define a custom objective function for our specific case
        def model_mse(params):
            if self.use_scaling:
                X_scaled = self.scaler.transform(self.X)
                y_pred = X_scaled.dot(params)
            else:
                y_pred = self.X.dot(params)
            return np.mean((self.y - y_pred) ** 2)

        # Convert trajectory list to numpy array
        trajectory_array = np.array(self.trajectory)

        # Get min/max bounds for visualization
        param_min = np.min(trajectory_array, axis=0) - 0.5
        param_max = np.max(trajectory_array, axis=0) + 0.5

        # Create a grid of points
        p1 = np.linspace(param_min[0], param_max[0], 50)
        p2 = np.linspace(param_min[1], param_max[1], 50)
        P1, P2 = np.meshgrid(p1, p2)
        Z = np.zeros_like(P1)

        # Compute MSE values on the grid
        for i in range(P1.shape[0]):
            for j in range(P1.shape[1]):
                Z[i, j] = model_mse([P1[i, j], P2[i, j]])

        # Create subplots: 2D contour and 3D surface
        fig = plt.figure(figsize=(15, 6))

        # 2D contour plot with trajectory
        ax1 = fig.add_subplot(121)
        contour = ax1.contour(P1, P2, Z, levels=20, cmap='viridis')
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'r.-', linewidth=1, markersize=5)
        ax1.plot(trajectory_array[0, 0], trajectory_array[0, 1], 'go', markersize=8, label='Start')
        ax1.plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 'ro', markersize=8, label='End')
        ax1.set_title('Contour Plot of Loss Function')
        ax1.set_xlabel('Parameter 1')
        ax1.set_ylabel('Parameter 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(contour, ax=ax1)

        # 3D surface plot with trajectory
        ax2 = fig.add_subplot(122, projection='3d')
        surface = ax2.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.7, rstride=1, cstride=1)

        # Calculate MSE for each point in the trajectory
        trajectory_z = [model_mse(params) for params in trajectory_array]

        # Plot the trajectory on the 3D surface
        ax2.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_z, 'r.-', linewidth=2, markersize=5)
        ax2.set_title('3D Surface of Loss Function')
        ax2.set_xlabel('Parameter 1')
        ax2.set_ylabel('Parameter 2')
        ax2.set_zlabel('Mean Squared Error')
        plt.colorbar(surface, ax=ax2, shrink=0.7)

        # Add overall title
        scaling_status = "with Scaling" if self.use_scaling else "without Scaling"
        plt.suptitle(f"Gradient Descent Optimization {scaling_status}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# Example usage
def main():
    # First, demonstrate the basic gradient descent on a simple function
    initial_point = np.array([1.5, 1.5])
    learning_rate = 0.1
    optimal_x, trajectory = gradient_descent(
        objective_function, gradient_function, initial_point,
        learning_rate=learning_rate, n_iterations=100
    )
    print(f"Optimal point: {optimal_x}")
    plot_contour_and_trajectory(objective_function, trajectory,
                                title="Gradient Descent on Quadratic Function")

    # Now demonstrate linear regression with and without scaling
    np.random.seed(42)
    X = np.random.randn(100, 2) * np.array([10, 1])  # Features with different scales
    true_params = np.array([0.5, 2.0])
    y = X.dot(true_params) + np.random.randn(100) * 0.5

    # Without scaling
    lr_no_scaling = LinearRegression(learning_rate=0.01, n_iterations=100, use_scaling=False)
    lr_no_scaling.fit(X, y)
    print("Parameters without scaling:", lr_no_scaling.params)
    print("MSE without scaling:", lr_no_scaling.mse_loss(y, lr_no_scaling.predict(X)))
    lr_no_scaling.plot_regression_results(title="Linear Regression without Scaling")

    # With scaling
    lr_with_scaling = LinearRegression(learning_rate=0.01, n_iterations=100, use_scaling=True)
    lr_with_scaling.fit(X, y)
    print("Parameters with scaling:", lr_with_scaling.params)
    print("MSE with scaling:", lr_with_scaling.mse_loss(y, lr_with_scaling.predict(X)))
    lr_with_scaling.plot_regression_results(title="Linear Regression with Scaling")


if __name__ == "__main__":
    main()
