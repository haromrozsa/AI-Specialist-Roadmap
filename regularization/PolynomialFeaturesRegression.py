import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate simple synthetic data for visualization
# Creating a non-linear relationship that would benefit from polynomial features
n_samples = 100
X_simple = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_simple = 2 + 0.8 * X_simple[:, 0] + 2 * X_simple[:, 0] ** 2 + 0.2 * X_simple[:, 0] ** 3 + np.random.randn(
    n_samples) * 0.5

# 2. Visualize the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, s=40, alpha=0.8)
plt.title('Synthetic Non-Linear Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()

# 3. Fit different degree polynomial models to the synthetic data
degrees = [1, 2, 3, 5, 10]  # Degrees of polynomial features to try
X_plot = np.linspace(-3.5, 3.5, 1000).reshape(-1, 1)

plt.figure(figsize=(14, 10))
for i, degree in enumerate(degrees):
    # Create and fit polynomial regression model
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_simple)

    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y_simple)

    # Predict on smooth grid for visualization
    X_plot_poly = poly_features.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)

    # Plot the results
    plt.subplot(2, 3, i + 1)
    plt.scatter(X_simple, y_simple, s=30, alpha=0.6, label='Data points')
    plt.plot(X_plot, y_plot_pred, color='red', label=f'Degree {degree} Polynomial')
    plt.title(f'Polynomial Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Now use the California housing dataset for a more realistic example
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {housing.feature_names}")

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Use only 2 features for demonstration (to avoid too many polynomial features)
X_train_small = X_train[['MedInc', 'AveRooms']]
X_test_small = X_test[['MedInc', 'AveRooms']]


# 7. Create a pipeline with preprocessing and model
def create_poly_pipeline(degree=1, alpha=0.0):
    steps = [
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', Ridge(alpha=alpha))
    ]
    return Pipeline(steps)


# 8. Train and evaluate models with different polynomial degrees
max_degree = 5
results = []

for degree in range(1, max_degree + 1):
    # Create and train the model
    pipeline = create_poly_pipeline(degree=degree, alpha=0.1)
    pipeline.fit(X_train_small, y_train)

    # Make predictions
    y_train_pred = pipeline.predict(X_train_small)
    y_test_pred = pipeline.predict(X_test_small)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Store results
    results.append({
        'Degree': degree,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': train_r2,
        'Test R²': test_r2
    })

    # Print results
    print(f"Degree {degree}:")
    print(f"  Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"  Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    # Calculate number of features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    n_features = poly.fit_transform(X_train_small).shape[1]
    print(f"  Number of polynomial features: {n_features}")
    print()

# 9. Plot the learning curves (training vs testing error) - FIXED CODE
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))

# Fix for the plotting error - convert to numpy arrays before plotting
degrees = results_df['Degree'].values  # Convert to numpy array
train_mse = results_df['Train MSE'].values
test_mse = results_df['Test MSE'].values
train_r2 = results_df['Train R²'].values
test_r2 = results_df['Test R²'].values

plt.subplot(1, 2, 1)
plt.plot(degrees, train_mse, marker='o', label='Training MSE')
plt.plot(degrees, test_mse, marker='s', label='Testing MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Polynomial Degree')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(degrees, train_r2, marker='o', label='Training R²')
plt.plot(degrees, test_r2, marker='s', label='Testing R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('R² vs Polynomial Degree')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Visualize the polynomial features effect on 3D surface plot (for 2 features)
from mpl_toolkits.mplot3d import Axes3D

# Create a mesh grid for visualization
x_min, x_max = X_train_small['MedInc'].min() - 1, X_train_small['MedInc'].max() + 1
y_min, y_max = X_train_small['AveRooms'].min() - 1, X_train_small['AveRooms'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))

# Choose two degrees to compare
degrees_to_visualize = [1, 3]

plt.figure(figsize=(18, 8))
for i, degree in enumerate(degrees_to_visualize):
    # Create and train the model
    pipeline = create_poly_pipeline(degree=degree, alpha=0.1)
    pipeline.fit(X_train_small, y_train)

    # Predict on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = pipeline.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot the surface
    ax = plt.subplot(1, 2, i + 1, projection='3d')
    surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.7)

    # Plot the actual data points
    ax.scatter(X_train_small['MedInc'], X_train_small['AveRooms'], y_train,
               c='red', s=20, alpha=0.5)

    ax.set_xlabel('Median Income')
    ax.set_ylabel('Average Rooms')
    ax.set_zlabel('House Value')
    ax.set_title(f'Polynomial Regression (Degree {degree})')

plt.tight_layout()
plt.show()

# 11. Analyze feature importance with different polynomial degrees
# Get the index of the best degree based on Test R²
best_degree_idx = np.argmax(test_r2)
best_degree = degrees[best_degree_idx]
print(f"\nBest polynomial degree based on testing R²: {best_degree}")

# Fit the model with the best degree
best_pipeline = create_poly_pipeline(degree=int(best_degree), alpha=0.1)
best_pipeline.fit(X_train_small, y_train)

# Get feature names from PolynomialFeatures
poly_features = PolynomialFeatures(degree=int(best_degree), include_bias=False)
poly_features.fit(X_train_small)
feature_names = poly_features.get_feature_names(['MedInc', 'AveRooms'])


# Extract coefficients
coefficients = best_pipeline.named_steps['model'].coef_

# Create a DataFrame with feature names and coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 most important polynomial features:")
print(coef_df.head(10))

# Plot the top coefficients
plt.figure(figsize=(12, 8))
top_n = 10
top_coef = coef_df.head(top_n)
plt.barh(top_coef['Feature'], top_coef['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Feature Coefficients (Degree {best_degree})')
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Make final predictions and evaluate the best model
y_pred = best_pipeline.predict(X_test_small)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print(f"\nFinal Model Performance:")
print(f"  MSE: {final_mse:.4f}")
print(f"  R²: {final_r2:.4f}")

# 13. Comparing with original features (all features, not just 2)
print("\nComparison with all original features:")
full_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=0.1))
])
full_pipeline.fit(X_train, y_train)
full_y_pred = full_pipeline.predict(X_test)
full_mse = mean_squared_error(y_test, full_y_pred)
full_r2 = r2_score(y_test, full_y_pred)
print(f"  All features (no polynomials) - MSE: {full_mse:.4f}, R²: {full_r2:.4f}")
