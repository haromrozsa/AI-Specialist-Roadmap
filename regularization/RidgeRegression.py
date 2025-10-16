import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and prepare the California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {housing.feature_names}")

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize the features (important for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train the Ridge regression model
# Alpha is the regularization strength: higher values = more regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = ridge.predict(X_test_scaled)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': ridge.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# 8. Visualize the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Ridge Regression Coefficients')
plt.tight_layout()
plt.show()

# 9. Try different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    # Using 5-fold cross-validation to get robust performance estimates
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train,
                                cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(np.mean(np.sqrt(-cv_scores)))

# 10. Plot the effect of alpha
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas), ridge_scores)
plt.xlabel('log(alpha)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Ridge Regression: RMSE vs. Alpha (Regularization Strength)')
plt.tight_layout()
plt.show()

# 11. Find the best alpha value
best_alpha_idx = np.argmin(ridge_scores)
best_alpha = alphas[best_alpha_idx]
print(f"\nBest Alpha Value: {best_alpha}")

# 12. Final model with best alpha
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(X_train_scaled, y_train)
final_y_pred = final_ridge.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, final_y_pred)
final_r2 = r2_score(y_test, final_y_pred)

print(f"Final Model MSE: {final_mse:.4f}")
print(f"Final Model R²: {final_r2:.4f}")

# 13. Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()
