import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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

# 3. Standardize the features (critical for Lasso regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train the Lasso regression model
# Alpha is the regularization strength: higher values = more regularization
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = lasso.predict(X_test_scaled)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': lasso.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# 8. Visualize the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Lasso Regression Coefficients')
plt.tight_layout()
plt.show()

# 9. Count non-zero coefficients (feature selection aspect of Lasso)
non_zero = np.sum(lasso.coef_ != 0)
print(f"\nNumber of features used by the Lasso model: {non_zero} out of {len(housing.feature_names)}")

# 10. Try different alpha values
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
lasso_scores = []
n_features_selected = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    # Using 5-fold cross-validation to get robust performance estimates
    cv_scores = cross_val_score(lasso, X_train_scaled, y_train,
                                cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(np.mean(np.sqrt(-cv_scores)))

    # Fit on all training data to count selected features
    lasso.fit(X_train_scaled, y_train)
    n_features_selected.append(np.sum(lasso.coef_ != 0))

# 11. Plot the effect of alpha on RMSE
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas), lasso_scores)
plt.xlabel('log(alpha)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Lasso Regression: RMSE vs. Alpha (Regularization Strength)')
plt.tight_layout()
plt.show()

# 12. Plot the effect of alpha on feature selection
plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas), n_features_selected, marker='o')
plt.xlabel('log(alpha)')
plt.ylabel('Number of Features Selected')
plt.title('Lasso Regression: Feature Selection vs. Alpha')
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. Find the best alpha value
best_alpha_idx = np.argmin(lasso_scores)
best_alpha = alphas[best_alpha_idx]
print(f"\nBest Alpha Value: {best_alpha}")

# 14. Final model with best alpha
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train_scaled, y_train)
final_y_pred = final_lasso.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, final_y_pred)
final_r2 = r2_score(y_test, final_y_pred)

print(f"Final Model MSE: {final_mse:.4f}")
print(f"Final Model R²: {final_r2:.4f}")
print(f"Features selected by final model: {np.sum(final_lasso.coef_ != 0)} out of {len(housing.feature_names)}")

# 15. Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()

# 16. Print which features were selected by the final model
selected_features = coef_df[coef_df['Coefficient'] != 0]
print("\nSelected Features and their coefficients:")
print(selected_features)
