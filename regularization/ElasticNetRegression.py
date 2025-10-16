import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
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

# 3. Standardize the features (critical for regularization methods)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train the Elastic Net regression model
# alpha: controls overall regularization strength
# l1_ratio: controls the balance between L1 and L2 penalties (0=Ridge, 1=Lasso)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
elastic_net.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = elastic_net.predict(X_test_scaled)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': elastic_net.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coef_df)

# 8. Visualize the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Elastic Net Regression Coefficients (alpha=0.1, l1_ratio=0.5)')
plt.tight_layout()
plt.show()

# 9. Count non-zero coefficients (feature selection aspect from Lasso component)
non_zero = np.sum(elastic_net.coef_ != 0)
print(f"\nNumber of features used by the Elastic Net model: {non_zero} out of {len(housing.feature_names)}")

# 10. Grid search for best hyperparameters (alpha and l1_ratio)
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid_search = GridSearchCV(
    ElasticNet(max_iter=10000),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']

print(f"\nBest Alpha: {best_alpha}")
print(f"Best L1 Ratio: {best_l1_ratio}")
print(f"Best MSE: {-grid_search.best_score_:.4f}")

# 11. Visualize hyperparameter search results
results = pd.DataFrame(grid_search.cv_results_)
alpha_values = param_grid['alpha']
l1_ratio_values = param_grid['l1_ratio']

scores = np.array([results.loc[results['param_alpha'] == alpha].loc[results['param_l1_ratio'] == l1_ratio][
                       'mean_test_score'].values[0]
                   for alpha in alpha_values
                   for l1_ratio in l1_ratio_values]).reshape(len(alpha_values), len(l1_ratio_values))

plt.figure(figsize=(12, 8))
sns.heatmap(scores, annot=True, fmt='.3g', xticklabels=l1_ratio_values, yticklabels=alpha_values, cmap='viridis')
plt.xlabel('L1 Ratio')
plt.ylabel('Alpha')
plt.title('Negative MSE for Different Hyperparameters')
plt.tight_layout()
plt.show()

# 12. Final model with best hyperparameters
final_elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
final_elastic_net.fit(X_train_scaled, y_train)
final_y_pred = final_elastic_net.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, final_y_pred)
final_r2 = r2_score(y_test, final_y_pred)

print(f"\nFinal Model MSE: {final_mse:.4f}")
print(f"Final Model R²: {final_r2:.4f}")
print(f"Features selected by final model: {np.sum(final_elastic_net.coef_ != 0)} out of {len(housing.feature_names)}")

# 13. Compare coefficient patterns across different l1_ratios
l1_ratios_to_compare = [0.0, 0.25, 0.5, 0.75, 1.0]
models = []
feature_counts = []

for l1_ratio in l1_ratios_to_compare:
    model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    models.append(model)
    feature_counts.append(np.sum(model.coef_ != 0))

# Plot feature selection vs l1_ratio
plt.figure(figsize=(10, 6))
plt.plot(l1_ratios_to_compare, feature_counts, marker='o')
plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
plt.ylabel('Number of Features Selected')
plt.title(f'Feature Selection vs L1 Ratio (alpha={best_alpha})')
plt.grid(True)
plt.tight_layout()
plt.show()

# 14. Compare coefficients across different l1_ratios
coefs = pd.DataFrame()
for i, l1_ratio in enumerate(l1_ratios_to_compare):
    model = models[i]
    coefs[f'l1_ratio={l1_ratio}'] = model.coef_

coefs.index = housing.feature_names

plt.figure(figsize=(12, 8))
sns.heatmap(coefs, annot=True, cmap='coolwarm', center=0, fmt='.3g')
plt.title('Coefficients for Different L1 Ratios')
plt.tight_layout()
plt.show()

# 15. Plot actual vs predicted values for final model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Elastic Net Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()

# 16. Print which features were selected by the final model
final_coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': final_elastic_net.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

selected_features = final_coef_df[final_coef_df['Coefficient'] != 0]
print("\nSelected Features and their coefficients:")
print(selected_features)
