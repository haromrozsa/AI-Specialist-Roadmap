import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Very simple
## 1. Feature Engineering
## 2. Model Selection
## 3. Regularization
## 4. Hyperparameter Tuning
## 5. Outlier Handling
## 6. Feature Selection
## 7. Data Transformations
## 8. Cross-Validation Strategy
## 9. Ensemble Methods
## 10. Advanced Preprocessing


california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {california.feature_names}")
print("\nFeature statistics:")
print(X.describe())

plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], y, alpha=0.5)
plt.title('Median Income vs. Median House Value')
plt.xlabel('Median Income ($10k)')
plt.ylabel('Median House Value ($100k)')
plt.grid(True)
plt.show()

X.hist(bins=100, figsize=(25, 20))
plt.show()

# Split first, then scale. Fitting the scaler on all 20,640 rows before the
# split would compute its mean and standard deviation partly from the test rows
# the model is about to be scored on -- the preprocessing leak that
# feature_pipelines/ measures. Here it is fit on the training rows only and
# merely applied to the test rows, so the test set stays genuinely unseen.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

coefficients = pd.DataFrame(
    model.coef_,
    index=california.feature_names,
    columns=['Coefficient']
)

coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(coefficients)

plt.figure(figsize=(10, 6))
coefficients['Coefficient'].sort_values().plot(kind='barh')
plt.title('Feature Coefficients in Linear Regression Model')
plt.xlabel('Coefficient Value')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs. Predicted House Values')
plt.xlabel('Actual Values ($100k)')
plt.ylabel('Predicted Values ($100k)')
plt.grid(True)
plt.show()
