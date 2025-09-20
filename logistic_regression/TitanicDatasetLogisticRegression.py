# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
# Option 1: If you have seaborn installed
titanic = sns.load_dataset('titanic')

# Option 2: Alternative URL source
# url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# titanic = pd.read_csv(url)

# Set style
sns.set_theme(style="whitegrid")

# Survival distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='survived', data=titanic)
plt.title('Survival Distribution')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# Survival by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival by Gender')
plt.show()

# Survival by passenger class
plt.figure(figsize=(10, 6))
sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival Status')
plt.show()

# Select features for the model
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = titanic[features]
y = titanic['survived']

# Define numerical and categorical features
numerical_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create the pipeline with preprocessing and logistic regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Get the logistic regression coefficients
classifier = model.named_steps['classifier']
coefficients = classifier.coef_[0]

# Get feature names after preprocessing
preprocessor = model.named_steps['preprocessor']
column_names = []
for name, trans, features in preprocessor.transformers_:
    if name == 'num':
        column_names.extend(features)
    elif name == 'cat':
        # For categorical features, get the one-hot encoded column names
        for i, feature in enumerate(features):
            categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
            # First category is dropped due to drop='first' in OneHotEncoder
            column_names.extend([f"{feature}_{cat}" for cat in categories[1:]])

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({
    'Feature': column_names,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value
feature_importance = feature_importance.reindex(
    feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Coefficient (Impact on Survival Probability)')
plt.axvline(x=0, color='gray', linestyle='-')
plt.grid(axis='x')
plt.show()

# Print interpretation
print("Model Interpretation:")
print("-" * 50)
print("The logistic regression model shows that:")

# Print top positive coefficients (increase survival probability)
print("\nTop factors that INCREASE survival probability:")
for _, row in feature_importance[feature_importance['Coefficient'] > 0].head(5).iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

# Print top negative coefficients (decrease survival probability)
print("\nTop factors that DECREASE survival probability:")
for _, row in feature_importance[feature_importance['Coefficient'] < 0].head(5).iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

print("\nConclusions:")
print("-" * 50)
print("1. Gender was a critical factor (women had higher survival rates)")
print("2. Passenger class played an important role (higher class = better survival)")
print("3. Age was significant (children more likely to survive)")
print("4. Family size had a complex effect on survival")

