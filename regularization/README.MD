# ML Fundamentals – Linear Regression 

# Session Summary: Regularization & Polynomial Features

## What I Did During the Session

1. **Implemented Multiple Regression Models**:
    - Developed polynomial regression with varying complexity degrees
    - Implemented ridge regression with different regularization strengths
    - Created lasso regression for feature selection and sparse modeling
    - Built elastic net regression combining L1 and L2 regularization approaches

2. **Performed Feature Engineering and Transformation**:
    - Generated polynomial features of different degrees (1-10) from original features
    - Created interaction terms between different features
    - Applied feature scaling and standardization for regularized models
    - Built scikit-learn pipelines combining preprocessing and modeling steps

3. **Conducted Model Selection and Hyperparameter Tuning**:
    - Tested different alpha values for ridge regression regularization
    - Examined varying penalty weights in lasso regression
    - Explored L1 ratio impact in elastic net models
    - Determined optimal polynomial degrees to balance complexity and generalization

4. **Evaluated Model Performance**:
    - Split data into training and testing sets to evaluate generalization
    - Calculated mean squared error (MSE) and R² metrics across models
    - Visualized bias-variance tradeoffs with learning curves
    - Compared model performance across different complexity levels

5. **Visualized Results and Model Behavior**:
    - Created 2D and 3D plots showing regression surfaces for polynomial models
    - Plotted coefficient values to interpret feature importance
    - Visualized regularization effects on model coefficients
    - Generated actual vs. predicted plots to assess model fit

## What I Learned
1. **Regularization Techniques**:
    - I understood how ridge regression (L2) shrinks coefficients without eliminating features
    - I discovered how lasso regression (L1) performs feature selection by zeroing out coefficients
    - I learned how elastic net combines L1 and L2 penalties to handle correlated features
    - I gained insight into how regularization strength affects model complexity

2. **Model Complexity Management**:
    - I learned how to select appropriate polynomial degrees to balance fit and generalization
    - I understood the bias-variance tradeoff across different model complexities
    - I discovered how regularization prevents overfitting in high-dimensional feature spaces
    - I developed intuition for selecting optimal regularization parameters

3. **Feature Engineering Applications**:
    - I understood how polynomial features capture non-linear relationships
    - I learned how interaction terms model dependencies between features
    - I discovered the exponential growth in feature space with higher polynomial degrees
    - I gained experience in feature importance analysis across transformed feature spaces

4. **Practical Implementation Skills**:
    - I implemented efficient scikit-learn pipelines for preprocessing and modeling
    - I learned techniques for visualizing high-dimensional model predictions
    - I developed systematic approaches for hyperparameter tuning
    - I gained experience adapting code for different scikit-learn API versions

Through this session, I've developed a comprehensive understanding of advanced regression techniques, including polynomial transformation, ridge, lasso, and elastic net regularization. I've gained practical experience in implementing these models, visualizing their behavior, and selecting optimal parameters to balance model complexity with generalization performance.


