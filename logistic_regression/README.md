# ML Fundamentals – Logistic Regression 

# Session Summary: Logistic Regression Basics + Titanic Dataset

## What I Did During the Session

1. **Implemented Logistic Regression from Scratch**
  - Created a custom `SimpleLogisticRegression` class with:
    - Core methods like sigmoid function, fit, predict_proba, predict, and score
    - Configurable parameters including learning rate, iterations, and intercept fitting
    - Cost history tracking for monitoring convergence

2. **Used Scikit-learn's Logistic Regression**
  - Implemented a scikit-learn based logistic regression model with:
    - L2 regularization (ridge regression)
    - LBFGS solver for optimization
    - Standard parameters configuration for binary classification

3. **Worked with Different Datasets**
   - Generated synthetic data for binary classification using scikit-learn
   - Applied logistic regression to the Titanic dataset, a real-world survival prediction problem

4. **Data Preprocessing**
   - Split data into training and testing sets
   - Applied feature standardization using StandardScaler
   - Handled both numerical and categorical features in the Titanic dataset
   - Created appropriate preprocessing pipelines for mixed data types

5. **Model Evaluation**
   - Calculated and analyzed accuracy scores
   - Generated and interpreted classification reports
   - Created and examined confusion matrices
   - Visualized model performance

6. **Visualization Techniques**
   - Plotted decision boundaries to visualize how the model separates classes
   - Created scatter plots of training and test data
   - Implemented custom visualization functions to better understand model behavior
   - Compared model performance on training versus test data

7. **Feature Analysis**
   - Examined model coefficients to understand feature importance
   - Analyzed how different features contribute to the classification decision
   - Investigated the relationship between features and target variables in the Titanic dataset

## What I Learned
1. **Logistic Regression Fundamentals**:
    - Logistic regression predicts binary outcomes using probability
    - The sigmoid function transforms linear predictions to [0,1] range
    - Decision boundaries are linear in feature space

2. **Mathematical Concepts**:
    - Binary cross-entropy as the loss function
    - Gradient descent optimization principles
    - The role of learning rate in convergence

3. **Model Evaluation Techniques**:
    - Accuracy as a primary performance metric
    - Using confusion matrices to understand error types
    - Interpreting precision, recall, and F1-score

4. **Visualization Approaches**:
    - Plotting decision boundaries
    - Visualizing class separation
    - Feature importance interpretation

5. **Feature Importance Analysis**:
    - Interpreting model coefficients
    - Identifying influential features
    - Understanding how features impact classification decisions

6. **Practical Applications**:
    - Binary classification for real-world problems
    - Survival prediction in the Titanic dataset
    - Creating effective preprocessing pipelines for mixed data types

This session provided both theoretical understanding and practical implementation experience with logistic regression, covering everything from basic concepts to advanced techniques for real-world classification problems.
