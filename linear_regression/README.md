# ML Fundamentals – Linear Regression 

# Session Summary: Linear Regression on California Housing Dataset

## What I Did During the Session

1. **Linear Regression Fundamental**:
   - Data Splitting
   - Implement basic linear regression using scikit-learn
   - Model Evaluation
   - The Impact of Noise
   - Data Visualization

2. **Explored the California Housing Dataset**:
   - I worked with the scikit-learn California Housing dataset
   - I examined the dataset's features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`)

3. **Built a Linear Regression Model**:
   - I loaded and preprocessed the housing data
   - I applied StandardScaler to normalize the features
   - I split the data into training and testing sets
   - I trained a linear regression model
   - I evaluated the model's performance using MSE, RMSE, and R² metrics

4. **Analyzed Model Results**:
   - I examined feature coefficients and their importance
   - I created visualizations including:
     - Scatter plot of median income vs. house value
     - Histograms of feature distributions
     - Feature coefficient importance chart
     - Actual vs. predicted values comparison

5. **Sought Improvement Methods**:
   - approaches to improve my model and reduce RMSE

## What I Learned

1. **Dataset Characteristics**:
   - I confirmed the features available in the scikit-learn California Housing dataset
   - I understood the difference between this dataset and other variants that might include "ocean proximity"

2. **Model Improvement Techniques**:
   - **Feature Engineering**: I can create interaction terms, polynomial features, and geographical insights
   - **Advanced Models**: I have options beyond linear regression (Random Forest, Gradient Boosting, etc.)
   - **Regularization**: I can use Ridge, Lasso, and Elastic Net to prevent overfitting
   - **Hyperparameter Tuning**: I can systematically optimize with GridSearchCV
   - **Outlier Handling**: I learned methods to deal with extreme values
   - **Feature Selection**: I discovered techniques to identify and use the most relevant features
   - **Data Transformations**: I can apply mathematical transformations to improve feature distributions
   - **Cross-Validation**: I should use robust validation techniques
   - **Ensemble Methods**: I can combine multiple models for better predictions
   - **Advanced Preprocessing**: I learned about sophisticated data preparation techniques

Through this session, I've learned about the linear regression, I've gained insights into both the characteristics of the California Housing dataset and a comprehensive set of techniques to improve regression model performance, specifically focused on reducing RMSE to enhance my forecasting accuracy.