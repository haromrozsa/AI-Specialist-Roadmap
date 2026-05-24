# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## Run Experiments

Execute the training script to run 5 experiments with different hyperparameters:

```bash
python train.py
```

This will:
- Train 5 Random Forest models with different configurations
- Log all parameters (n_estimators, max_depth, min_samples_split)
- Log metrics (accuracy, precision, recall, f1_score)
- Log dataset version and input statistics
- Save all results to `mlruns/` directory

## View MLflow UI

Launch the MLflow tracking UI:

```bash
mlflow ui
```

Then open your browser to: **http://localhost:5000**

In the UI you can:
- Compare all experiment runs
- Sort by metrics (accuracy, F1-score)
- View parameters for each run
- See dataset versions used
- Download trained models

## What Gets Tracked

### Parameters
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of trees
- `min_samples_split`: Minimum samples required to split a node
- `dataset_version`: Hash-based version of the dataset
- `n_samples_train`: Number of training samples
- `n_features`: Number of features

### Metrics
- `accuracy`: Classification accuracy
- `precision`: Weighted precision score
- `recall`: Weighted recall score
- `f1_score`: Weighted F1 score
- `feature_mean_*`: Mean value of each feature (13 wine features)

### Artifacts
- Trained scikit-learn Random Forest model

## Dataset

Uses the **Wine Dataset** from scikit-learn:
- 178 samples
- 13 features (chemical properties)
- 3 classes (wine cultivars)
- 80/20 train/test split
