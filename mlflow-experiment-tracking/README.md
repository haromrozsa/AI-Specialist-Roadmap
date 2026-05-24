# MLflow Experiment Tracking

## Session Summary: Built a simple MLflow experiment tracking system demonstrating how to log parameters, metrics, dataset versions, and input statistics across multiple ML training runs, enabling systematic comparison and reproducibility of machine learning experiments.

## What I Did During the Session

1. **Implemented MLflow Tracking Integration**:
   - Created `train.py` with MLflow experiment tracking using `mlflow.start_run()` context manager
   - Logged hyperparameters (n_estimators, max_depth, min_samples_split) via `mlflow.log_param()`
   - Logged training metrics (accuracy, precision, recall, F1-score) via `mlflow.log_metric()`
   - Logged trained scikit-learn models as artifacts with `mlflow.sklearn.log_model()`
   - Named experiment "wine-classification" using `mlflow.set_experiment()`

2. **Added Dataset Versioning and Statistics**:
   - Implemented `get_dataset_version()` generating MD5 hash of dataset for version tracking
   - Built `calculate_data_stats()` computing mean, std, min, max for all features using pandas
   - Logged dataset metadata (version, name, sample count, feature count) as parameters
   - Logged per-feature mean values as metrics for input data profiling
   - Ensured reproducibility by tracking which dataset version produced which results

3. **Ran Multiple Experiments with Different Hyperparameters**:
   - Executed 5 training runs with varying configurations (50-200 estimators, depth 3-10, split 2-5)
   - Used Wine dataset from scikit-learn (178 samples, 13 features, 3 classes)
   - Achieved perfect classification (100% accuracy) on test set across all configurations
   - All experiments tracked same dataset version (`v1_147ca828`) for consistency
   - Results stored in `mlruns/` directory with automatic experiment organization

4. **Created MLflow UI Access and Documentation**:
   - Provided instructions to launch MLflow tracking UI with `mlflow ui` command
   - Documented how to view and compare experiments at http://localhost:5000
   - Created usage guide covering installation, execution, and UI navigation
   - Added screenshot instructions for capturing GitHub proof (experiments list, run comparison, run detail)

5. **Demonstrated MLflow Core Capabilities**:
   - Systematic experiment tracking replacing ad-hoc print statements
   - Side-by-side comparison of runs to identify best hyperparameters
   - Automatic artifact storage enabling model retrieval without manual file management
   - Data awareness through version tracking and statistics logging
   - Foundation for reproducible ML workflows

## What I Learned

1. **MLflow Experiment Tracking Basics**:
   - MLflow is an open-source platform for managing the ML lifecycle (tracking, projects, models, registry)
   - `mlflow.start_run()` creates an isolated run context tracking all logged params/metrics/artifacts
   - `mlflow.log_param()` logs input values (hyperparameters, config), `mlflow.log_metric()` logs output values (accuracy, loss)
   - `mlflow.sklearn.log_model()` saves trained models in MLflow format enabling easy reload and deployment
   - The `mlruns/` directory stores all experiment data as local file-based backend (production uses remote tracking servers)

2. **Dataset Versioning and Reproducibility**:
   - Dataset versioning via hashing (MD5) ensures you know exactly which data produced which model
   - Logging input statistics (mean, std, min, max) helps detect data drift when comparing experiments
   - Parameters in MLflow are immutable strings/numbers; metrics are mutable numeric values tracked over time
   - Tracking dataset version + hyperparameters + metrics creates full reproducibility chain
   - Without versioning, "accuracy dropped" is ambiguous — with versioning, you know if data or code changed

3. **MLflow UI for Experiment Comparison**:
   - The MLflow UI (`mlflow ui`) provides web-based visualization at http://localhost:5000
   - Experiments list shows all runs with sortable columns (params, metrics, run ID, timestamp)
   - Run comparison feature allows selecting multiple runs to see parameter/metric differences side-by-side
   - Single run view displays full details: all params, all metrics, artifacts (models), and metadata
   - UI replaces manual spreadsheet tracking and enables instant "which config was best?" answers

4. **Integration with ML Frameworks**:
   - MLflow has native integrations: `mlflow.sklearn`, `mlflow.pytorch`, `mlflow.tensorflow`, `mlflow.transformers`, `mlflow.onnx`
   - For scikit-learn, `log_model()` stores the entire pipeline (preprocessing + model) as a pickle-like artifact
   - Models logged with MLflow can be served as REST APIs via `mlflow models serve` (deployment without custom API code)
   - `mlflow.autolog()` can automatically track params/metrics/models for supported frameworks (TensorFlow, PyTorch, XGBoost, etc.)
   - Same MLflow tracking code works across frameworks — unified interface for all ML experiments

5. **MLflow in the ML Lifecycle**:
   - MLflow Tracking (used here) is phase 1: experimentation and finding best model
   - Future phases use MLflow Models (portable format), MLflow Registry (version control for production models), MLflow Projects (reproducible runs)
   - Complements existing tools: ONNX for model export, FastAPI for serving, Docker for deployment
   - Not a replacement for training frameworks (PyTorch, scikit-learn) but a layer on top for management
   - In production workflows: train with PyTorch → track with MLflow → export to ONNX → serve with FastAPI → deploy with Docker

This session provided hands-on experience with MLflow experiment tracking, demonstrating how to systematically log parameters, metrics, and dataset metadata to enable reproducible ML workflows and informed model selection.
