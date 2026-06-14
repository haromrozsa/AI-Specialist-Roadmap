# SageMaker Pipeline with CI/CD Demo

## Session Summary: Built a complete SageMaker ML pipeline demonstration with training, evaluation, and conditional model registration, automated via GitHub Actions CI/CD—showcasing modern MLOps workflows without requiring real AWS infrastructure.

## What I Did During the Session

1. **Designed 3-Step SageMaker Pipeline Architecture**:
   - Created sequential workflow: `TrainIrisClassifier` → `EvaluateModel` → `CheckAccuracyCondition` → `RegisterIrisModel`
   - Defined pipeline parameters (`NEstimators`, `TestSize`, `AccuracyThreshold`, `InstanceType`) for flexible configuration
   - Implemented conditional logic using `ConditionGreaterThanOrEqualTo` to register model only if accuracy ≥ threshold
   - Used `PropertyFile` to pass evaluation metrics between steps for condition checking
   - Configured model registration to SageMaker Model Registry with approval status `PendingManualApproval`

2. **Implemented ML Pipeline Components as Modular Scripts**:
   - **train.py**: Trains Random Forest (100 estimators) on Iris dataset, saves model as `model.pkl` and metadata JSON with train/test accuracy
   - **evaluate.py**: Loads trained model, calculates accuracy/precision/recall/F1-score, determines deployment approval based on threshold
   - **pipeline.py**: Defines complete SageMaker Pipeline with `TrainingStep`, `ProcessingStep`, `RegisterModel`, and `ConditionStep`
   - Used scikit-learn for simplicity—Iris dataset (150 samples, 4 features, 3 classes) trains in seconds
   - Structured outputs as JSON for easy consumption by downstream steps and CI/CD validation

3. **Created Local Pipeline Simulation Without AWS**:
   - Built `run_local.py` to execute pipeline workflow locally, simulating all 3 steps without SageMaker infrastructure
   - Reproduced training → evaluation → conditional registration logic with same scripts used in real pipeline
   - Structured logging with step-by-step execution status (✓ SUCCESS, ✗ FAILED, SKIPPED)
   - Generated comprehensive outputs: `model.pkl`, `training_metadata.json`, `evaluation_results.json`, `pipeline_execution.json`
   - Enabled rapid development and testing without AWS costs or complex setup

4. **Integrated GitHub Actions CI/CD Workflow**:
   - Created `.github/workflows/validate.yml` with 7 steps: checkout, setup Python, install deps, run training, run evaluation, validate pipeline, run full simulation
   - Automated testing on every push/PR to ensure pipeline definition is valid and scripts execute successfully
   - Added output validation checks to confirm all expected files are created (`model.pkl`, `evaluation_results.json`, etc.)
   - Displayed pipeline execution results in CI logs for visibility into model performance
   - Demonstrates modern MLOps practice: code changes trigger automated ML pipeline execution and validation

5. **Documented Deployment and Usage**:
   - Created `USAGE.md` with quick start guide, parameter customization, output file descriptions, and troubleshooting tips
   - Added LocalStack Docker setup for optional AWS simulation (limited SageMaker support, primarily for S3/IAM testing)
   - Included example commands for running pipeline with custom hyperparameters (`--n-estimators`, `--test-size`, `--accuracy-threshold`)
   - Explained pipeline step flow, conditional registration logic, and expected outputs
   - Provided GitHub Actions workflow overview showing automated validation on push/PR

## What I Learned

1. **SageMaker Pipelines for ML Workflow Orchestration**:
   - SageMaker Pipelines define ML workflows as directed acyclic graphs (DAGs) with `TrainingStep`, `ProcessingStep`, `TransformStep`, `ConditionStep`, etc.
   - `ParameterInteger`, `ParameterFloat`, `ParameterString` enable runtime configuration without code changes—critical for experimentation
   - `PropertyFile` extracts outputs (JSON) from one step to use as inputs/conditions in downstream steps (e.g., evaluation metrics → conditional registration)
   - `ConditionStep` with `ConditionGreaterThanOrEqualTo` enables data-driven decisions (deploy only if metrics pass threshold)
   - Model registration to SageMaker Model Registry creates versioned, auditable model lineage with approval workflows

2. **Pipeline Parameters and Conditional Logic**:
   - Pipeline parameters are defined at pipeline creation but resolved at execution time, enabling reusable pipeline definitions
   - `PropertyFile` reads JSON output from processing/training steps, allowing dynamic conditions based on actual results (not hardcoded)
   - Conditional steps create branching logic: `if_steps` execute when condition is true, `else_steps` when false
   - This pattern (train → evaluate → conditional deploy) prevents bad models from reaching production automatically
   - Real-world pipelines use similar logic for A/B testing, multi-model comparison, and staged rollouts

3. **Local Development for Cloud Pipelines**:
   - Developing/debugging SageMaker Pipelines directly in AWS is slow and expensive (training jobs take minutes to start, cost money)
   - Local simulation scripts (`run_local.py`) replicate pipeline logic using same code, enabling fast iteration without cloud resources
   - LocalStack provides limited SageMaker support (basic API stubs), but full workflow simulation requires custom scripting
   - Key strategy: write scripts (`train.py`, `evaluate.py`) that work locally AND in SageMaker (using standard paths like `/opt/ml/model`)
   - This approach speeds up development 10x—test locally first, deploy to SageMaker only when confident

4. **CI/CD for Machine Learning Pipelines**:
   - GitHub Actions (or GitLab CI, Jenkins) can automate ML pipeline testing: validate definition syntax, run scripts, check outputs
   - Unlike traditional software CI (unit tests → build → deploy), ML CI includes model training, evaluation, and metric validation
   - Automated pipeline execution on every commit ensures code changes don't break training/evaluation scripts
   - CI can enforce quality gates (e.g., fail build if accuracy < threshold) to prevent deploying degraded models
   - Modern MLOps = version control (Git) + automation (CI/CD) + orchestration (Pipelines) + monitoring (CloudWatch/MLflow)

5. **MLOps Best Practices and Production Patterns**:
   - Separating orchestration (SageMaker Pipelines) from execution (training/evaluation scripts) enables testing scripts independently
   - Parameterized pipelines support experimentation without code duplication (change `NEstimators` via parameter, not hardcoding)
   - Conditional registration prevents manual errors—model only reaches registry if metrics are acceptable
   - JSON-based communication between steps (via S3/PropertyFile) creates auditable, debuggable workflows
   - Portfolio value: demonstrates understanding of end-to-end ML lifecycle (training → evaluation → registration → deployment), not just modeling

This session provided hands-on experience building SageMaker ML pipelines with conditional logic, implementing local simulation for rapid development, automating validation via GitHub Actions CI/CD, and documenting production-ready MLOps workflows—demonstrating the full ML engineering stack beyond just training models.
