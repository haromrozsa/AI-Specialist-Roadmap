# Usage Guide

## Quick Start (Local Simulation)

### Prerequisites
- Python 3.11+
- pip

### Installation

**Option 1: Automated Setup (Easiest)**

**Windows:**
```bash
cd sagemaker-pipeline-demo
setup.bat
```

**Linux/Mac:**
```bash
cd sagemaker-pipeline-demo
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual Setup**

**Step 1: Create Virtual Environment (Recommended)**

To avoid conflicts with other Python projects, use a virtual environment:

**Windows:**
```bash
cd sagemaker-pipeline-demo
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
cd sagemaker-pipeline-demo
python -m venv .venv
source .venv/bin/activate
```

**Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

> **Note**: If you don't use a virtual environment, packages will be installed globally and may conflict with other projects.

### Run Pipeline Locally

```bash
python run_local.py
```

With custom parameters:
```bash
python run_local.py --n-estimators 200 --test-size 0.3 --accuracy-threshold 0.95
```

### Output Files

After execution, check the `outputs/` directory:

```
outputs/
├── model/
│   ├── model.pkl                    # Trained Random Forest model
│   └── training_metadata.json       # Training configuration and metrics
├── evaluation/
│   └── evaluation_results.json      # Evaluation metrics and classification report
└── pipeline_execution.json          # Pipeline execution summary
```

## Pipeline Steps

### Step 1: Training
- Loads Iris dataset (150 samples, 4 features, 3 classes)
- Trains Random Forest classifier
- Saves model and metadata

### Step 2: Evaluation
- Loads trained model
- Calculates accuracy, precision, recall, F1-score
- Determines if model meets threshold

### Step 3: Conditional Registration
- If accuracy ≥ threshold → Model approved for registration
- If accuracy < threshold → Model rejected

## GitHub Actions

The workflow automatically runs on push/PR:
- Validates pipeline definition
- Runs training and evaluation
- Executes full pipeline simulation
- Checks output files

## LocalStack (Optional)

To test with LocalStack AWS simulation:

```bash
# Start LocalStack
docker-compose up -d

# Wait for LocalStack to be ready
sleep 10

# Check LocalStack status
curl http://localhost:4566/_localstack/health

# Stop LocalStack
docker-compose down
```

**Note**: Full SageMaker Pipeline execution on LocalStack has limited support. Use `run_local.py` for complete workflow demonstration.

## Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Permission denied (outputs directory)
```bash
chmod -R 755 outputs/
```

### Issue: Model accuracy below threshold
- Adjust `--accuracy-threshold` parameter (e.g., `0.85` instead of `0.90`)
- Increase `--n-estimators` for better model performance
