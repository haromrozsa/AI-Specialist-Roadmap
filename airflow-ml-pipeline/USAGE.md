# Usage Guide: Airflow ML Pipeline

This guide explains how to run the ML pipeline with LLM summary generation using Apache Airflow.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **At least 4GB RAM** available for Docker
- **~2GB disk space** for Docker images and dependencies

## Quick Start

### 1. Build and Start Airflow

```bash
cd airflow-ml-pipeline

# Build the custom Docker image (includes ML dependencies)
docker-compose build

# Start Airflow services
docker-compose up -d
```

This will start:
- PostgreSQL database
- Airflow webserver (port 8080)
- Airflow scheduler

**Wait ~30 seconds** for services to initialize.

### 2. Access Airflow UI

1. Open browser: `http://localhost:8080`
2. Login credentials:
   - **Username**: `admin`
   - **Password**: `admin`

### 3. Run the Pipeline

In the Airflow UI:

1. Find the DAG named **`ml_pipeline_with_llm`**
2. Click the **toggle** to enable it (if paused)
3. Click the **▶ Play button** (Trigger DAG)
4. Wait for execution (~30-60 seconds)

### 4. Monitor Execution

- Click on the DAG name to view details
- **Graph View**: Shows task dependencies and status
- **Grid View**: Shows historical runs
- Click on individual tasks to view logs

### 5. View Results

**Option A: Via Airflow UI**
- Navigate to task: `generate_llm_summary`
- Click **Logs** to see the LLM-generated summary

**Option B: Via File System**
```bash
# View all outputs
ls -lh outputs/

# View LLM summary
cat outputs/llm_summary.txt

# View evaluation metrics
cat outputs/evaluation_results.json
```

## Output Files

After successful execution, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `metadata.json` | Dataset information (size, features, classes) |
| `X_train.csv`, `y_train.csv` | Training data |
| `X_test.csv`, `y_test.csv` | Test data |
| `wine_model.pkl` | Trained Random Forest model |
| `training_info.json` | Training configuration and metrics |
| `evaluation_results.json` | Model performance metrics and predictions |
| `llm_summary.txt` | **LLM-generated summary** (main deliverable) |
| `llm_summary.json` | Structured summary with prompt and metadata |

## Pipeline Tasks

The DAG consists of 4 tasks that run sequentially:

```
load_data → train_model → evaluate_model → generate_llm_summary
```

### Task 1: `load_data`
- Loads Wine dataset (178 samples, 13 features, 3 classes)
- Splits into 80% train / 20% test
- Duration: ~2 seconds

### Task 2: `train_model`
- Trains Random Forest (100 trees, max_depth=10)
- Saves model as `wine_model.pkl`
- Duration: ~1 second

### Task 3: `evaluate_model`
- Calculates accuracy, precision, recall, F1-score
- Generates sample predictions with confidence scores
- Duration: ~1 second

### Task 4: `generate_llm_summary`
- Uses Hugging Face `flan-t5-small` model
- Generates natural language summary from metrics
- Duration: ~20-30 seconds (downloads model on first run)

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Clean restart
docker-compose down
docker-compose up -d
```

### DAG not visible

- Wait 30 seconds for scheduler to detect DAG
- Check logs: `docker-compose logs airflow-scheduler`
- Verify file: `ls -l dags/ml_pipeline_dag.py`

### Task failed

1. Click on failed task in UI
2. View **Logs** tab
3. Check error message
4. Click **Clear** to retry

### Out of memory

```bash
# Increase Docker memory allocation
# Docker Desktop → Settings → Resources → Memory → 6GB
```

### Permission errors on Windows

```bash
# Set AIRFLOW_UID in .env file
echo "AIRFLOW_UID=50000" > .env
```

## Stopping Airflow

```bash
# Stop services (keeps data)
docker-compose stop

# Stop and remove containers (keeps volumes)
docker-compose down

# Complete cleanup (removes all data)
docker-compose down -v
```

## Re-running the Pipeline

You can trigger the pipeline multiple times:

1. Each run creates a new **DAG Run ID** with timestamp
2. Outputs are **overwritten** each time
3. Old logs are preserved in Airflow UI
4. To compare runs, copy `outputs/` before re-running

## Advanced Usage

### View XCom Data

XCom is used to pass data between tasks:

1. Click on a task → **XCom** tab
2. View data passed from previous tasks
3. Example keys: `metadata`, `training_info`, `evaluation_results`

### Modify Pipeline

Edit files and restart:

```bash
# Edit task logic
nano tasks/llm_summarizer.py

# Edit DAG configuration
nano dags/ml_pipeline_dag.py

# Restart scheduler to pick up changes
docker-compose restart airflow-scheduler
```

### Change LLM Model

Edit `tasks/llm_summarizer.py`:

```python
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # Larger model
    max_length=512,
    device=-1
)
```

### Schedule Automatic Runs

Edit `dags/ml_pipeline_dag.py`:

```python
dag = DAG(
    'ml_pipeline_with_llm',
    schedule_interval='@daily',  # Run daily at midnight
    ...
)
```

## Example LLM Output

```
The Random Forest model successfully classified wine samples into three
categories with 97.2% accuracy. The model was trained on 142 wine samples
and tested on 36 samples, correctly predicting 35 out of 36 wines. The
model uses 13 chemical features such as alcohol content, acidity, and
color intensity to distinguish between class_0, class_1, and class_2 wines.
With balanced precision (97.3%) and recall (97.2%) scores, this classifier
demonstrates reliable performance for automated wine quality assessment tasks.
```

## Resource Usage

- **Disk**: ~1.5GB (Docker images + models)
- **RAM**: ~2-3GB during execution
- **CPU**: Uses all available cores for Random Forest training
- **First run**: Downloads Hugging Face model (~150MB)

## Architecture

```
Docker Container (apache/airflow:2.8.1)
├── PostgreSQL (metadata database)
├── Airflow Webserver (UI on port 8080)
├── Airflow Scheduler (executes tasks)
└── Python 3.11 + ML libraries
    ├── scikit-learn (Random Forest)
    ├── transformers (Hugging Face LLM)
    └── PyTorch (LLM backend)
```

## Next Steps

- Experiment with different ML models (SVM, Gradient Boosting)
- Try different LLM models (GPT-2, BART, Llama)
- Add visualization tasks (confusion matrix, feature importance)
- Deploy to cloud (AWS MWAA, Google Cloud Composer)
- Add email notifications on pipeline completion
- Integrate with MLflow for experiment tracking
