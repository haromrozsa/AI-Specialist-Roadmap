# Airflow ML Pipeline with LLM Summary Generation

## Session Summary: Built an end-to-end ML workflow orchestrated by Apache Airflow, integrating a Hugging Face LLM to automatically generate human-readable summaries of model performance—demonstrating modern ML automation with AI-powered insights.

## What I Did During the Session

1. **Designed Airflow DAG Architecture for ML Pipeline**:
   - Created 4-task sequential workflow: `load_data` → `train_model` → `evaluate_model` → `generate_llm_summary`
   - Used `PythonOperator` with task dependencies defined via `>>` operator for clean pipeline definition
   - Configured XCom (cross-communication) to pass metadata, training info, and evaluation results between tasks
   - Set `schedule_interval=None` for manual triggering, added retry logic (1 retry, 2-minute delay), and tagged DAG with `machine-learning`, `llm`, `wine-classification`

2. **Implemented ML Pipeline Tasks as Modular Python Functions**:
   - **data_loader.py**: Loads Wine dataset (178 samples, 13 features, 3 classes), splits 80/20 train/test, saves CSVs and metadata.json
   - **model_trainer.py**: Trains Random Forest (100 trees, max_depth=10) with scikit-learn, saves model as `wine_model.pkl`, logs training time and configuration
   - **model_evaluator.py**: Calculates accuracy/precision/recall/F1, generates sample predictions with confidence scores, saves evaluation_results.json
   - **llm_summarizer.py**: **Key innovation**—uses Hugging Face `flan-t5-small` to convert metrics into natural language summary for non-technical audiences

3. **Integrated Hugging Face LLM for Automated Insight Generation**:
   - Used `transformers.pipeline("text2text-generation")` with Google's `flan-t5-small` model for text summarization
   - Crafted prompt template combining model config, metrics, and sample predictions as context for LLM
   - Generated both LLM summary and structured manual summary for comparison and fallback
   - Saved outputs as `llm_summary.txt` (human-readable) and `llm_summary.json` (structured with prompt and metadata)
   - LLM output example: "The Random Forest model successfully classified wine samples into three categories with 97.2% accuracy..."

4. **Created Docker-based Airflow Deployment**:
   - Built custom Dockerfile extending `apache/airflow:2.8.1-python3.11` with ML dependencies (scikit-learn, transformers, torch)
   - Configured `docker-compose.yml` with 3 services: PostgreSQL (metadata DB), Airflow webserver (UI on port 8080), Airflow scheduler
   - Used volume mounts for `dags/`, `tasks/`, `outputs/`, and `logs/` to persist data and enable local development
   - Set up init service to auto-create admin user (username/password: `admin`/`admin`) and migrate database schema
   - Enabled `LocalExecutor` for single-machine task execution with proper health checks and restart policies

5. **Documented DAG Visualization and Usage Instructions**:
   - Created `DAG_VISUALIZATION.md` with Mermaid diagrams showing task flow, dependencies, XCom data passing, and execution timeline
   - Wrote comprehensive `USAGE.md` with quick start guide, troubleshooting tips, and advanced configuration examples
   - Provided example outputs (`EXAMPLE_llm_summary.txt`, `EXAMPLE_evaluation_results.json`) to demonstrate pipeline results
   - Documented resource requirements (4GB RAM, 2GB disk), expected execution time (~25-60 seconds), and file outputs (9 files including model, metrics, summaries)

## What I Learned

1. **Apache Airflow for ML Workflow Orchestration**:
   - Airflow DAGs are defined as Python code with tasks (nodes) and dependencies (edges) forming a directed acyclic graph
   - `PythonOperator` wraps Python functions as tasks, `>>` operator defines execution order, XCom enables data sharing between tasks via `ti.xcom_push()` / `ti.xcom_pull()`
   - Airflow separates **orchestration** (when/how tasks run) from **tracking** (what happened)—unlike MLflow which focuses on experiment logging
   - Key benefits: retry logic, task isolation, visual monitoring via web UI, scheduler handles execution automatically when dependencies are met

2. **LLM Integration for Automated Reporting**:
   - Hugging Face `pipeline("text2text-generation")` provides a simple API for loading and using LLMs like `flan-t5-small` (~80MB model)
   - Prompt engineering is critical: structured prompts with clear context (metrics, sample predictions) produce better summaries than raw data dumps
   - Text-to-text models like FLAN-T5 excel at summarization tasks—given "input: metrics", they generate "output: natural language explanation"
   - LLMs can transform technical ML outputs (accuracy=0.9722) into business-friendly insights ("97.2% accuracy, reliable for wine classification")
   - This pattern (ML pipeline → LLM summary) makes AI/ML results accessible to non-technical stakeholders without manual report writing

3. **Docker Compose for Airflow Deployment**:
   - Airflow requires 3+ services (webserver, scheduler, database), making Docker Compose ideal for local development and reproducibility
   - Multi-stage builds aren't needed here, but custom Dockerfile extends base image to add ML libraries (scikit-learn, transformers) not in default Airflow image
   - Volume mounts (`./dags:/opt/airflow/dags`) enable hot-reloading—edit DAG locally, scheduler auto-detects changes without rebuild
   - Health checks are essential: PostgreSQL `pg_isready`, webserver `curl /health`, scheduler `curl /health` prevent race conditions during startup
   - `LocalExecutor` uses PostgreSQL as backend (vs `SequentialExecutor` with SQLite)—supports parallel task execution within single machine

4. **XCom for Inter-Task Data Sharing**:
   - XCom (cross-communication) stores small data in Airflow's metadata database, accessible via task instance (`ti`) object
   - Pattern: `ti.xcom_push(key='metadata', value=dict)` in one task → `ti.xcom_pull(key='metadata', task_ids='load_data')` in downstream task
   - XCom is designed for **metadata and small payloads** (JSON-serializable), not large datasets—use file storage (S3, NFS) for big data
   - All tasks receive `**context` dict with `ti` (task instance), `ds` (execution date), `ts` (timestamp), etc.—enables access to runtime metadata
   - XCom enables clean separation: tasks write files to disk for models/data, use XCom for coordination metadata (paths, metrics, configs)

5. **Modern ML Automation Stack (2025-Ready)**:
   - Combining **Airflow** (orchestration) + **Hugging Face** (LLM) + **Docker** (deployment) creates production-grade ML automation
   - This pattern scales: add more tasks (data validation, model comparison, deployment), replace local Docker with cloud (AWS MWAA, GCP Composer)
   - LLMs are becoming infrastructure—not just chat, but automated reporting, data validation, anomaly explanation, code generation in pipelines
   - Airflow's Python-first approach enables rapid prototyping: write function, wrap in `PythonOperator`, define dependencies, deploy
   - Portfolio value: demonstrates understanding of ML **engineering** (not just modeling)—orchestration, automation, deployment, monitoring

This session provided hands-on experience orchestrating ML workflows with Apache Airflow, integrating Hugging Face LLMs for automated insight generation, deploying multi-service applications with Docker Compose, and building production-ready ML automation pipelines that combine classical ML with modern generative AI.
