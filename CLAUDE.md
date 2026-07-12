# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **learning and portfolio monorepo**, not an application. Each top-level directory is a self-contained, independently runnable demo of one AI/ML concept — there is no shared package, no root `requirements.txt`, no root build, and no test framework. Directories do not import from each other. Treat each one as its own mini-project.

Two distinct kinds of code live here, and they should be held to different standards:

- **Teaching demos** (`cnn/`, `gradient_descent/`, `linear_regression/`, `logistic_regression/`, `regularization/`, `rnn/`, `lstm/`, `feedforward_neural_networks*/`, `hugging_face*/`, `langchain*/`, `langgraph*/`, `onnx/`, `yolo_basics/`): single scripts that print results and often plot. Verbose comments are intentional and pedagogical. Missing tests, type hints, logging, and error handling are deliberate, not defects.
- **Production-shaped projects** (`fastapi-production/`, `fastapi-onnx/`, `yolo-api/`, `springboot-ml-api/`, `onnx-java/`, `serverless-ai-inference/`, `sagemaker-pipeline-demo/`, `airflow-ml-pipeline/`, `mlflow-experiment-tracking/`, `multi-service-docker/`): these deliberately demonstrate production practice — validation, structured logging, error handling, health checks, IaC, CI. Normal production standards apply here.

Most directories carry a `README.MD` documenting what was learned and why; several also have a `USAGE.md` with the run instructions. Read those before changing a directory — they are the spec.

## Running things

Everything runs **from inside its own directory**. There is no root entry point.

Plain demo scripts (no dependency file — install into whatever environment you have):

```bash
cd cnn && python CNNClassification.py
```

Projects with pinned dependencies (`airflow-ml-pipeline/`, `mlflow-experiment-tracking/`, `sagemaker-pipeline-demo/`, `serverless-ai-inference/`, `yolo-api/`):

```bash
cd sagemaker-pipeline-demo
pip install -r requirements.txt
python run_local.py --n-estimators 200 --test-size 0.3 --accuracy-threshold 0.95
```

FastAPI services:

```bash
cd yolo-api && uvicorn app.main:app --host 0.0.0.0 --port 8000
cd fastapi-production && python train.py && uvicorn main:app --port 8000
```

Java (`onnx-java/`, `springboot-ml-api/`):

```bash
mvn clean package
mvn spring-boot:run      # springboot-ml-api only
```

Docker-based projects (`multi-service-docker/`, `airflow-ml-pipeline/`, `serverless-ai-inference/`, `sagemaker-pipeline-demo/`):

```bash
docker-compose up --build
```

## Testing

There is **no pytest suite and no test runner**. Files named `test_*.py` are not unit tests:

- `fastapi-production/test_api.py` — a live HTTP script hitting `http://127.0.0.1:8000`; the server must already be running. Run it with `python test_api.py`.
- `serverless-ai-inference/local_test.py`, `test_localstack.py` — manual scripts for offline/LocalStack exercise.
- `airflow-ml-pipeline/test_pipeline_standalone.py` — a standalone runner.

Do not assume `pytest` will discover or pass anything. If verification is needed, run the relevant script and read its output.

## CI

`.github/workflows/validate-sagemaker.yml` is the only workflow, and it covers **only `sagemaker-pipeline-demo/`**. On push/PR to `main` it installs that project's requirements, runs `src/train.py` and `src/evaluate.py`, imports `pipeline.create_pipeline()` against dummy AWS credentials to validate the pipeline definition, runs `run_local.py`, and asserts the expected output files exist. Nothing else in the repo is covered by CI — changes elsewhere are unverified by automation.

## Repository gotchas

- **`.gitignore` contains `**/*.txt`**, which silently ignores every `requirements.txt`. The handful that are tracked were force-added. When adding a new project's dependency file you must `git add -f path/requirements.txt` or it will appear to commit and simply not exist for anyone else — including CI.
- `sagemaker-pipeline-demo/.venv/` exists on disk (ignored by that directory's own `.gitignore`). Exclude it from searches; a naive `find`/`grep` across the repo will drown in site-packages.
- Work has historically been committed directly to `main`.

## Code review guidance

When reviewing changes in this repository:

- Prioritize **ML/statistical correctness**: data leakage (scaler or encoder fitted before the train/test split), wrong loss for the task, mis-shaped tensors, metrics computed on the wrong split, thresholds that silently do nothing.
- Prioritize **bugs that break the documented behaviour** in the directory's `README.MD`/`USAGE.md`.
- Always flag **security issues**: hardcoded credentials or AWS keys, unsafe `pickle`/`eval` on untrusted input, path traversal in file-serving endpoints, secrets committed to the repo.
- In **teaching demos**, do not flag missing tests, missing type hints, absent logging/retry/monitoring, or explanatory comments as problems. That feedback is noise here.
- In **production-shaped projects**, do apply those standards.
