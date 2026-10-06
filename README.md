# Automated CI/CD and Airflow ML Pipeline Deployment

## Objective
Automate ML model testing, preprocessing, training, quality-gate validation,
deployment/promotion, Airflow orchestration, GitHub Actions CI/CD, and inference smoke testing.

## Pipeline
Preprocess -> Train -> Evaluate -> Quality Gate (Accuracy >= 0.85) -> Deploy -> Predict

## Project Structure
- `.github/workflows/ml_cicd.yml` - GitHub Actions CI/CD
- `dags/ml_pipeline_dag.py` - Airflow DAG
- `src/` - preprocessing, training, evaluation, deployment and prediction
- `tests/` - Pytest test modules
- `data/processed/` - generated processed data
- `models/` - generated model artifacts and metrics
- `production_model/` - promoted production artifacts

## Local Commands
```powershell
python -m pip install -r requirements.txt
pytest tests/ -v
python run_pipeline.py
python src/predict.py
```

## Airflow Commands
```powershell
airflow db migrate
airflow tasks test ml_pipeline_dag preprocess_data_task 2026-01-01
airflow tasks test ml_pipeline_dag train_model_task 2026-01-01
airflow tasks test ml_pipeline_dag evaluate_model_task 2026-01-01
airflow tasks test ml_pipeline_dag deploy_model_task 2026-01-01
```

## Quality Gate
Production deployment is allowed only when accuracy is at least `0.85`.

## GitHub
Push the project to a GitHub repository. GitHub Actions runs automatically on push/PR.
