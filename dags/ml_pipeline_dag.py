from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model

with DAG(
    dag_id="ml_pipeline_dag",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "cicd"]
) as dag:

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data_task",
        python_callable=preprocess_data
    )

    train_model_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model_task",
        python_callable=evaluate_model
    )

    deploy_model_task = PythonOperator(
        task_id="deploy_model_task",
        python_callable=deploy_model
    )

    preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
