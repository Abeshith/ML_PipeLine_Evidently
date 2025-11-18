from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from heartpipeline.logging import logger
from heartpipeline.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from heartpipeline.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from heartpipeline.pipeline.stage_03_feature_engineering import FeatureEngineeringTrainingPipeline
from heartpipeline.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from heartpipeline.pipeline.stage_05_model_trainer import ModelTrainerTrainingPipeline
from heartpipeline.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from heartpipeline.pipeline.stage_07_monitoring import ModelMonitoringPipeline

default_args = {
    'owner': 'abeshith',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_pipeline_evidently',
    default_args=default_args,
    description='ML Pipeline with Evidently AI Monitoring',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'evidently', 'monitoring']
)

def run_data_ingestion():
    logger.info("Starting Data Ingestion")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info("Data Ingestion completed")

def run_data_validation():
    logger.info("Starting Data Validation")
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()
    logger.info("Data Validation completed")

def run_feature_engineering():
    logger.info("Starting Feature Engineering")
    pipeline = FeatureEngineeringTrainingPipeline()
    pipeline.main()
    logger.info("Feature Engineering completed")

def run_data_transformation():
    logger.info("Starting Data Transformation")
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()
    logger.info("Data Transformation completed")

def run_model_trainer():
    logger.info("Starting Model Training")
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()
    logger.info("Model Training completed")

def run_model_evaluation():
    logger.info("Starting Model Evaluation")
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
    logger.info("Model Evaluation completed")

def run_monitoring():
    logger.info("Starting Model Monitoring")
    pipeline = ModelMonitoringPipeline()
    pipeline.main()
    logger.info("Model Monitoring completed")

# Define tasks
data_ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag
)

data_validation_task = PythonOperator(
    task_id='data_validation',
    python_callable=run_data_validation,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag
)

data_transformation_task = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    dag=dag
)

model_trainer_task = PythonOperator(
    task_id='model_trainer',
    python_callable=run_model_trainer,
    dag=dag
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag
)

monitoring_task = PythonOperator(
    task_id='monitoring',
    python_callable=run_monitoring,
    dag=dag
)

# Set task dependencies
data_ingestion_task >> data_validation_task >> feature_engineering_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task >> monitoring_task
