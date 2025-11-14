# ML Pipeline with Evidently AI Monitoring

End-to-end machine learning pipeline for road accident risk prediction with comprehensive monitoring, deployment, and CI/CD automation.

## Project Overview

This project implements a complete MLOps pipeline for predicting road accident risk using Gradient Boosting Regressor. The pipeline includes data ingestion, validation, transformation, feature engineering, model training, evaluation, and monitoring with Evidently AI.

## Architecture

```mermaid
graph LR
    A[Data Ingestion] --> B[Data Validation]
    B --> C[Feature Engineering]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Monitoring]
    G --> H[MLflow Tracking]
    I[main.py] --> A
    J[airflow_dag.py] --> A
    K[app.py] --> L[Flask Web App]
    E --> L
```

## Pipeline Stages

### Stage 1: Data Ingestion
- Loads raw dataset from source
- Splits data into train and test sets
- Stores in artifacts directory

### Stage 2: Data Validation
- Validates schema and data types
- Checks for missing values
- Ensures data quality

### Stage 3: Feature Engineering
- Creates interaction features (lanes_speed, curvature_speed)
- Generates risk indicators (high_speed, few_lanes, no_signs)
- Builds categorical features (speed_category, curvature_category)

### Stage 4: Data Transformation
- Encodes categorical features using LabelEncoder
- Scales numerical features using StandardScaler
- Prepares data for model training

### Stage 5: Model Training
- Trains Gradient Boosting Regressor
- Performs hyperparameter tuning
- Saves trained model to artifacts

### Stage 6: Model Evaluation
- Evaluates model performance (MAE, RMSE, R2 Score)
- Logs metrics to MLflow
- Stores evaluation results

### Stage 7: Monitoring
- Generates data drift reports using Evidently AI
- Tracks model performance over time
- Creates interactive HTML dashboards

## Technologies Used

- **ML Framework**: scikit-learn
- **Monitoring**: Evidently AI
- **Experiment Tracking**: MLflow, DagsHub
- **Orchestration**: Apache Airflow
- **Web Framework**: Flask
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Kubernetes

## Project Structure

```
ML_PipeLine_Evidently/
├── src/heartpipeline/
│   ├── components/          # Pipeline components
│   ├── pipeline/            # Stage implementations
│   ├── config/              # Configuration management
│   ├── entity/              # Entity definitions
│   └── utils/               # Utility functions
├── config/                  # YAML configurations
├── artifacts/               # Generated artifacts
├── monitoring/              # Evidently monitoring
├── deployment/              # Kubernetes manifests
├── templates/               # Flask HTML templates
├── static/                  # CSS/JS/Images
├── .github/workflows/       # CI/CD pipelines
├── app.py                   # Flask web application
├── main.py                  # Pipeline executor
├── airflow_dag.py          # Airflow DAG definition
├── Dockerfile              # Container image
└── requirements.txt        # Python dependencies
```

## Features

### Web Application (app.py)
- Interactive prediction interface
- Real-time risk assessment (Low/Medium/High)
- Monitoring dashboard with drift reports
- REST API endpoints for predictions

### Orchestration (airflow_dag.py)
- Automated pipeline execution
- Task dependencies and scheduling
- Failure handling and retries

### Monitoring (Evidently AI)
- Data drift detection
- Model performance tracking
- Feature-level analysis
- Interactive visualizations

### MLflow Integration
- Experiment tracking
- Model versioning
- Metric logging
- DagsHub remote tracking

## Installation

```bash
git clone https://github.com/Abeshith/ML_PipeLine_Evidently.git
cd ML_PipeLine_Evidently
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Run Flask Web App
```bash
python app.py
```
Access at: http://localhost:5000

### Run with Airflow
```bash
airflow dags trigger ml_pipeline_dag
```

### Run with Docker
```bash
docker build -t ml-pipeline .
docker run -p 5000:5000 ml-pipeline
```

## CI/CD Pipeline

GitHub Actions workflow automates:
1. Code checkout
2. Docker image build
3. Trivy security scan
4. Push to DockerHub

## Kubernetes Deployment

```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/ingress.yaml
```

## Model Details

- **Algorithm**: Gradient Boosting Regressor
- **Features**: 12 input features + 10 engineered features
- **Target**: accident_risk (continuous)
- **Metrics**: MAE, RMSE, R2 Score

### Input Features
- road_type, num_lanes, curvature, speed_limit
- lighting, weather, road_signs_present, public_road
- time_of_day, holiday, school_season, num_reported_accidents

## API Endpoints

### Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "road_type": "highway",
  "speed_limit": 60,
  "num_lanes": 2,
  ...
}
```

### Dashboard
- `/` - Home page
- `/predict` - Prediction form
- `/dashboard` - Monitoring dashboard
- `/reports/drift` - Data drift report
- `/reports/performance` - Performance metrics

## Monitoring Dashboard

Access Evidently AI reports:
- Data Drift Analysis
- Feature Distribution Changes
- Model Performance Metrics
- MLflow Experiments: https://dagshub.com/abheshith7/ML-Pipeline-Evidently.mlflow

## Configuration

Edit `config/config.yaml` for pipeline settings:
- Artifact paths
- Model parameters
- Data sources

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request