import os
import sys
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        """Load trained model"""
        try:
            with open(self.config.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {self.config.model_path}")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_test_data(self):
        """Load test data"""
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]
            logger.info(f"Test data loaded. Shape: {X_test.shape}")
            return X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def calculate_metrics(self, y_true, y_pred) -> dict:
        """Calculate evaluation metrics"""
        try:
            import numpy as np
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = (abs((y_true - y_pred) / y_true).mean()) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape
            }
            
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def save_metrics(self, metrics: dict):
        """Save metrics to file"""
        try:
            with open(self.config.metric_file_name, 'w') as f:
                f.write("Model Evaluation Metrics\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Mean Squared Error (MSE): {metrics['mse']:.6f}\n")
                f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}\n")
                f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
                f.write(f"R2 Score: {metrics['r2_score']:.6f}\n")
                f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n\n")
                f.write("=" * 50 + "\n")
                f.write("Model Performance Interpretation:\n")
                if metrics['r2_score'] > 0.8:
                    f.write(" Excellent: Model explains >80% of variance\n")
                elif metrics['r2_score'] > 0.6:
                    f.write(" Good: Model explains >60% of variance\n")
                elif metrics['r2_score'] > 0.4:
                    f.write("~ Fair: Model explains >40% of variance\n")
                else:
                    f.write(" Poor: Model needs improvement\n")
            
            logger.info(f"Metrics saved to {self.config.metric_file_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def log_to_mlflow(self, metrics: dict):
        """Log metrics to MLflow"""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment("Road Accident Risk Prediction")
            
            with mlflow.start_run(run_name="Model Evaluation"):
                mlflow.log_metrics(metrics)
                logger.info("Metrics logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")

    def evaluate(self) -> dict:
        """Main evaluation method"""
        try:
            logger.info("Starting model evaluation...")
            
            # Load model and data
            model = self.load_model()
            X_test, y_test = self.load_test_data()
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Log metrics
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R2 Score: {metrics['r2_score']:.4f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            
            # Save metrics
            self.save_metrics(metrics)
            
            # Log to MLflow
            self.log_to_mlflow(metrics)
            
            logger.info("Model evaluation completed")
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
