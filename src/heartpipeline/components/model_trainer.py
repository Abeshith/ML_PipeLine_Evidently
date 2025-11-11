import os
import sys
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def load_data(self):
        """Load train and test data"""
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
            
            # Separate features and target
            X_train = train_data.drop(self.config.target_column, axis=1)
            y_train = train_data[self.config.target_column]
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]
            
            logger.info(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, y_true, y_pred) -> dict:
        """Evaluate model performance"""
        try:
            import numpy as np
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, X_test, y_train, y_test, model_name: str, model, params: dict):
        """Train a model and log to MLflow"""
        try:
            logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=model_name):
                # Set model parameters
                model.set_params(**params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate
                train_metrics = self.evaluate_model(y_train, y_train_pred)
                test_metrics = self.evaluate_model(y_test, y_test_pred)
                
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metric("train_rmse", train_metrics['rmse'])
                mlflow.log_metric("train_mae", train_metrics['mae'])
                mlflow.log_metric("train_r2", train_metrics['r2_score'])
                mlflow.log_metric("test_rmse", test_metrics['rmse'])
                mlflow.log_metric("test_mae", test_metrics['mae'])
                mlflow.log_metric("test_r2", test_metrics['r2_score'])
                
                # Skip model logging to DagHub (unsupported endpoint)
                # mlflow.sklearn.log_model(model, model_name)
                
                logger.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.4f}, Test R2: {test_metrics['r2_score']:.4f}")
                
                return model, test_metrics['r2_score']
                
        except Exception as e:
            raise CustomException(e, sys)

    def train(self):
        """Main training method"""
        try:
            logger.info("Starting model training...")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/abheshith7/ML-Pipeline-Evidently.mlflow"))
            mlflow.set_experiment("Road Accident Risk Prediction")
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Define models
            models = {
                'RandomForest': (RandomForestRegressor(random_state=42), 
                               self.config.params.get('RandomForestRegressor', {})),
                'GradientBoosting': (GradientBoostingRegressor(random_state=42), 
                                   self.config.params.get('GradientBoostingRegressor', {})),
                'XGBoost': (XGBRegressor(random_state=42, objective='reg:squarederror'), 
                          self.config.params.get('XGBRegressor', {})),
                'Ridge': (Ridge(random_state=42), 
                        self.config.params.get('Ridge', {}))
            }
            
            # Train all models
            best_model = None
            best_score = -float('inf')
            best_model_name = None
            
            for model_name, (model, params) in models.items():
                trained_model, score = self.train_model(
                    X_train, X_test, y_train, y_test,
                    model_name, model, params
                )
                
                if score > best_score:
                    best_score = score
                    best_model = trained_model
                    best_model_name = model_name
            
            # Save best model
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Best model: {best_model_name} with R2 Score: {best_score:.4f}")
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
