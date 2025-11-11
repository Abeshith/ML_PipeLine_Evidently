import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoders = {}

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder"""
        try:
            logger.info("Encoding categorical features...")
            
            categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                              'speed_category', 'curvature_category']
            
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Encoded {col}")
            
            # Save label encoders
            encoders_path = os.path.join(self.config.root_dir, 'label_encoders.pkl')
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            logger.info(f"Label encoders saved to {encoders_path}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets"""
        try:
            logger.info("Splitting data into train and test sets...")
            
            # Drop id column if exists
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Separate features and target
            if 'accident_risk' in df.columns:
                X = df.drop('accident_risk', axis=1)
                y = df['accident_risk']
            elif 'target' in df.columns:
                X = df.drop('target', axis=1)
                y = df['target']
            else:
                raise ValueError("Target column not found")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise CustomException(e, sys)

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Scale features using StandardScaler"""
        try:
            logger.info("Scaling features...")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Save scaler
            with open(self.config.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {self.config.scaler_path}")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self):
        """Main transformation method"""
        try:
            logger.info("Starting data transformation...")
            
            # Load data
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded data shape: {df.shape}")
            
            # Encode categorical features
            df = self.encode_categorical_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(df)
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Combine features and target
            train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
            test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
            
            # Save train and test data
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Train data saved to {self.config.train_data_path}")
            logger.info(f"Test data saved to {self.config.test_data_path}")
            logger.info("Data transformation completed")
            
        except Exception as e:
            raise CustomException(e, sys)
