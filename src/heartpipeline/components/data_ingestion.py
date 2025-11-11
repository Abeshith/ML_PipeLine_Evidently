import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_from_kaggle(self):
        """Download dataset from Kaggle using Kaggle API"""
        try:
            logger.info("Downloading dataset from Kaggle...")
            
            # Kaggle competition name
            competition_name = "playground-series-s5e10"
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            # Download using kaggle CLI
            download_dir = os.path.dirname(self.config.local_data_file)
            cmd = f"kaggle competitions download -c {competition_name} -p {download_dir}"
            
            subprocess.run(cmd, shell=True, check=True)
            
            # Unzip the downloaded file
            zip_file = os.path.join(download_dir, f"{competition_name}.zip")
            if os.path.exists(zip_file):
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                logger.info(f"Dataset extracted to {download_dir}")
                
                # Remove zip file
                os.remove(zip_file)
            
            logger.info("Dataset downloaded successfully from Kaggle")
            
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self) -> pd.DataFrame:
        """Load data from local file, download if not exists"""
        try:
            # Check if final processed file exists
            if os.path.exists(self.config.local_data_file):
                logger.info(f"Loading existing data from {self.config.local_data_file}")
                df = pd.read_csv(self.config.local_data_file)
                logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
                return df
            
            # Check for train.csv from Kaggle
            train_file = os.path.join(os.path.dirname(self.config.local_data_file), self.config.source_file)
            
            if not os.path.exists(train_file):
                logger.info(f"Train file not found. Downloading from Kaggle...")
                self.download_from_kaggle()
            
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Train file not found at {train_file} after download attempt")
            
            logger.info(f"Loading data from {train_file}")
            df = pd.read_csv(train_file)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def process_data(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """Process and sample data if needed"""
        try:
            logger.info(f"Original dataset size: {len(df)} rows")
            
            # Sample data if sample_size is specified and less than total rows
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled data to {len(df)} rows")
            
            # Save processed data
            df.to_csv(self.config.local_data_file, index=False)
            logger.info(f"Processed data saved to {self.config.local_data_file}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def ingest(self) -> Path:
        """Main ingestion method"""
        try:
            logger.info("Starting data ingestion...")
            
            # Load data
            df = self.load_data()
            
            # Process and sample data
            df = self.process_data(df, sample_size=self.config.sample_size)
            
            logger.info(f"Data ingestion completed. Output: {self.config.local_data_file}")
            return self.config.local_data_file
            
        except Exception as e:
            raise CustomException(e, sys)
