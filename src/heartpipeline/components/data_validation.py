import os
import sys
import pandas as pd
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self, df: pd.DataFrame) -> bool:
        try:
            all_cols = list(df.columns)
            missing_cols = [col for col in self.config.required_columns if col not in all_cols]
            
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                return False
            
            logger.info("All required columns present")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_nulls(self, df: pd.DataFrame) -> bool:
        try:
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                logger.warning(f"Found {total_nulls} null values")
                logger.warning(f"Null counts per column:\n{null_counts[null_counts > 0]}")
                return False
            
            logger.info("No null values found")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_types(self, df: pd.DataFrame) -> bool:
        try:
            logger.info("Data types validation:")
            logger.info(f"\n{df.dtypes}")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)

    def save_validation_status(self, status: str):
        try:
            status_file = os.path.join(self.config.root_dir, self.config.STATUS_FILE)
            with open(status_file, 'w') as f:
                f.write(f"Validation Status: {status}")
            
            logger.info(f"Validation status saved to {status_file}")
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate(self) -> bool:
        try:
            logger.info("Starting data validation...")
            
            df = pd.read_csv(self.config.data_dir)
            logger.info(f"Loaded data shape: {df.shape}")
            
            cols_valid = self.validate_columns(df)
            nulls_valid = self.validate_nulls(df)
            types_valid = self.validate_data_types(df)
            validation_status = all([cols_valid, nulls_valid, types_valid])
            
            if validation_status:
                self.save_validation_status("PASSED")
                logger.info("Data validation PASSED")
            else:
                self.save_validation_status("FAILED")
                logger.error("Data validation FAILED")
            
            return validation_status
            
        except Exception as e:
            raise CustomException(e, sys)
