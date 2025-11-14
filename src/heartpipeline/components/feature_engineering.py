import os
import sys
import pandas as pd
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.entity.config_entity import FeatureEngineeringConfig


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Creating interaction features...")
            
            df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
            
            df['curvature_speed'] = df['curvature'] * df['speed_limit']
            df['lanes_curvature'] = df['num_lanes'] * df['curvature']
            
            logger.info("Interaction features created")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Creating risk indicator features...")
            
            df['high_speed'] = (df['speed_limit'] > df['speed_limit'].median()).astype(int)
            
            df['high_curvature'] = (df['curvature'] > df['curvature'].median()).astype(int)
            
            df['few_lanes'] = (df['num_lanes'] <= 2).astype(int)
            
            df['no_signs'] = (df['road_signs_present'] == False).astype(int)
            df['holiday_risk'] = (df['holiday'] == True).astype(int)
            
            logger.info("Risk indicator features created")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Creating categorical features...")
            
            df['speed_category'] = pd.cut(df['speed_limit'], 
                                         bins=[0, 40, 60, 80, 120], 
                                         labels=['low', 'medium', 'high', 'very_high'])
            df['curvature_category'] = pd.cut(df['curvature'], 
                                              bins=[-0.1, 0.3, 0.6, 1.0], 
                                              labels=['low', 'medium', 'high'])
            
            logger.info("Categorical features created")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def engineer_features(self) -> pd.DataFrame:
        try:
            logger.info("Starting feature engineering...")
            
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded data shape: {df.shape}")
            
            df = self.create_interaction_features(df)
            df = self.create_risk_indicators(df)
            df = self.create_categorical_features(df)
            df.to_csv(self.config.output_path, index=False)
            logger.info(f"Feature engineering completed. Output shape: {df.shape}")
            logger.info(f"Engineered data saved to {self.config.output_path}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
