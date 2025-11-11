import sys
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.feature_engineering import FeatureEngineering

STAGE_NAME = "Feature Engineering Stage"


class FeatureEngineeringTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            
            config_manager = ConfigurationManager()
            feature_engineering_config = config_manager.get_feature_engineering_config()
            feature_engineering = FeatureEngineering(config=feature_engineering_config)
            feature_engineering.engineer_features()
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
            logger.info(f"Feature engineering output: {feature_engineering_config.output_path}")
            
            return feature_engineering_config.output_path
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = FeatureEngineeringTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
