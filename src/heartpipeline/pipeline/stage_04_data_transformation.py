import sys
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.transform()
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
            logger.info(f"Train data: {data_transformation_config.train_data_path}")
            logger.info(f"Test data: {data_transformation_config.test_data_path}")
            
            return data_transformation_config.train_data_path, data_transformation_config.test_data_path
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
