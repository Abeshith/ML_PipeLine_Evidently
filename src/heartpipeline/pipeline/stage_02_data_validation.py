import sys
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.data_validation import DataValidation

STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate()
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
            logger.info(f"Validation status: {'PASSED' if validation_status else 'FAILED'}")
            
            return validation_status
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
