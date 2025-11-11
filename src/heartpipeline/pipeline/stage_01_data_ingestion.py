import sys
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            output_file = data_ingestion.ingest()
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
            logger.info(f"Data ingestion output: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
