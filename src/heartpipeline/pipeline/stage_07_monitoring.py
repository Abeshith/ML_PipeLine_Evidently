import sys
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.monitoring import ModelMonitoring
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException


STAGE_NAME = "Model Monitoring Stage"

class ModelMonitoringPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            monitoring_config = config.get_monitoring_config()
            
            monitoring = ModelMonitoring(config=monitoring_config)
            report_path = monitoring.generate_report()
            
            return report_path
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelMonitoringPipeline()
        report_path = obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
        logger.info(f"Evidently report: {report_path}")
    except Exception as e:
        logger.exception(e)
        raise e
