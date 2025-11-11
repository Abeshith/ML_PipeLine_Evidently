import sys
import dagshub
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.model_trainer import ModelTrainer

dagshub.init(repo_owner='abheshith7', repo_name='ML-Pipeline-Evidently', mlflow=True)

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
            logger.info(f"Model saved at: {model_trainer_config.root_dir}/{model_trainer_config.model_name}")
            
            return model_trainer_config.root_dir
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
