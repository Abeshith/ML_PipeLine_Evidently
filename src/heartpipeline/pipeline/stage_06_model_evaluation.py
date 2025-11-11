import sys
from heartpipeline.config.configuration import ConfigurationManager
from heartpipeline.components.model_evaluation import ModelEvaluation
from heartpipeline.logging import logger
from heartpipeline.exception import CustomException


STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.evaluate()
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        metrics = obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<")
        logger.info(f"Evaluation metrics: {metrics}")
    except Exception as e:
        logger.exception(e)
        raise e
