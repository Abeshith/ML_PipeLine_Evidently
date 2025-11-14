from heartpipeline.logging import logger
from heartpipeline.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from heartpipeline.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from heartpipeline.pipeline.stage_03_feature_engineering import FeatureEngineeringTrainingPipeline
from heartpipeline.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from heartpipeline.pipeline.stage_05_model_trainer import ModelTrainerTrainingPipeline
from heartpipeline.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from heartpipeline.pipeline.stage_07_monitoring import ModelMonitoringPipeline


STAGE_NAME = "Complete ML Pipeline"

if __name__ == "__main__":
    try:
        logger.info("=" * 80)
        logger.info(f">>>>>> Starting {STAGE_NAME} <<<<<<")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Data Ingestion")
        logger.info("=" * 80)
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info("=" * 80)
        logger.info("STAGE 1: Data Ingestion - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 2: Data Validation")
        logger.info("=" * 80)
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info("=" * 80)
        logger.info("STAGE 2: Data Validation - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 3: Feature Engineering")
        logger.info("=" * 80)
        feature_engineering = FeatureEngineeringTrainingPipeline()
        feature_engineering.main()
        logger.info("=" * 80)
        logger.info("STAGE 3: Feature Engineering - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 4: Data Transformation")
        logger.info("=" * 80)
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info("=" * 80)
        logger.info("STAGE 4: Data Transformation - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 5: Model Training")
        logger.info("=" * 80)
        model_trainer = ModelTrainerTrainingPipeline()
        model_trainer.main()
        logger.info("=" * 80)
        logger.info("STAGE 5: Model Training - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 6: Model Evaluation")
        logger.info("=" * 80)
        model_evaluation = ModelEvaluationPipeline()
        metrics = model_evaluation.main()
        logger.info(f"Evaluation Metrics: R2={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        logger.info("=" * 80)
        logger.info("STAGE 6: Model Evaluation - COMPLETED\n")
        
        logger.info("=" * 80)
        logger.info("STAGE 7: Model Monitoring")
        logger.info("=" * 80)
        monitoring = ModelMonitoringPipeline()
        report_path = monitoring.main()
        logger.info(f"Monitoring Report: {report_path}")
        logger.info("=" * 80)
        logger.info("STAGE 7: Model Monitoring - COMPLETED\n")
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nPipeline Summary:")
        logger.info("  Stage 1: Data Ingestion")
        logger.info("  Stage 2: Data Validation")
        logger.info("  Stage 3: Feature Engineering")
        logger.info("  Stage 4: Data Transformation")
        logger.info("  Stage 5: Model Training")
        logger.info("  Stage 6: Model Evaluation")
        logger.info("  Stage 7: Model Monitoring")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.exception("Pipeline execution failed!")
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        raise e
