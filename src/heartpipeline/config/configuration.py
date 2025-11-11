from heartpipeline.constants import *
from heartpipeline.utils.common import read_yaml, create_directories
from heartpipeline.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    MonitoringConfig
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_file=config.source_file,
            local_data_file=Path(config.local_data_file),
            sample_size=config.sample_size
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            data_dir=Path(config.data_dir),
            required_columns=list(schema.keys())
        )

        return data_validation_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering

        create_directories([config.root_dir])

        feature_engineering_config = FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            output_path=Path(config.output_path)
        )

        return feature_engineering_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            scaler_path=Path(config.scaler_path)
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        target_col = self.schema.target_column

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            target_column=target_col,
            params=params
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        target_col = self.schema.target_column

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            mlflow_uri=config.mlflow_uri,
            target_column=target_col
        )

        return model_evaluation_config

    def get_monitoring_config(self) -> MonitoringConfig:
        config = self.config.monitoring
        target_col = self.schema.target_column

        create_directories([config.root_dir])

        monitoring_config = MonitoringConfig(
            root_dir=Path(config.root_dir),
            reference_data_path=Path(config.reference_data_path),
            current_data_path=Path(config.current_data_path),
            model_path=Path(config.model_path),
            evidently_report_path=Path(config.evidently_report_path),
            target_column=target_col
        )

        return monitoring_config
