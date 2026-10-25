import sys

from ml_network_security_system.components.data_ingestion import DataIngestion
from ml_network_security_system.components.data_validation import DataValidation
from ml_network_security_system.components.data_transformation import DataTransformation
from ml_network_security_system.components.model_trainer  import ModelTrainer
from ml_network_security_system.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging


def run_pipeline():
    # One config for the whole run → one timestamp folder for all stages
    training_pipeline_config = TrainingPipelineConfig()
    print(f"▶ Run: {training_pipeline_config.artifact_dir}")

    # ── Stage 1: Data Ingestion ──
    logging.info("Initiate the data ingestion")
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion_artifact = DataIngestion(data_ingestion_config).initiate_data_ingestion()
    logging.info("Data ingestion completed")
    print("✔ Data ingestion completed")

    # ── Stage 2: Data Validation ──
    logging.info("Initiate the data validation")
    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation_artifact = DataValidation(
        data_ingestion_artifact, data_validation_config
    ).initiate_data_validation()
    logging.info("Data validation completed")

    if not data_validation_artifact.validation_status:
        message = (
            "Data drift detected. Pipeline stopped before transformation.\n"
            f"  Report: {data_validation_artifact.drift_report_file_path}\n"
            f"  Data:   {data_validation_artifact.invalid_train_file_path}"
        )
        logging.warning(message)
        print(f"⚠ {message}")
        return False
    print("✔ Data validation passed (no drift)")

    # ── Stage 3: Data Transformation ──
    logging.info("Initiate the data transformation")
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation_artifact = DataTransformation(
        data_validation_artifact, data_transformation_config
    ).initiate_data_transformation()
    logging.info("Data transformation completed")
    print("✔ Data transformation completed")
    print(f"  Train array: {data_transformation_artifact.transformed_train_file_path}")
    print(f"  Test array:  {data_transformation_artifact.transformed_test_file_path}")


    # ── Stage 4: Model training ──
    logging.info("Initiate the model training")
    model_trainer_config=ModelTrainerConfig(training_pipeline_config)
    model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
    model_trainer_artifact=model_trainer.initiate_model_trainer()

    logging.info("Model training completed")
    print("✔ Model training completed")
    print(f"  Train array: {data_transformation_artifact.transformed_train_file_path}")
    print(f"  Test array:  {data_transformation_artifact.transformed_test_file_path}")


    return True


if __name__ == "__main__":
    try:
        ok = run_pipeline()
        sys.exit(0 if ok else 1)
    except Exception as e:
        error = e if isinstance(e, AppException) else AppException(e, sys)
        logging.error(error)
        print(f"✖ Pipeline failed:\n  {error}")
        sys.exit(1)