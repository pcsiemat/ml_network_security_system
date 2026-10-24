"""
=====================================================================
 CONFIG ENTITY - Pipeline stage configuration
=====================================================================
This file does NOT create folders or run anything.
It only builds the PATHS (folder and file names) and the parameters
that each stage of the training pipeline will use.

Structure generated on each run:

    artifacts/<timestamp>/
    └── data_ingestion/
        ├── feature_store/raw.csv
        └── ingested/
            ├── train.csv
            └── test.csv

Fixed values (names, split ratio, collection) come from:
    ml_network_security_system/constant/training_pipeline
=====================================================================
"""

from datetime import datetime
import os
from ml_network_security_system.constant import training_pipeline


class TrainingPipelineConfig:
    """
    General pipeline configuration (shared by all stages).
    Defines the root folder for each run: artifacts/<timestamp>/
    and the folder where the final model is saved: final_model/
    """

    def __init__(self, timestamp=None):
        # Use the current date/time if no timestamp is given
        if timestamp is None:
            timestamp = datetime.now()

        # Format the timestamp to use it as a folder name (e.g. 10_18_2026_02_11_00)
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Pipeline name (from constants)
        self.pipeline_name = training_pipeline.PIPELINE_NAME

        # Base artifacts folder name (e.g. "artifacts")
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        # Root folder for this run: artifacts/<timestamp>/
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)

        # Folder where the final trained model will be saved
        self.model_dir = os.path.join("final_model")

        # Keep the timestamp string for later use
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Configuration for the DATA INGESTION stage.
    Defines where raw.csv, train.csv and test.csv are stored,
    plus the train/test split ratio and the MongoDB collection to read.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Main folder for this stage: artifacts/<timestamp>/data_ingestion/
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Raw data exported from MongoDB: .../data_ingestion/feature_store/raw.csv
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )

        # Training data after the split: .../data_ingestion/ingested/train.csv
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Testing data after the split: .../data_ingestion/ingested/test.csv
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Fraction of data used for testing (e.g. 0.2 = 20% test, 80% train)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        # Ramdom state for replicate
        self.random_state : int= training_pipeline.DATA_INGESTION_RANDOM_STATE

        # Target column used to stratify the train/test split
        self.target_column: str = training_pipeline.TARGET_COLUMN

        # MongoDB collection to read the data from
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # MongoDB database that contains the collection
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME



class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
        self.drift_threshold: float = training_pipeline.DATA_VALIDATION_DRIFT_THRESHOLD


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        self.cv_std_warning = training_pipeline.MODEL_TRAINER_CV_STD_WARNING

