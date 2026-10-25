"""
=====================================================================
 DATA VALIDATION - Stage 2 of the training pipeline
=====================================================================
INPUT  : DataIngestionArtifact (paths of ingested train.csv / test.csv)
         DataValidationConfig  (where to save results)
CHECKS : 1. number of columns      → len(schema["columns"]) == 31
         2. numerical columns exist → every schema["numerical_columns"] by name
         3. data drift              → KS test per column (p-value < 0.05 = drift)
OUTPUT : DataValidationArtifact
         no drift → data saved in validated/, valid_* paths set
         drift    → data saved in invalid/,  invalid_* paths set
=====================================================================
"""

import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from ml_network_security_system.constant.training_pipeline import SCHEMA_FILE_PATH
from ml_network_security_system.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from ml_network_security_system.entity.config_entity import DataValidationConfig
from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging
from ml_network_security_system.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config  = data_validation_config
            self._schema_config          = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise AppException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AppException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Check the dataframe has the same number of columns as the schema (31)."""
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise AppException(e, sys)

    def validate_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """Check every numerical column from the schema exists, by name."""
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            missing = [col for col in numerical_columns if col not in dataframe.columns]
            if missing:
                logging.info(f"Missing numerical columns: {missing}")
                return False
            return True
        except Exception as e:
            raise AppException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold= 0.05) -> bool:
        """
        KS test per column. Writes report.yaml.
        Returns True if NO drift was found in any column, False otherwise.
        """
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            ## read the data from the ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path  = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe  = DataValidation.read_data(test_file_path)

            error_message = ""

            ## 1. validate number of columns
            if not self.validate_number_of_columns(dataframe=train_dataframe):
                error_message += "Train dataframe does not contain all columns.\n"
            if not self.validate_number_of_columns(dataframe=test_dataframe):
                error_message += "Test dataframe does not contain all columns.\n"

            ## 2. validate numerical columns exist
            if not self.validate_numerical_columns_exist(dataframe=train_dataframe):
                error_message += "Numerical columns are missing in train dataframe.\n"
            if not self.validate_numerical_columns_exist(dataframe=test_dataframe):
                error_message += "Numerical columns are missing in test dataframe.\n"

            ## stop if any schema check failed
            if error_message:
                raise Exception(error_message)

            ## 3. check data drift
            status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe,
                threshold=self.data_validation_config.drift_threshold)

            ## save to validated/ or invalid/ depending on the result
            cfg = self.data_validation_config
            if status:
                train_out, test_out = cfg.valid_train_file_path, cfg.valid_test_file_path
                logging.info("No drift found. Saving data to validated/")
            else:
                train_out, test_out = cfg.invalid_train_file_path, cfg.invalid_test_file_path
                logging.info("Drift found. Saving data to invalid/")

            os.makedirs(os.path.dirname(train_out), exist_ok=True)
            train_dataframe.to_csv(train_out, index=False, header=True)
            test_dataframe.to_csv(test_out, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status       = status,
                valid_train_file_path   = cfg.valid_train_file_path   if status else None,
                valid_test_file_path    = cfg.valid_test_file_path    if status else None,
                invalid_train_file_path = cfg.invalid_train_file_path if not status else None,
                invalid_test_file_path  = cfg.invalid_test_file_path  if not status else None,
                drift_report_file_path  = cfg.drift_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys)