
"""
=====================================================================
 DATA TRANSFORMATION - Stage 3 of the training pipeline
=====================================================================
INPUT  : DataValidationArtifact     (paths of validated train.csv / test.csv)
         DataTransformationConfig   (where to save arrays and preprocessor)
STEPS  : 1. stop if validation failed (drift found)
         2. split X (30 features) and y (Result)
         3. map target: -1 phishing → 1, 1 legitimate → 0
         4. KNNImputer: fit on train, transform train and test
         5. join X + y → train.npy / test.npy (last column = target)
         6. save fitted preprocessor → preprocessing.pkl (+ copy in final_model/)
OUTPUT : DataTransformationArtifact
=====================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from ml_network_security_system.constant.training_pipeline import TARGET_COLUMN
from ml_network_security_system.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from ml_network_security_system.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from ml_network_security_system.entity.config_entity import DataTransformationConfig
from ml_network_security_system.exception.exception import AppException 
from ml_network_security_system.logger.log import logging
from ml_network_security_system.utils.main_utils.utils import save_numpy_array_data,save_object

# phishing = 1 (the class we want to detect), legitimate = 0
TARGET_MAPPING = {-1: 1, 1: 0}

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise AppException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AppException(e, sys)
        
    def get_data_transformer_object(self)->Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          self: DataTransformation

        Returns:
          A Pipeline object
        """
        logging.info(
            "Entered get_data_transformer_object method of Transformation class"
        )
        try:
           imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
           logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
           processor:Pipeline=Pipeline([("imputer",imputer)])
           return processor
        except Exception as e:
            raise AppException(e,sys)

        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            ## 1. stop if validation failed
            if not self.data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: drift found. "
                    f"Check the report: {self.data_validation_artifact.drift_report_file_path}"
                )
 
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df  = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
 
            ## 2-3. training dataframe: X / y + target mapping
            input_feature_train_df  = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace(TARGET_MAPPING)
 
            ## 2-3. testing dataframe: X / y + target mapping
            input_feature_test_df  = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TARGET_MAPPING)
 
            logging.info(f"Target train counts: {target_feature_train_df.value_counts().to_dict()}")
            logging.info(f"Target test counts: {target_feature_test_df.value_counts().to_dict()}")
 
            ## 4. fit on train, transform both
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature  = preprocessor_object.transform(input_feature_test_df)
 
            ## 5. join features + target (last column = target)
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr  = np.c_[transformed_input_test_feature,  np.array(target_feature_test_df)]
            logging.info(f"Train array: {train_arr.shape}, Test array: {test_arr.shape}")
 
            ## 6. save arrays and preprocessor
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
 
            # latest copy for predictions (overwritten on every run)
            # save_object(self.data_transformation_config.transformed_final_model_dir, preprocessor_object)
 
            ## prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
 
        except Exception as e:
            raise AppException(e, sys)
