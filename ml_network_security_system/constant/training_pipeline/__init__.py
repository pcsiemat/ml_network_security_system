import os
import sys
import numpy as np
import pandas as pd


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN        = "Result"
PIPELINE_NAME: str   = "NetworkSecurity"
ARTIFACT_DIR: str    = "Artifacts"   #All the history
FILE_NAME: str       = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str  = "test.csv"
SAVED_MODEL_DIR      = os.path.join("saved_models")

MODEL_FILE_NAME      = "model.pkl"
FINAL_MODEL_OBJECT_DIR: str = "final_model" #Objects for production : model + preprocessing
METRICS_FILE_NAME: str = "metrics.yaml" 

TEMPLATE_DIR:str       = "Templates"
PREDICTION_OUTPUT_DIR: str = "prediction_output"



SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
sample : 
 -artifacts/<timestamp>/data_ingestion/feature_store/<raw>.csv
 -artifacts/<timestamp>/data_ingestion/ingested/train.csv
 -artifacts/<timestamp>/data_ingestion/ingested/test.csv
"""
DATA_INGESTION_COLLECTION_NAME: str           = "Network_Data_Collection"
DATA_INGESTION_DATABASE_NAME: str             = "Cmatos_DB"
DATA_INGESTION_DIR_NAME: str                  = "data_ingestion"   #the phase
DATA_INGESTION_FEATURE_STORE_DIR: str         = "feature_store"    #for raw file. subfolder of data_ingestion
DATA_INGESTION_INGESTED_DIR: str              = "ingested"         #for train and test files. subfolder of data_ingestion
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float  = 0.2
DATA_INGESTION_RANDOM_STATE: int              = 42



"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_THRESHOLD: float      = 0.05    # ← AGREGAR: drift if p-value < threshold




"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
#MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
MODEL_TRAINER_CV_STD_WARNING = 0.02
MODEL_OBJECT_FILE_NAME = "model.pkl"



"""
MLflow related constants
"""
MLFLOW_DEFAULT_SERVER: str  = "http://127.0.0.1:5000"
MLFLOW_OFFLINE_URI: str     = "sqlite:///mlflow_offline.db"   # separate DB, never clashes with the server's
MLFLOW_EXPERIMENT_NAME: str = "ml_network_security_system"    #for local mflow and dagshub
MLFLOW_REGISTERED_MODEL_NAME: str = "network_security_model"  # ← NEW: fixed name in the Model Registry for local mflow and dagshub

"""
Dagshub related constants
"""
DAGSHUB_REPO_OWNER: str = "pcsiemat"
DAGSHUB_REPO_NAME:  str = "ml_network_security_system"