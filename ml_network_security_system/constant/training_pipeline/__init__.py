import os
import sys
import numpy as np
import pandas as pd


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN        = "Result"
PIPELINE_NAME: str   = "NetworkSecurity"
ARTIFACT_DIR: str    = "Artifacts"
FILE_NAME: str       = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str  = "test.csv"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
sample : 
 -artifacts/<timestamp>/data_ingestion/feature_store/raw.csv
 -artifacts/<timestamp>/data_ingestion/ingested/train.csv
 -artifacts/<timestamp>/data_ingestion/ingested/test.csv
"""
DATA_INGESTION_COLLECTION_NAME: str           = "Network_Data_Collection"
DATA_INGESTION_DATABASE_NAME: str             = "Cmatos_DB"
DATA_INGESTION_DIR_NAME: str                  = "data_ingestion"   #the phase
DATA_INGESTION_FEATURE_STORE_DIR: str         = "feature_store"    #for raw file. subfolder of data_ingestion
DATA_INGESTION_INGESTED_DIR: str              = "ingested"         #for train and test files. subfolder of data_ingestion
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
