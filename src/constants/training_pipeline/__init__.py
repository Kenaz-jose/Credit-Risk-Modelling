import sys
import os
import pandas as pd
import numpy as np

"""
Defining common constant variable for training pipeline
"""
TARGET_COLUMN = "default"
PIPELINE_NAME:str = "CreditRiskModelling"
ARTIFACT_DIR:str = "Artifacts"
FILE_NAME:str = "CreditData.csv"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("schema","schema.yaml")
OUTPUT_DIR: str = "output"
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"
FEATURE_NAMES: str = "feature_names.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "CreditData"
DATA_INGESTION_DATABASE_NAME: str = "CREDIT_RISK_MODELLING"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformer_object"

"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str ="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05