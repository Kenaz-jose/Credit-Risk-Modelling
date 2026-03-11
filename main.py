import os
import sys
from creditriskmodelling.components.data_ingestion import DataIngestion,DataIngestionArtifact
from creditriskmodelling.components.data_validation import DataValidation
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from creditriskmodelling.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from creditriskmodelling.entity.config_entity import TrainingPipelineConfig
from creditriskmodelling.components.data_transformation import DataTransformation
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        logging.info("initiate the Data Ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Initiation Completed")

        logging.info("Initiating Data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Data Validation Completed")

        logging.info("Initiated Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation completed")
        
    except Exception as e:
        raise CreditRiskModellingException(e,sys)