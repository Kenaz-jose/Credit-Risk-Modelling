import os
import sys
from creditriskmodelling.components.data_ingestion import DataIngestion
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from creditriskmodelling.entity.config_entity import DataIngestionConfig
from creditriskmodelling.entity.config_entity import TrainingPipelineConfig

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("initiate the Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        
    except Exception as e:
        raise CreditRiskModellingException(e,sys)