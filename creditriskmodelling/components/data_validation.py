import pandas as pd
from scipy.stats import ks_2samp
import sys
import os
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from creditriskmodelling.entity.config_entity import DataValidationConfig
from creditriskmodelling.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from creditriskmodelling.constants.training_pipeline import SCHEMA_FILE_PATH
from creditriskmodelling.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
        
    def validate_number_of_column(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of column: {number_of_columns}")
            logging.info(f"Data Frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
        
    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]

            dataframe_columns = dataframe.columns

            missing_numerical_columns = []

            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
                return False

            return True

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:

                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                if d1.dtype == "object" or d2.dtype == "object":
                    continue

                ks_test = ks_2samp(d1, d2)

                if ks_test.pvalue < threshold:
                    drift = True
                    status = False
                else:
                    drift = False

                report[column] = {
                    "p_value": float(ks_test.pvalue),
                    "drift_status": drift
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            ## Validate number of columns
            status = self.validate_number_of_column(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"
            
            status = self.validate_number_of_column(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            ## Validate numerical columns in  dataset
            status = self.is_numerical_column_exist(train_dataframe)
            if not status:
                error_message = "Train dataframe does not contain all numerical columns.\n"

            status = self.is_numerical_column_exist(test_dataframe)
            if not status:
                error_message = "Test dataframe does not contain all numerical columns.\n"
            
            ## Check Data drift
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifacts = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            
            return data_validation_artifacts
        except Exception as e:
            raise CreditRiskModellingException(e,sys)