import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from creditriskmodelling.logging.logger import logging
from creditriskmodelling.exception.exception import CreditRiskModellingException
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where() 

class CreditDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
        
    def csv_to_json_convertor(self,file_path_customer,file_path_loan,file_path_bureau):
        try:
            logging.info("Reading CSV files")
            df_customers = pd.read_csv(file_path_customer)
            df_loans = pd.read_csv(file_path_loan)
            df_bureau = pd.read_csv(file_path_bureau)

            logging.info("Merging Dataframes")
            df = pd.merge(df_customers, df_loans, on='cust_id', how='inner')
            df = pd.merge(df, df_bureau, on='cust_id', how='inner')

            logging.info("Converting DataFrame to JSON")
            json_records = json.loads(df.to_json(orient="records"))

            return json_records
        
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
    
    def insert_data_to_mongodb(self,records,database,collection):
        try:        
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = self.mongo_client[database]
            collection = db[collection]

            logging.info("Inserting data into MongoDB")

            collection.insert_many(records)

            logging.info("Data inserted successfully")

            return len(records)

        except Exception as e:
            raise CreditRiskModellingException(e,sys)    
        
if __name__ == '__main__':
    FILE_PATH_CUSTOMER = "Credit_Data\\customers.csv"
    FILE_PATH_LOAN = "Credit_Data\\loans.csv"
    FILE_PATH_BUREAU = "Credit_Data\\bureau_data.csv"

    DATABASE="CREDIT_RISK_MODELLING"
    Collection="CreditData"
    credit_obj = CreditDataExtract()
    records = credit_obj.csv_to_json_convertor(file_path_customer=FILE_PATH_CUSTOMER,file_path_loan=FILE_PATH_LOAN,file_path_bureau=FILE_PATH_BUREAU)
    number_of_records = credit_obj.insert_data_to_mongodb(records,DATABASE,Collection)
    print(f"Number of records: {number_of_records}")