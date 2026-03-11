import yaml
import dill
import os
import sys
import pickle
import numpy as np
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path,"rb") as  yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditRiskModellingException(e,sys) from e

def write_yaml_file(file_path:str, content:object, replace:bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content, file)
            
    except Exception as e:
        raise CreditRiskModellingException(e,sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to a file
    file_path: str location of the file to save
    array: np.array to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditRiskModellingException(e,sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CreditRiskModellingException(e,sys)