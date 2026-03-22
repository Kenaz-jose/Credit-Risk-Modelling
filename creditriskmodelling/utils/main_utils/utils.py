import yaml
import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
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

def load_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CreditRiskModellingException(e,sys)

def load_numpy_array_data(file_path:str) -> np.array:
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj,allow_pickle=True)
    except Exception as e:
        raise CreditRiskModellingException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, model, param_grid):
    try:
   
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        randomsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=cv,
            scoring="neg_log_loss",   
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        
        randomsearch.fit(X_train, y_train)
        best_model = randomsearch.best_estimator_

        calibrated_model = CalibratedClassifierCV(
            best_model,
            method="isotonic",  
            cv=3
        )

        calibrated_model.fit(X_train, y_train)

        y_train_pred = calibrated_model.predict_proba(X_train)[:, 1]
        y_test_pred = calibrated_model.predict_proba(X_test)[:, 1]

        
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        train_logloss = log_loss(y_train, y_train_pred)
        test_logloss = log_loss(y_test, y_test_pred)

        report = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_log_loss": train_logloss,
            "test_log_loss": test_logloss
        }

        return report, calibrated_model

    except Exception as e:
        raise CreditRiskModellingException(e, sys)