import yaml
import os
import sys
import pickle
import numpy as np
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

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

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:

        report = {}
        best_models = {}

        for model_name, model in models.items():

            param = params[model_name]

            randomsearch = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param,
                        n_iter=5,        # number of random combinations
                        cv=3,             # reduce folds for speed
                        scoring="roc_auc",
                        n_jobs=-1,
                        verbose=1,
                        random_state=42
                    )

            randomsearch.fit(X_train, y_train)

            best_model = randomsearch.best_estimator_
            best_models[model_name] = best_model

            if hasattr(best_model, "predict_proba"):
                y_train_pred = best_model.predict_proba(X_train)[:, 1]
                y_test_pred = best_model.predict_proba(X_test)[:, 1]
            else:
                y_train_pred = best_model.decision_function(X_train)
                y_test_pred = best_model.decision_function(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred)
            test_model_score = roc_auc_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report,best_models

    except Exception as e:
        raise CreditRiskModellingException(e, sys)