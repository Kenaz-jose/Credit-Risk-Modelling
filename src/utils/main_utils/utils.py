import yaml
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from src.exception.exception import CreditRiskModellingException
from src.logging.logger import logging

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


def evaluate_model(X_train, y_train, X_test, y_test, model, param_grid,preprocessor):
    try:
   
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        randomsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=cv,
            scoring="neg_log_loss",   ## neg_log_loss measures the uncertainty and accuracy of a classification model by penalizing wrong, confident predictions.
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        
        randomsearch.fit(X_train, y_train)
        best_model = randomsearch.best_estimator_

        calibrated_model = CalibratedClassifierCV(
            best_model,
            method="sigmoid",  
            cv=3
        )

        calibrated_model.fit(X_train, y_train)

        print("\n========== TARGET DISTRIBUTION ==========") 
        print(pd.Series(y_train).value_counts())

        print("\n========== MODEL CLASSES ==========") 
        print(calibrated_model.classes_)

        print("\n========== FEATURE COEFFICIENTS ==========") 
        final_model = calibrated_model.estimator
        feature_names = preprocessor.get_feature_names_out() 
        for feat, coef in zip( feature_names, final_model.coef_[0] ): 
            print(f"{feat}: {coef}")
        
        

        print("\n========== FEATURE MEANS BY TARGET ==========")

        temp_df = pd.DataFrame( X_train, columns=feature_names)
        temp_df["target"] = y_train

        print(
            temp_df.groupby("target")[
                [
                    "num__credit_utilization_ratio",
                    "num__loan_to_income",
                    "num__enquiry_count",
                    "num__avg_dpd_per_delinquency"
                ]
            ].mean()
        )


        y_train_pred = calibrated_model.predict_proba(X_train)[:, 1]
        y_test_pred = calibrated_model.predict_proba(X_test)[:, 1]
        
        print("\n========== PROBABILITY SUMMARY ==========")
        print(pd.Series(y_test_pred).describe())

        bins = np.arange(0, 1.1, 0.1)
        hist = (
            pd.Series(pd.cut(y_test_pred, bins=bins, include_lowest=True))
            .value_counts(sort=False)
        )

        print("\n========== PROBABILITY HISTOGRAM ==========")
        print(hist)
        
        print("\n========== BRIER SCORE ==========")
        train_brier = brier_score_loss(y_train, y_train_pred)
        test_brier = brier_score_loss(y_test, y_test_pred)
        print(f"Train Brier Score: {train_brier:.4f}")
        print(f"Test Brier Score: {test_brier:.4f}")
        
        print("\n========== CALIBRATION CURVE ==========")
        prob_true, prob_pred = calibration_curve(
            y_test,
            y_test_pred,
            n_bins=10,
            strategy="quantile"
        )

        for pred, actual in zip(prob_pred, prob_true):
            print(f"Predicted={pred:.3f}  Actual={actual:.3f}")

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