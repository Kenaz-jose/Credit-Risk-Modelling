import os
import sys
import numpy as np
import mlflow
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from creditriskmodelling.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from creditriskmodelling.entity.config_entity import ModelTrainerConfig
from creditriskmodelling.utils.main_utils.utils import save_object,load_object
from creditriskmodelling.utils.main_utils.utils import load_numpy_array_data,evaluate_model
from creditriskmodelling.utils.ml_utils.model.estimator import CreditModel
from creditriskmodelling.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
import dagshub
dagshub.init(repo_owner='kenazjose007', repo_name='Credit-Risk-Modelling', mlflow=True)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric, dataset_type):

        mlflow.set_experiment("credit_risk_pipeline")

        with mlflow.start_run(run_name=dataset_type):

            metrics = {
                f"{dataset_type}_f1_score": classificationmetric.f1_score,
                f"{dataset_type}_precision_score": classificationmetric.precision_score,
                f"{dataset_type}_recall_score": classificationmetric.recall_score,
                f"{dataset_type}_roc_auc_score": classificationmetric.roc_auc_score,
                f"{dataset_type}_ks_statistic": classificationmetric.ks_statistic
            }

            mlflow.log_metrics(metrics)

            mlflow.log_params(best_model.get_params())

            mlflow.sklearn.log_model(best_model, "model")


    def train_model(self,X_train,y_train,X_test,y_test):
        models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "ExtraTrees": ExtraTreesClassifier()
                }
        param_grid = {

                    "LogisticRegression": {
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "penalty": ["l1", "l2"],
                        "solver": ["liblinear"],
                        "max_iter": [100, 200, 500]
                    },

                    "RandomForest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],

                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2"]
                    },

                    "GradientBoosting": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [3, 5],
                        "subsample": [0.8, 1.0],
                        "min_samples_split": [2, 5]
                    },

                    "ExtraTrees": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2"]
                    },

                    "AdaBoost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1]
                    }

                }

        print("NaN in X_train:", np.isnan(X_train).sum())
        print("NaN in X_test:", np.isnan(X_test).sum())
        print("NaN in y_train:", np.isnan(y_train).sum())
        print("NaN in y_test:", np.isnan(y_test).sum())

        model_report,best_models = evaluate_model(X_train,y_train,X_test,y_test,models=models,params=param_grid)
        
        best_model_name = max(model_report, key=model_report.get)
        best_model_score = model_report[best_model_name]
        best_model = best_models[best_model_name]
        logging.info(f"The best model: {best_model} and it's score: {best_model_score}")

        y_train_pred = best_model.predict(X_train)
        y_train_pred_proba = best_model.predict_proba(X_train)[:,1]
        classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred,y_pred_proba=y_train_pred_proba)
        self.track_mlflow(best_model,classification_train_metric,"train")

        y_test_pred = best_model.predict(X_test)
        y_test_pred_proba = best_model.predict_proba(X_test)[:,1]
        classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred,y_pred_proba=y_test_pred_proba)
        self.track_mlflow(best_model,classification_test_metric,"test")

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        Credit_Model = CreditModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Credit_Model)
        
        save_object("final_model/model.pkl",best_model)
        ### Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric)
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            print("Train shape:", train_arr.shape)
            print("Test shape:", test_arr.shape)

            print("NaN in train:", np.isnan(train_arr).sum())
            print("NaN in test:", np.isnan(test_arr).sum())

            print("Train dtype:", train_arr.dtype)

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise CreditRiskModellingException(e,sys)