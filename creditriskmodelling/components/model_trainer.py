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
        model = LogisticRegression(random_state=42)
        param_grid = {
            
            "LogisticRegression": [
               # 🔵 L2 (main stable model)
                {
                    "penalty": ["l2"],
                    "C": [0.01, 0.1, 1, 5],
                    "solver": ["lbfgs"],
                    "max_iter": [1000],
                    "class_weight": [None, "balanced"]
                },

                # 🟡 ElasticNet (controlled sparsity)
                {
                    "penalty": ["elasticnet"],
                    "solver": ["saga"],
                    "C": [0.01, 0.1, 1],
                    "l1_ratio": [0.1, 0.2, 0.3],
                    "max_iter": [2000],
                    "class_weight": [None, "balanced"]
                }
            ]
        }
            

        print("NaN in X_train:", np.isnan(X_train).sum())
        print("NaN in X_test:", np.isnan(X_test).sum())
        print("NaN in y_train:", np.isnan(y_train).sum())
        print("NaN in y_test:", np.isnan(y_test).sum())


        model_report, best_model = evaluate_model(
        X_train, y_train, X_test, y_test,
        model=model,
        param_grid=param_grid["LogisticRegression"])
        best_model_score = model_report["test_auc"]
        logging.info(f"Best Model: {best_model}")
        logging.info(f"Train AUC: {model_report['train_auc']}")
        logging.info(f"Test AUC: {model_report['test_auc']}")

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