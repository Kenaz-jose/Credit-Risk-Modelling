import sys
import os
import pandas as pd
import numpy as np
from src.exception.exception import CreditRiskModellingException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.constants.training_pipeline import TARGET_COLUMN
from src.utils.main_utils.utils import save_numpy_array_data, save_object
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
    
    def handle_missing_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        This function handles missing values in the residence column
        """
        try:
            residence_mode = train_df["residence_type"].mode()[0]
            train_df["residence_type"] = train_df["residence_type"].fillna(residence_mode)
            test_df["residence_type"] = test_df["residence_type"].fillna(residence_mode)

            return train_df, test_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    
    def filter_processing_fee(self,train_df:pd.DataFrame, test_df:pd.DataFrame) -> tuple:
        """
        Keep rows where processing_fee / loan_amount < 0.03
        Safe version handling division by zero and NaN values
        """
        try:
            train_df = train_df[train_df["loan_amount"].notna() & (train_df["loan_amount"] != 0)]
            test_df = test_df[test_df["loan_amount"].notna() & (test_df["loan_amount"] != 0)]

            train_ratio = train_df["processing_fee"]/train_df["loan_amount"]
            test_ratio = test_df["processing_fee"]/test_df["loan_amount"]

            train_df = train_df[train_ratio < 0.03]
            test_df = test_df[test_ratio < 0.03]

            return train_df,test_df

        except Exception as e:
            raise CreditRiskModellingException(e,sys)
        
    def correct_loan_purpose_typo(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Correct typo in loan_purpose column:
        Replace 'Personaal' with 'Personal'
        """
        try:
            train_df["loan_purpose"] = train_df["loan_purpose"].replace("Personaal", "Personal")
            test_df["loan_purpose"] = test_df["loan_purpose"].replace("Personaal", "Personal")
            return train_df, test_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
        
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
            df["delinquent_ratio"] = df["delinquent_months"] / df["total_loan_months"].replace(0, np.nan)
            df["avg_dpd_per_delinquency"] = df["total_dpd"] / df["delinquent_months"].replace(0, np.nan)
            return df
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def clean_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median(numeric_only=True))
            return df
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            cols_to_drop = [
                "cust_id",
                "loan_id",
                "disbursal_date",
                "installment_start_dt",
                "loan_amount",
                "income",
                "total_loan_months",
                "delinquent_months",
                "total_dpd","state",
                "zipcode",
                "city",
                "sanction_amount",
                "processing_fee",
                "gst",
                "net_disbursement",
                "principal_outstanding"
            ]
            return df.drop(columns=cols_to_drop)
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)
        

    
    def calculate_woe_iv(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        try:

            categorical_cols = df.select_dtypes(include=["object"]).columns
            iv_list = []

            for col in categorical_cols:
                temp = pd.crosstab(df[col], df[target_column])
                temp.columns = ["good", "bad"]
                temp["dist_good"] = temp["good"]/temp["good"].sum()
                temp["dist_bad"] = temp["bad"]/temp["bad"].sum()
                temp["WOE"] = np.log(temp["dist_good"] / temp["dist_bad"])
                temp["IV"] = (temp["dist_good"] - temp["dist_bad"]) * temp["WOE"]
                iv_list.append((col, temp["IV"].sum()))
            return pd.DataFrame(iv_list, columns=["feature", "IV"])
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def select_features_by_iv(self, df: pd.DataFrame, iv_df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        try:
            selected_cat = iv_df[iv_df["IV"] > threshold]["feature"].tolist()
            numeric = df.select_dtypes(exclude=["object"]).columns.tolist()
            return df[numeric + selected_cat]
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    ## Scaling & Encoding Pipeling
    def create_column_transformer(self, X_train: pd.DataFrame):
        try:
            numeric_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

            numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(drop=None, sparse_output=False))])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ], remainder="passthrough")
            return preprocessor, numeric_features, categorical_features
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return X_res, y_res
        
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    ## Full Pipeline
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Reading train and test data...")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Handling missing values")
            train_df, test_df = self.handle_missing_values(train_df, test_df)

            logging.info("Filtering processing fee")
            train_df, test_df = self.filter_processing_fee(train_df, test_df)

            logging.info("Correcting loan purpose typo")
            train_df, test_df = self.correct_loan_purpose_typo(train_df, test_df)

            logging.info("Adding and cleaning engineered features...")
            train_df = self.add_engineered_features(train_df)
            train_df = self.clean_engineered_features(train_df)

            test_df = self.add_engineered_features(test_df)
            test_df = self.clean_engineered_features(test_df)

            logging.info("Dropping unnecessary columns...")
            train_df = self.drop_unnecessary_columns(train_df)
            test_df = self.drop_unnecessary_columns(test_df)

            X_train, y_train = train_df.drop(columns=[TARGET_COLUMN]), train_df[TARGET_COLUMN]
            print(y_train.value_counts())
            X_test, y_test = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]
            print(y_test.value_counts())

            print(train_df[[TARGET_COLUMN]].head(20))

            logging.info("Calculating IV and selecting features...")
            temp_df = pd.concat([X_train, y_train], axis=1)
            iv_df = self.calculate_woe_iv(temp_df, TARGET_COLUMN)
            X_train = self.select_features_by_iv(X_train, iv_df)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            logging.info("Creating Column Transformer...")
            preprocessor, numeric_features, categorical_features = self.create_column_transformer(X_train)

            logging.info("Fitting and transforming train data...")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            X_train, y_train = self.apply_smote(pd.DataFrame(X_train), y_train)

            logging.info("Saving transformed data and objects...")
            feature_names = list(preprocessor.get_feature_names_out())            
            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object(self.data_transformation_config.feature_object_file_path,feature_names)

            logging.info("Data transformation completed.")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                feature_object_file_path=self.data_transformation_config.feature_object_file_path
            )

        except Exception as e:
            raise CreditRiskModellingException(e, sys)