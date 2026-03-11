import sys
import os
import pandas as pd
import numpy as np
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging
from creditriskmodelling.entity.config_entity import DataTransformationConfig
from creditriskmodelling.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from creditriskmodelling.constants.training_pipeline import TARGET_COLUMN
from creditriskmodelling.utils.main_utils.utils import save_numpy_array_data,save_object
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
    
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

    def convert_zipcode_to_string(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Convert zip_code column to string datatype
        """
        try:
            train_df["zipcode"] = train_df["zipcode"].astype(str)
            test_df["zipcode"] = test_df["zipcode"].astype(str)
            return train_df, test_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def add_engineered_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Create new engineered features for the model
        """
        try:
            train_df["loan_to_income"] = train_df["loan_amount"] / train_df["income"].replace(0, np.nan)
            test_df["loan_to_income"] = test_df["loan_amount"] / test_df["income"].replace(0, np.nan)

            train_df["delinquent_ratio"] = train_df["delinquent_months"] / train_df["total_loan_months"].replace(0, np.nan)
            test_df["delinquent_ratio"] = test_df["delinquent_months"] / test_df["total_loan_months"].replace(0, np.nan)

            train_df["avg_dpd_per_delinquency"] = train_df["total_dpd"] / train_df["delinquent_months"].replace(0, np.nan)
            test_df["avg_dpd_per_delinquency"] = test_df["total_dpd"] / test_df["delinquent_months"].replace(0, np.nan)

            return train_df, test_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
        
    def drop_unnecessary_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Drop columns that are not required for model training
        """
        try:
            
            columns_to_drop = [
                "cust_id",
                "loan_id",
                "disbursal_date",
                "installment_start_dt",
                "loan_amount",
                "income",
                "total_loan_months",
                "delinquent_months",
                "total_dpd"
            ]
            train_df = train_df.drop(columns=columns_to_drop)
            test_df = test_df.drop(columns=columns_to_drop)
            return train_df, test_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    
    def split_input_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Separate input features and target column from train and test datasets
        """
        try:
            
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            return X_train, y_train, X_test, y_test

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    

    def scale_numerical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Scale numerical columns using StandardScaler
        """
        try:
            scaler = StandardScaler()
            numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns
            X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

            return X_train, X_test, scaler

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    

    def remove_high_vif_features(self, df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
        """
        Iteratively remove features with high VIF
        """
        try:
            numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            numeric_df = numeric_df.fillna(numeric_df.median())

            while True:
                vif_df = pd.DataFrame()
                vif_df["feature"] = numeric_df.columns
                vif_df["VIF"] = [
                    variance_inflation_factor(numeric_df.values, i)
                    for i in range(numeric_df.shape[1])
                ]

                max_vif = vif_df["VIF"].max()

                if max_vif > threshold:
                    drop_feature = vif_df.sort_values("VIF", ascending=False)["feature"].iloc[0]
                    logging.info(f"Dropping feature {drop_feature} with VIF {max_vif}")

                    numeric_df = numeric_df.drop(columns=[drop_feature])
                else:
                    break

            remaining_numeric = list(numeric_df.columns)
            categorical_cols = list(df.select_dtypes(exclude=["int64","float64"]).columns)

            return df[remaining_numeric + categorical_cols]

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def calculate_woe_iv(self, df: pd.DataFrame, target_column: str):
        """
        Calculate WOE and IV for categorical features
        """
        try:
            
            categorical_cols = df.select_dtypes(include=["object"]).columns

            iv_list = []

            for col in categorical_cols:

                temp_df = pd.crosstab(df[col], df[target_column])

                temp_df.columns = ["good", "bad"]

                temp_df["dist_good"] = temp_df["good"] / temp_df["good"].sum()
                temp_df["dist_bad"] = temp_df["bad"] / temp_df["bad"].sum()

                temp_df["WOE"] = np.log(temp_df["dist_good"] / temp_df["dist_bad"])

                temp_df["IV"] = (temp_df["dist_good"] - temp_df["dist_bad"]) * temp_df["WOE"]

                iv = temp_df["IV"].sum()

                iv_list.append((col, iv))

            iv_df = pd.DataFrame(iv_list, columns=["feature", "IV"])

            return iv_df

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    
    def select_features_by_iv(self, df: pd.DataFrame, iv_df: pd.DataFrame, threshold: float = 0.02):

        try:
            selected_cat_features = iv_df[iv_df["IV"] > threshold]["feature"].tolist()

            numerical_features = df.select_dtypes(exclude=["object"]).columns.tolist()

            final_features = numerical_features + selected_cat_features

            logging.info(f"Selected categorical features: {selected_cat_features}")

            return df[final_features]

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Encode categorical features using One-Hot Encoding
        """
        try:
            categorical_cols = X_train.select_dtypes(include=["object"]).columns

            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

            # Align train and test columns
            X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

            logging.info("Categorical encoding completed")

            return X_train, X_test

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Balance dataset using SMOTE
        """
        try:
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_train = X_train.fillna(X_train.median())
            
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            return X_train_resampled, y_train_resampled

        except Exception as e:
            raise CreditRiskModellingException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        logging.info("Entered initiate_data_transformation method")

        try:
            
            logging.info("Reading train and test dataset")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Handling missing values")
            train_df, test_df = self.handle_missing_values(train_df, test_df)

            logging.info("Filtering processing fee")
            train_df, test_df = self.filter_processing_fee(train_df, test_df)

            logging.info("Correcting loan purpose typo")
            train_df, test_df = self.correct_loan_purpose_typo(train_df, test_df)

            logging.info("Converting zipcode to string")
            train_df, test_df = self.convert_zipcode_to_string(train_df, test_df)

            logging.info("Adding engineered features")
            train_df, test_df = self.add_engineered_features(train_df, test_df)

            logging.info("Dropping unnecessary columns")
            train_df, test_df = self.drop_unnecessary_columns(train_df, test_df)

            logging.info("Splitting input and target")
            X_train, y_train, X_test, y_test = self.split_input_target(train_df, test_df)

            logging.info("Scaling numerical features")
            X_train, X_test, scaler = self.scale_numerical_features(X_train, X_test)

            logging.info("Applying VIF feature selection")
            X_train = self.remove_high_vif_features(X_train)
            X_test = X_test[X_train.columns]

            logging.info("Calculating IV for categorical features")
            temp_df = pd.concat([X_train, y_train], axis=1)
            iv_df = self.calculate_woe_iv(temp_df, TARGET_COLUMN)

            logging.info("Selecting features based on IV")
            X_train = self.select_features_by_iv(X_train, iv_df)
            X_test = X_test[X_train.columns]

            logging.info("Encoding categorical features")
            X_train, X_test = self.encode_categorical_features(X_train, X_test)

            logging.info("Applying SMOTE")
            X_train, y_train = self.apply_smote(X_train, y_train)
            
            logging.info("Converting train and test to numpy array")
            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]
            
            logging.info("Saving the train and test numpy array")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,scaler)
            
            logging.info("Preparing Data Transformation Artifact")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise CreditRiskModellingException(e, sys)