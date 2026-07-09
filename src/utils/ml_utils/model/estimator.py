import os
import sys
import numpy as np
import pandas as pd
from src.exception.exception import CreditRiskModellingException


class CreditModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def _prepare_input(self, X: pd.DataFrame):
        try:

            expected_cols = list(self.preprocessor.feature_names_in_)

            # Add missing columns
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0

            # Remove extra columns
            X = X[expected_cols]

            return X

        except Exception as e:
                raise CreditRiskModellingException(e, sys)

    def predict(self, X: pd.DataFrame):
        try:
            X = self._prepare_input(X)

            X_transformed = self.preprocessor.transform(X)

            return self.model.predict(X_transformed)

        except Exception as e:
            raise CreditRiskModellingException(e, sys)

    def predict_proba(self, X: pd.DataFrame):
        try:
            X = self._prepare_input(X)

            X_transformed = self.preprocessor.transform(X)

            return self.model.predict_proba(X_transformed)

        except Exception as e:
            raise CreditRiskModellingException(e, sys)