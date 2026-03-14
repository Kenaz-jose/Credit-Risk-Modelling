import os
import sys
from creditriskmodelling.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from creditriskmodelling.exception.exception import CreditRiskModellingException
from creditriskmodelling.logging.logger import logging

class CreditModel:

    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CreditRiskModellingException(e,sys)
    
    def predict(self,X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transform)
            return y_hat
        except Exception as e:
            raise CreditRiskModellingException(e,sys)