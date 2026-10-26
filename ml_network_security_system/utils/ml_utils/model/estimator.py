from ml_network_security_system.constant.training_pipeline import MODEL_FILE_NAME

import os
import sys

from ml_network_security_system.exception.exception import AppException
from ml_network_security_system.logger.log import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise AppException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise AppException(e,sys)