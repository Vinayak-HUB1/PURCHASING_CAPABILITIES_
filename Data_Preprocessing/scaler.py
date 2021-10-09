# creating a class to scale the data

import joblib
from Logging.setup_logger import setup_logger
import pandas as pd
import numpy as np
# setting up logs
log = setup_logger("scaler_logs", "Logs/scaling_logs.log")

class Scaler():
    """Class to scale the input data
    """
    def __init__(self, scaler_model):
        """
        Initializing the class

        Methode : __init__

        Input: scaler_model file path, data as pandas dataframe or numpy array

        Output: None
        """
        try:
            log.info("Initilization Scaling Class")
            log.info("Loading scaler model")
            self.scaler_model_path = scaler_model # file path to scaler model
            log.info("Scaler Model loaded Successfully")
    
            # loading scaler model
            log.info("Loading scaler model")
            self.scaler_model = joblib.load(self.scaler_model_path) # loading scaler model
            self.scaler_model.clip = False # setting clip to false
            log.info("Scaler model loaded Successfully")
        except Exception as e:
            log.error("Error in Initialization Scaling Class")
            log.error(e)



    def scale_data(self, data):
        """Scaling the data

        Methode : scale_data

        Input: None
        
        Output: scaled data as DataFrame or Numpy array
        """

        try:
            if (isinstance(data, np.ndarray)): # checking the input data is pandas dataframe or not
                log.info("Data is valid")
            else:
                log.error("Data is invalid")
                raise ValueError("Data is invalid") # if data is not valid then raise error
            log.info("Scaling the numpy array") # if data is numpy array then
            data_scaled = self.scaler_model.transform(data) # converting data to numpy array
            log.info("Scaling the numpy array Successfully")
            return data_scaled
        except Exception as e:
            log.error("Error in Scaling the numpy array")
            log.error(e)    