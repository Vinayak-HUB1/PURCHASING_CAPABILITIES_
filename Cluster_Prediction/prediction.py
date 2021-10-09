import joblib
from Logging.setup_logger import setup_logger
import pandas as pd
import numpy as np

# setting up logs
log = setup_logger("model_logs", "Logs/Prediction_logs.log")


class Predict():
    def __init__(self, cluster_model_path):
        self.cluster_model_path = cluster_model_path # path to cluster model
        # self.moedel_folder_path = moedel_folder_path # path to moedel folder
        try:
            log.info("Loading model from {}".format(self.cluster_model_path))
            self.cluster_model = joblib.load(self.cluster_model_path)
            log.info("Cluster Model loaded")
        except Exception as e:
            log.error("Error occured while loading model")
            log.error(e)



    def cluster_predict(self, data):
        try:
            log.info("Predicting cluster for data")
            cluster = self.cluster_model.predict(data)
            log.info("Predicted cluster {}".format(cluster))
            return cluster # return cluster number
        except Exception as e:
            log.error("Error occured while predicting cluster")
            log.error(e)


   
