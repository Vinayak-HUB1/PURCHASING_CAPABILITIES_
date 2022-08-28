from flask.globals import request
from flask.json import jsonify
import pandas as pd
import numpy as np
from flask import Flask,render_template
import os
from Cluster_Prediction.prediction import Predict
from Data_Preprocessing.scaler import Scaler
from Logging.setup_logger import setup_logger
import warnings
warnings.filterwarnings('ignore')

# setting up logs
log = setup_logger("flask_app_log", "Logs/app_Logs.log")


# creating scaler class instance
scaler = Scaler("Saved_models\Scaler.pkl")
pred = Predict("Saved_models\cluster_model.pkl")



# creating flask app instance
app = Flask(__name__)

# home page
@app.route('/',methods=['GET','POST'])
def home():
    return render_template("index.html") # return home.html


# predict page
@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        log.info("Start taking input from webpage")
        if request.method == "POST":
            try:
                log.info("Received POST request")
                BALANCE = int(request.form["BALANCE"])
                log.info("BALANCE: {}".format(BALANCE))
                BALANCE_FREQUENCY=int(request.form["BALANCE_FREQUENCY"])
                log.info("BALANCE_FREQUENCY: {}".format(BALANCE_FREQUENCY))
                PURCHASES=int(request.form["PURCHASES"])
                log.info("PURCHASES: {}".format(PURCHASES))
                ONEOFF_PURCHASES = int(request.form["ONEOFF_PURCHASES"])
                log.info("ONEOFF_PURCHASES: {}".format(ONEOFF_PURCHASES))
                INSTALLMENTS_PURCHASES = int(request.form["INSTALLMENTS_PURCHASES"])
                log.info("INSTALLMENTS_PURCHASES: {}".format(INSTALLMENTS_PURCHASES))
                CASH_ADVANCE = int(request.form["CASH_ADVANCE"])
                log.info("CASH_ADVANCE: {}".format(CASH_ADVANCE))
                PURCHASES_FREQUENCY = int(request.form["PURCHASES_FREQUENCY"])
                log.info("PURCHASES_FREQUENCY: {}".format(PURCHASES_FREQUENCY))
                ONEOFF_PURCHASES_FREQUENCY = int(request.form["ONEOFF_PURCHASES_FREQUENCY"])
                log.info("ONEOFF_PURCHASES_FREQUENCY: {}".format(ONEOFF_PURCHASES_FREQUENCY))
                PURCHASES_INSTALLMENTS_FREQUENCY = int(request.form["PURCHASES_INSTALLMENTS_FREQUENCY"])
                log.info("PURCHASES_INSTALLMENTS_FREQUENCY: {}".format(PURCHASES_INSTALLMENTS_FREQUENCY))
                CASH_ADVANCE_FREQUENCY = int(request.form["CASH_ADVANCE_FREQUENCY"])
                log.info("CASH_ADVANCE_FREQUENCY: {}".format(CASH_ADVANCE_FREQUENCY))
                CASH_ADVANCE_TRX = int(request.form["CASH_ADVANCE_TRX"])
                log.info("CASH_ADVANCE_TRX: {}".format(CASH_ADVANCE_TRX))
                PURCHASES_TRX = int(request.form["PURCHASES_TRX"])
                log.info("PURCHASES_TRX: {}".format(PURCHASES_TRX))
                CREDIT_LIMIT = int(request.form["CREDIT_LIMIT"])
                log.info("CREDIT_LIMIT: {}".format(CREDIT_LIMIT))
                PAYMENTS = int(request.form["PAYMENTS"])
                log.info("PAYMENTS: {}".format(PAYMENTS))
                MINIMUM_PAYMENTS = int(request.form["MINIMUM_PAYMENTS"])
                log.info("MINIMUM_PAYMENTS: {}".format(MINIMUM_PAYMENTS))
                PRC_FULL_PAYMENT = int(request.form["PRC_FULL_PAYMENT"])
                log.info("PRC_FULL_PAYMENT: {}".format(PRC_FULL_PAYMENT))
                TENURE = int(request.form["TENURE"])
                log.info("TENURE: {}".format(TENURE))
                log.info("Input taken from webpage")
            except Exception as e:
                log.error("Error in taking input from webpage: {}".format(e))
                return render_template("error.html",error="Error in taking input from webpage") # returning Error page
            log.info("Start data scaling")
            # making array of data
            data = np.array([[BALANCE,BALANCE_FREQUENCY,PURCHASES,ONEOFF_PURCHASES,
            INSTALLMENTS_PURCHASES,
            CASH_ADVANCE,PURCHASES_FREQUENCY,
            ONEOFF_PURCHASES_FREQUENCY,
            PURCHASES_INSTALLMENTS_FREQUENCY,CASH_ADVANCE_FREQUENCY,
            CASH_ADVANCE_TRX,PURCHASES_TRX,
            CREDIT_LIMIT, PAYMENTS,
            MINIMUM_PAYMENTS,PRC_FULL_PAYMENT,TENURE]])
            # scaling data
            scl_data = scaler.scale_data(data)  # sacling the data: return ndarray
            log.info("Data scaling complete")
            log.info("PredictingCluster number")
            # prediction
            clus = int(pred.cluster_predict(scl_data)) # predicting cluster
            log.info("Complete predicting clusert: " + str(clus))
    
            # creating a pandas dataframe
            pred_df = pd.DataFrame() # empety dataframe
            pred_df['cluster'] = clus # adding cluster
            # saving the dataframe to csv file
            pred_df.to_csv("prediction.csv",index=False)

            # giving text
            if clus == 0:
                text = "Customer Belongs to Cluster 0. \n They Likey have HIGH Purchase Capability."
            elif clus == 1:
                text = "Customer Belongs to Cluster 1. \n They Likey have Low Purchase Capability."
            elif clus == 2:
                text = "Customer Belongs to Cluster 2. \n These Customer are Likely to Buy Items in Installment."

            try:
                log.info("Sending the prediction on results.html")
                return render_template("results.html", prediction="Cluster Number:- " + str(clus), text=text) # return prediction as html
            except Exception as e:
                log.error("Error occured while sending prediction on results.html")
                log.error(e)
                return render_template("index.html", prediction="Error occured while sending prediction on results.html")
    
    except Exception as e:
        log.error("Error occured while predicting: " + str(e))
        return jsonify({"error":str(e)})



# run flask app
if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)



