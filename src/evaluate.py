import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Arsid-Tarune/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="Arsid-Tarune"
os.environ["MLFLOW_TRACKING_PASSWORD"]="883966fa76bf08c7dd5e84b20b26e80747efaa22"

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["trains"]

def evaluate(data_path, model_path):
    data=pd.read(data_path)
    X= data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_url = ("https://dagshub.com/Arsid-Tarune/machinelearningpipeline.mlflow")
    ## LOading the model from the disk 

    model= pickle.load(open(model_path,'rb'))

    predictions= model.predict(X)
    accuarcy= accuracy_score(y,predictions)

    ## Log metrics to MLfLOW

    mlflow.log_metric("accuracy" , accuarcy)
    print("Model accuracy : {accuracy}")
    
    
if __name__ = "__main__":
     evaluate(params["data"], params["model"])