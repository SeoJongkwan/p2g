import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import datetime
import argparse
import configparser
from flask import Flask, request
import json
import h2o
from h2o.automl import H2OAutoML

h2o.init()

app = Flask(__name__)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('load_server.init')

args.host = conf.get('server','host')
# args.host = conf.get('server','container_host')
args.port = conf.get('server','port')
args.feature = json.loads(conf.get('settings', 'feature'))
args.feature.remove('r_id')

@app.route("/predict", methods=['POST'])
def predict():
    receive_data = pd.DataFrame(json.loads(request.get_data()))
    r_id = receive_data["r_id"][0]
    print("Receive Time: ", datetime.datetime.now())
    print("r_id: {}".format(r_id))
    print("Receive Data\n ", receive_data)

    receive_feature = receive_data[args.feature]
    hdata = h2o.H2OFrame(receive_feature)

    args.model_path = 'model1/{}/'.format(r_id)
    model_list = os.listdir(args.model_path)
    args.model = model_list[0]
    print("model path: ", args.model_path + args.model)
    model = h2o.upload_model(args.model_path + args.model)

    prediction = model.predict(hdata).as_data_frame()
    print("model\n", prediction)
    print(prediction)

    predict = prediction["predict"].to_json()
    print("AI Model Prediction Success")

    return {'predict': predict}

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)