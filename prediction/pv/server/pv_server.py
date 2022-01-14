import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
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
conf.read('pv_server.init')

args.host = conf.get('server','host')
# args.host = conf.get('server','docker_container')
args.port = conf.get('server','port')
args.features = json.loads(conf.get('settings','feature'))
args.features.remove('r_id')

scaler = MinMaxScaler()
def normalization(data):
    nor = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(nor, columns=args.features)
    scaled = scaled_df.to_json()
    return scaled

def inverse_transform(value):
    _value = np.zeros(shape=(len(value), len(args.features)))
    _value[:, 0] = value[:, 0]
    inv_value = scaler.inverse_transform(_value)[:, 0]
    return inv_value


@app.route("/predict", methods=['POST'])
def predict():
    receive_data = pd.DataFrame(json.loads(request.get_data()))
    r_id = receive_data["r_id"][0]
    print("r_id: {}".format(r_id))
    print("Receive Data\n ", receive_data)

    receive_feature = receive_data[args.features]
    receive_nor = pd.read_json(normalization(receive_feature))
    # print("Normalization\n ", receive_nor)
    hdata = h2o.H2OFrame(receive_nor)

    args.model_path = 'model/{}/'.format(r_id)
    model_list = os.listdir(args.model_path)
    args.model = model_list[0]
    print("model path: ", args.model_path + args.model)

    print("model: ", args.model)
    model = h2o.upload_model(args.model_path + args.model)

    prediction = model.predict(hdata).as_data_frame()
    prediction["pred_value"] = inverse_transform(prediction["predict"].values.reshape(-1, 1))
    print(prediction)
    predict = prediction["pred_value"].to_json()

    print("AI Model Prediction Success")

    return {'predict': predict}

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)