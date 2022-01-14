import pandas as pd
import numpy as np
import requests
import psycopg2
import json
import os
import time
import datetime
import argparse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from datetime import timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from common import Database

parser = argparse.ArgumentParser()
args = parser.parse_args("")

schedule = BackgroundScheduler()

db = Database.Database()

# =======================================CONFIGURATION=======================================*
args.host = 'http://127.0.0.1:'
# args.host ='http://load-server:'
args.port = '5533'
args.url = args.host + args.port + '/predict'
args.table = "pg_load_metering"
args.interval = 15
args.count = 4
args.id = 'UG0000000003'
# ============================================================================================*

def collect():
    cond = "WHERE u_grp_id = '{}' ORDER BY metering_time DESC LIMIT {}".format(args.id, args.count)
    load_sel = db.select(args.table, "*", cond)
    print('load: {}'.format(args.id))
    return load_sel

select_input = collect()
# select_input.columns
args.features = ["meter_value", "day_class"]
load = select_input[args.features]

load_json = load.to_json()
print(load_json)
# scaler = MinMaxScaler()
# normalization = scaler.fit_transform(load)
# scaled_df = pd.DataFrame(normalization, columns=args.features)
# scaled = scaled_df.to_json()
# with open("input.json", "w") as j:
#     json.dump(scaled, j, indent="\t")


# def serving(data):
#     print('input:', data)
#     res = requests.post(args.url, data, verify=False)
#     # res_json = res.json()
#
#     predictions = res['predict']
#     print(predictions)
#     receive_predict = pd.read_json(predictions)
#
#     # def inverse_transform(value):
#     #     _value = np.zeros(shape=(len(value), len(args.features)))
#     #     _value[:, 0] = value[:, 0]
#     #     inv_value = scaler.inverse_transform(_value)[:, 0]
#     #     return inv_value
#
#     # receive_predict['pred_value'] = inverse_transform(receive_predict['predict'].values.reshape(-1, 1))
#
#     rdf = pd.merge(select_input[['u_grp_id','metering_time','r_id']], receive_predict['pred_value'], left_index=True, right_index=True)
#     rdf.to_csv("rdf.csv", mode='w')
#
#
#     insert_db = "INSERT INTO pg_load_prediction(u_grp_id, pred_time, r_id, pred_value) VALUES (%s,%s,%s,%s)"
#     predict_data = []
#     for i in range(len(rdf)):
#         predict_data.append(rdf.loc[i].tolist())
#
#     db.cursor.executemany(insert_db, predict_data)
#     db.db.commit()

# serving(load_json)

#
# @schedule.scheduled_job('interval', minutes=args.interval, id='test_2')
# def repeat():
#     for rtu_id_inv in enumerate(args.rtu_id_inv):
#         val = rtu_id_inv[1]
#         serving(val)
#
# repeat()
# schedule.start()
#
# while True:
#     time.sleep(1)
