import pandas as pd
import json
import configparser
import argparse
from apscheduler.schedulers.background import BackgroundScheduler

import DB

import warnings
warnings.simplefilter("ignore")
pd.set_option('mode.chained_assignment', None)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

db = DB.Database()

conf = configparser.ConfigParser()
conf.read('load.init')



schedule = BackgroundScheduler()


# =======================================CONFIGURATION=======================================*
args.host = conf.get('server','host')
# args.host = conf.get('server','local')
# args.host = conf.get('server','container_host')
args.port = conf.get('server','port')
args.url = 'http://' + args.host + ':' + args.port + '/predict'
# args.url = 'http://125.131.88.57' + ':' + args.port + '/predict'

args.minutes = conf.getint('settings','minutes')
args.interval = conf.getint('settings','interval')
args.count = conf.getint('settings','count')
args.features = json.loads(conf.get('settings','feature'))

select = 'UG0000000003'
args.id = conf.get('sid', select)
# ============================================================================================*

def collect(uid):
    cond = "WHERE u_grp_id = '{}' ORDER BY metering_time DESC LIMIT {}".format(uid, args.count)
    load = db.select("pg_load_metering", "*", cond)
    print('load: {}'.format(args.id))
    return load


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
#
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
