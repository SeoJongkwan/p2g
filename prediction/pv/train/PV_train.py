import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import json
import argparse
import configparser
import psycopg2
import h2o
from h2o.automl import H2OAutoML

import warnings
warnings.simplefilter("ignore")
pd.set_option('mode.chained_assignment', None)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('pv.init')

dbname = conf.get('DB', 'dbname')
host = conf.get('DB', 'host')
user = conf.get('DB', 'user')
password = conf.get('DB', 'password')
port = conf.get('DB', 'port')
table = json.loads(conf.get('DB','table')) #train table
print("<DB Info>")
print("dbname:", dbname + "\nhost:", host + "\nport:", port)

args.rid = json.loads(conf.get('settings','rid')) #train table
args.train_start = conf.get('settings', 'train_start')
args.train_end = conf.get('settings', 'train_end')
args.timezone = json.loads(conf.get('settings', 'timezone')) #이상감지 시간대
feature = ["meter_value", "icsr", "ta", "rn"]
model_path = "model/"

con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
cursor = con.cursor()

def collect(rid):
    inv_cond = "r_id = '{}'".format(rid)
    inv_db = "SELECT * FROM {} WHERE {}".format(table[0], inv_cond)
    cursor.execute(inv_db)
    inv = pd.DataFrame(cursor.fetchall())
    inv.columns = [desc[0] for desc in cursor.description]
    inv1 = inv[(inv['metering_time'] > args.train_start) & (inv['metering_time'] < args.train_end)]
    inv1 = inv1[["r_id", "metering_time", "meter_value"]].sort_values('metering_time').reset_index(drop=True)

    env_db = "SELECT * FROM {}".format(table[2])
    cursor.execute(env_db)
    env = pd.DataFrame(cursor.fetchall())
    env.columns = [desc[0] for desc in cursor.description]
    env1 = env[(env['tm'] > args.train_start) & (env['tm'] < args.train_end)]
    env1 = env1[["tm", "icsr", "ta", "rn"]].sort_values('tm').reset_index(drop=True)
    env1 = env1.set_index('tm').resample('15T').mean()
    env1 = env1.fillna(method='pad')

    plant = pd.merge(inv1.set_index('metering_time'), env1, left_index=True,
                     right_index=True).reset_index().rename(columns={"index": "time"})
    return plant


def create_model(rid):
    print("Select Plant: ", rid)
    original = collect(rid)
    train_data = original[feature]

    h2o.init()

    train = train_data[0:int(len(train_data) * .67)]
    test = train_data[int(len(train_data) * .67) + 1:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)
    train_df = pd.DataFrame(train_scaled, columns=feature)
    test_df = pd.DataFrame(test_scaled, columns=feature)
    htrain = h2o.H2OFrame(train_df)
    htest = h2o.H2OFrame(test_df)

    x = htrain.columns
    y = 'meter_value'

    aml = H2OAutoML(max_runtime_secs = 600) #nfolds=9
    aml.train(x=x, y=y, training_frame=htrain, leaderboard_frame = htest) #validation_frame=hval,

    lb = aml.leaderboard
    print("LeaderBoard:", lb.head())

    pred = aml.leader.predict(htest)
    pred_df = pred.as_data_frame()
    htest_df = htest.as_data_frame()

    def inverse_value(value):
        _value = np.zeros(shape=(len(value), len(feature)))
        _value[:,0] = value[:,0]
        inv_value = scaler.inverse_transform(_value)[:,0]
        return inv_value

    inv_pred = inverse_value(pred_df['predict'].values.reshape(-1, 1))
    inv_test = inverse_value(htest_df['meter_value'].values.reshape(-1, 1))

    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            if actual[j] != 0:
                res[j] = (actual[j] - predicted[j]) / actual[j]
            else:
                res[j] = predicted[j] / np.mean(actual)
        return res

    def evaluate_model(actual, pred):
        # plt.style.use('dark_background')
        with open(model_path + 'model_accuracy.json', 'r') as j:
            accuracy = json.load(j)
        mape = "%.3f" % np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(pred))))
        mae = "%.3f" % mean_absolute_error(actual, pred)
        rmse = "%.3f" % np.sqrt(mean_squared_error(actual, pred))
        rsquared = '%.3f' % r2_score(actual, pred)

        accuracy[rid][0]["MAPE"] = mape
        accuracy[rid][0]["MAE"] = mae
        accuracy[rid][0]["RMSE"] = rmse
        accuracy[rid][0]["RSQUARED"] = rsquared
        with open(model_path + 'model_accuracy.json', 'w') as w:
            json.dump(accuracy, w, indent=4)

        plt.figure(figsize=(20, 6))
        plt.plot(actual, 'navy', label='actual')
        plt.plot(pred, 'salmon', label='predict')
        plt.title('PV Predict Result: {}'.format(rid))
        plt.xlabel('date')
        plt.ylabel('power')
        plt.legend()
        plt.savefig(model_path + rid + '/{}_chart'.format(rid, dpi=300))
        plt.tight_layout()
        plt.show()

    evaluate_model(inv_test, inv_pred)
    aml.leader.model_performance(htest)

    return aml

for r in args.rid:
    aml = create_model(r)
    model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
    best_model = h2o.download_model(aml.leader, model_path + r)
    print("Best Model:", model_ids[0])
    print("\n\n")

    h2o.remove_all()