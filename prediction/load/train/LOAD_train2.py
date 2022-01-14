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
h2o.init()

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('load.init')

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
args.feature = json.loads(conf.get('settings','feature'))
model_path = "model1/"

con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
cursor = con.cursor()

def collect(rid):
    ld_cond = "r_id = '{}'".format(rid)
    ld_db = "SELECT * FROM {} WHERE {}".format(table[0], ld_cond)
    cursor.execute(ld_db)
    ld = pd.DataFrame(cursor.fetchall())
    ld.columns = [desc[0] for desc in cursor.description]

    env_cond = "local = '{}'".format('ulsan')
    env_db = "SELECT * FROM {} WHERE {}".format(table[3], env_cond)
    cursor.execute(env_db)
    env = pd.DataFrame(cursor.fetchall())
    env.columns = [desc[0] for desc in cursor.description]

    ld = ld[["r_id", "metering_time", "meter_value", "day_class"]]
    ld = ld.astype({"meter_value": "float"})
    # ld['hour'] = ld['metering_time'].dt.hour
    ld1 = ld[(ld['metering_time'] > args.train_start) & (ld['metering_time'] < args.train_end)]
    ld1 = ld1.set_index('metering_time').resample('1H').sum()

    env = env[["base_date_time", "t1h"]]
    env.columns = ["base_date_time", "tmp"] #train model column: tmp->t1h
    env = env[(env['base_date_time'] > args.train_start) & (env['base_date_time'] < args.train_end)].reset_index(drop=True)
    env = env.astype({"tmp": "float"})
    # env1 = env.set_index('base_date_time').resample('15T').mean()
    # env1 = env1.fillna(method='pad')

    load = pd.merge(ld1, env.set_index("base_date_time"), left_index=True, right_index=True).reset_index().rename(columns={"index": "time"})
    load = load[load['tmp'] > -10]
    load['hour'] = load['time'].dt.hour
    load['day_class'] = ld['day_class'][0]

    return load


def create_model(rid):
    print("Select load: ", rid)
    original = collect(rid)
    train_data = original[args.feature]

    train = train_data[0:int(len(train_data) * .67)]
    test = train_data[int(len(train_data) * .67) + 1:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)
    train_df = pd.DataFrame(train_scaled, columns=args.feature)
    test_df = pd.DataFrame(test_scaled, columns=args.feature)
    htrain = h2o.H2OFrame(train_df)
    htest = h2o.H2OFrame(test_df)

    x = htrain.columns
    y = 'meter_value'

    aml = H2OAutoML(max_runtime_secs = 500) #nfolds=9
    aml.train(x=x, y=y, training_frame=htrain, leaderboard_frame = htest) #validation_frame=hval,

    lb = aml.leaderboard
    lb.head()

    pred=aml.leader.predict(htest)
    pred_df = pred.as_data_frame()
    htest_df = htest.as_data_frame()

    def inverse_value(value):
        _value = np.zeros(shape=(len(value), len(args.feature)))
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
        with open(model_path + 'model_accuracy2.json', 'r') as j:
            accuracy = json.load(j)
        mape = "%.3f" % np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(pred))))
        mae = "%.3f" % mean_absolute_error(actual, pred)
        rmse = "%.3f" % np.sqrt(mean_squared_error(actual, pred))
        rsquared = '%.3f' % r2_score(actual, pred)

        accuracy[rid][0]["MAPE"] = mape
        accuracy[rid][0]["MAE"] = mae
        accuracy[rid][0]["RMSE"] = rmse
        accuracy[rid][0]["RSQUARED"] = rsquared
        with open(model_path + 'model_accuracy2.json', 'w') as w:
            json.dump(accuracy, w, indent=4)

        plt.figure(figsize=(20, 6))
        plt.plot(actual, 'red', label='actual')
        plt.plot(pred, 'blue', label='predict')
        plt.title('load Predict Result')
        plt.xlabel('date')
        plt.ylabel('meter')
        plt.legend()
        plt.savefig(model_path + rid + '/{}_chart'.format(rid, dpi=300))
        plt.tight_layout()
        plt.show()

    evaluate_model(inv_test, inv_pred)
    performance = aml.leader.model_performance(htest)

    return aml, performance

r = args.rid[0]
aml, performance = create_model(r)
save_model = h2o.download_model(aml.leader, model_path + r)

##Leaderboard(ranked by xval metrics)
# lb = aml.leaderboard
# aml.leader.model_performance(hval)
print("Best Model: ", aml.leader.model_id)
print("\n")
# evaluate_model(htest_df['meter_value'], pred_df)

aml.leader.varimp()

h2o.remove_all()
