import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

feature=['day_class', 'meter_value', 'tmp', 'hour']

scaler = MinMaxScaler()
def normalization(data):
    nor = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(nor, columns=feature)
    scaled = scaled_df.to_json()
    return scaled

def inverse_value(value):
    _value = np.zeros(shape=(len(value), len(feature)))
    _value[:, 0] = value[:, 0]
    inv_value = scaler.inverse_transform(_value)[:, 0]
    return inv_value

a = [0.0343406768891054, 0.0343406768891054, 0.0343406768891054, 0.0343406768891054, 0.0175400867019381, 0.0175400867019381, 0.0175400867019381, 0.0175400867019381, 0.0107689660335885, 0.0107689660335885, 0.0107689660335885, 0.0107689660335885, 0.0221274590836645, 0.0221274590836645, 0.0221274590836645, 0.0221274590836645, 0.0242675019000972, 0.0242675019000972, 0.0242675019000972, 0.0242675019000972, 0.0369638628214361, 0.0369638628214361, 0.0369638628214361, 0.0369638628214361, 0.0608469179430067, 0.0608469179430067, 0.0608469179430067, 0.0608469179430067, 0.0877635632161227, 0.0877635632161227, 0.0877635632161227, 0.0877635632161227, 0.3746891567594877, 0.3746891567594877, 0.3746891567594877, 0.3746891567594877, 0.9036648638096432, 0.9036648638096432, 0.9036648638096432, 0.9036648638096432, 0.2346813153014179, 0.2346813153014179, 0.2346813153014179, 0.2346813153014179, 0.1186427086337021, 0.1186427086337021, 0.1186427086337021, 0.1186427086337021, 0.1303534400888081, 0.1303534400888081, 0.1303534400888081, 0.1303534400888081, 0.1370965140863041, 0.1370965140863041, 0.1370965140863041, 0.1370965140863041, 0.0196988101296803, 0.0196988101296803, 0.0196988101296803, 0.0196988101296803, 0.022038329866494, 0.022038329866494, 0.022038329866494, 0.022038329866494, 0.1459983822986943, 0.1459983822986943, 0.1459983822986943, 0.1459983822986943, 0.2543411726492435, 0.2543411726492435, 0.2543411726492435, 0.2543411726492435, 0.3508853787138177, 0.3508853787138177, 0.3508853787138177, 0.3508853787138177, 0.4697684092931838, 0.4697684092931838, 0.4697684092931838, 0.4697684092931838, 0.1325123334107559, 0.1325123334107559, 0.1325123334107559, 0.1325123334107559, 0.0952779713332355, 0.0952779713332355, 0.0952779713332355, 0.0952779713332355, 0.0762481687714471, 0.0762481687714471, 0.0762481687714471, 0.0762481687714471, 0.0848470966489155, 0.0848470966489155, 0.0848470966489155, 0.0848470966489155]
df = pd.DataFrame({"pred": a})

df["result"] = inverse_value(df["pred"].values.reshape(-1, 1))

