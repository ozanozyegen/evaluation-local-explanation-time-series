import pandas as pd
import numpy as np
from datetime import timedelta, datetime


def resample(data, freq):
    """
    :param data: dataframe
    :param freq: sampling frequency min
    :return: resampled data between the the first day at 00:00:00 and the last day at 23:60-freq:00 at freq sample frequency
    """
    start = data.datetime.iloc[0].strftime('%Y-%m-%d') + " 00:00:00"
    end = datetime.strptime(data.datetime.iloc[-1].strftime('%Y-%m-%d'), "%Y-%m-%d") + timedelta(days=1) - timedelta(
        minutes=freq)
    pick_popular = lambda x: x.value_counts().index[0] if len(x.value_counts())>0 else np.nan 
    index = pd.period_range(start=start,
                            end=end,
                            freq=str(freq) + 'min').to_timestamp()
    data = data.resample(str(freq) + 'min', on="datetime").agg({'glucose': np.mean, 'CHO': np.sum, 
        "insulin": np.sum, 'meal': pick_popular})
    data = data.reindex(index=index)
    data = data.reset_index()
    data = data.rename(columns={"index": "datetime"})
    return data

def create_samples(data, ph, hist, day_len, subject):
    """
    Create samples consisting in glucose, insulin and CHO histories (past hist-length values)
    :param data: dataframe
    :param ph: prediction horizon in minutes in sampling frequency scale
    :param hist: history length in sampling frequency scale
    :param day_len: length of day in sampling frequency scale
    :return: dataframe of samples
    """
    n_samples = data.shape[0] - ph - hist + 1
    # Add time indexes
    data['hour'] = data['datetime'].dt.hour
    data['dayofweek'] = data['datetime'].dt.dayofweek

    y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
    d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
    t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
    g = np.array([data.loc[i:i + n_samples - 1, "glucose"] for i in range(hist)]).transpose()
    c = np.array([data.loc[i:i + n_samples - 1, "CHO"] for i in range(hist)]).transpose()
    i = np.array([data.loc[i:i + n_samples - 1, "insulin"] for i in range(hist)]).transpose()
    dh = np.array([data.loc[i:i + n_samples - 1, "hour"] for i in range(hist)]).transpose()
    dw = np.array([data.loc[i:i + n_samples - 1, "dayofweek"] for i in range(hist)]).transpose()
    s = np.array([[subject]*(n_samples) for _ in range(hist)]).transpose()

    m = np.array([data.loc[i:i + n_samples - 1, "meal"] for i in range(hist)]).transpose()

    new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], 
        ["CHO_" + str(i) for i in range(hist)], 
        ["insulin_" + str(i) for i in range(hist)], 
        ["hour_" + str(i) for i in range(hist)],
        ["dayofweek_" + str(i) for i in range(hist)],
        ["meal_" + str(i) for i in range(hist)],
        ["subject_" + str(i) for i in range(hist)],
        ["y"]]
    new_data = pd.DataFrame(data=np.c_[t, g, c, i, dh, dw, m, s, y], columns=new_columns)
    new_data["datetime"] = d
    new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first

    return new_data

