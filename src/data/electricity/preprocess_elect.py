import numpy as np
import pandas as pd
from pandas.core import series
from scipy import stats
from tqdm import tqdm
from data.helpers import week_of_month, multivariate_data
from sklearn import preprocessing
import pickle

def gen_covariates(times, num_covariates=4):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.hour
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = week_of_month(input_time)
        covariates[i, 3] = input_time.month
    for i in range(num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def generate_elect():
    """ Turn raw data into a pandas DataFrame.
    - Trim the train test range
    - Sample hourly 
    - Add time covariates
    """
    train_start = '2012-01-01 00:00:00'
    train_end = '2014-08-31 23:00:00'
    test_start = '2014-08-25 00:00:00' #need additional 7 days as given info
    test_end = '2014-09-07 23:00:00'
    
    df = pd.read_csv("data/raw/LD2011_2014.txt", sep=";", index_col=0, parse_dates=True, decimal=',')
    df = df.resample('1H',label = 'left',closed = 'right').sum()[train_start:test_end].sort_index()
    covariates = gen_covariates(df[train_start:test_end].index)
    print(covariates.shape)
    df = df[train_start:test_end]

    series_labels = df.columns
    series_dict = {}

    for series_label in tqdm(series_labels):
        numeric_series_label = int(series_label[-3:])
        series_dict[numeric_series_label] = pd.DataFrame({'series':df[series_label], 'hour':covariates[:,0],
        'weekday':covariates[:,1], 'weekofmonth':covariates[:,2], 'month':covariates[:,3]})
    return series_dict

def elect_loader(config):
    series_dict = generate_elect()
    target_scalers = {}
    if config['NUM_SERIES'] > 1:
        train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
        np.random.seed(0)
        selected_houses = np.random.choice(list(series_dict.keys()), size=config['NUM_SERIES']) 
        for selected_house in tqdm(selected_houses):
            series = series_dict[selected_house]
            # Add house number as a feature
            normalized_house_num = selected_house / 370 # Normalized house num (0-370)
            series['house'] = normalized_house_num
            # Normalization
            target_scalers[normalized_house_num] = preprocessing.MinMaxScaler()
            series['series'] = target_scalers[normalized_house_num].fit_transform(series[['series']])

            split_index = 15000
            """ Create dataset for keras NN models """
            train_x, train_y = multivariate_data(series.values, target=series['series'].values, start_index=0, end_index=split_index,
                                history_size=config['HISTORY_SIZE'], target_size=config['TARGET_SIZE'], stride=config['STRIDE'])
            test_x, test_y = multivariate_data(series.values, target=series['series'].values, start_index=split_index, end_index=None,
                                history_size=config['HISTORY_SIZE'], target_size=config['TARGET_SIZE'], stride=config['STRIDE'])

            train_x_all.append(train_x)
            train_y_all.append(train_y)
            test_x_all.append(test_x)
            test_y_all.append(test_y)

        concat = lambda x: np.concatenate(x, axis=0)
        print('Packing dataset')
        train_x = concat(train_x_all)
        train_y = concat(train_y_all)
        test_x = concat(test_x_all)
        test_y = concat(test_y_all)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        scalers = {'load':target_scalers}
        dataset = {'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y, 'scalers':scalers, 'target_scaler': None}
        dataset_params=dict(
            FEATURES = ['load', 'hour', 'weekday', 'weekofmonth', 'month', 'house']
        )
        return dataset_params, dataset
    else:
        raise NotImplementedError()
