"""
    Preprocessing code for the rossman dataset
    Generates data/processed/rossman_lstm_data.pickle file
"""

import numpy as np
import pandas as pd
from scipy import stats
from math import ceil
from tqdm import tqdm
import pickle
from sklearn import preprocessing
from data.helpers import week_of_month, multivariate_data

def gen_covariates(times, num_covariates=3):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday()
        covariates[i, 1] = input_time.month
        covariates[i, 2] = week_of_month(input_time)
    for i in range(num_covariates):
        covariates[:, i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def generate_rossmann_data(config):
    history_size, target_size, stride, num_series = config['HISTORY_SIZE'], config['TARGET_SIZE'], \
                                                    config['STRIDE'], config['NUM_SERIES']
    n_features = 10

    data = pd.read_csv('data/raw/rossmann-store-sales/train.csv',parse_dates=True,index_col='Date') 
    data['StateHoliday'] = data['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3})
    data = data.astype(float)
    data['StateHoliday'].fillna(0, inplace=True)
    # print('Num null: {}'.format(data.isnull().sum()))

    stores = data['Store'].unique()
    # Pick N random stores
    np.random.seed(0)
    selected_stores = np.random.choice(stores, size=num_series, replace=False)
    selected_store_count = 0

    scalers = {'Sales':{}}
    train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
    for selected_store in selected_stores:
        store = data.loc[data['Store'] == selected_store].sort_index()
        if store.shape[0] != 942 or selected_store == 988.0:
            continue
        selected_store_count += 1
        store = store.drop('DayOfWeek', 1)

        train_start = '2013-01-01'
        train_end = '2014-12-31'
        test_start = '2015-01-01'
        test_end = '2015-07-31'
        get_date_idx = lambda date: np.where(store.index == date)[0][0]

        standardize = lambda x: (x-x.mean()) / x.std()
        scalers['Sales'][selected_store] = preprocessing.MinMaxScaler()
        series = store['Sales'].values.reshape(-1, 1)
        normalized_series = scalers['Sales'][selected_store].fit_transform(series).astype(np.float)
        covariates_df = store.drop('Sales', 1)
        
        # print(f"Dataset features: {covariates_df.columns}")
        time_covariates = gen_covariates(store.index)
        covariates_df['weekday'] = time_covariates[:, 0]
        covariates_df['month'] = time_covariates[:, 1]
        covariates_df['weekofmonth'] = time_covariates[:, 2]

        normalized_data = np.concatenate((normalized_series, covariates_df.values.astype(np.float)), axis=1) 
        assert n_features == normalized_data.shape[1]

        train_x, train_y = multivariate_data(normalized_data, target=normalized_data[:, 0], start_index=get_date_idx(train_start), 
                end_index=get_date_idx(train_end), history_size=history_size, target_size=target_size, stride=stride)

        test_x, test_y   = multivariate_data(normalized_data, target=normalized_data[:, 0], start_index=get_date_idx(test_start), 
            end_index=None,  history_size=history_size, target_size=target_size, stride=stride)
        assert n_features == train_x.shape[2] and train_x.shape[2] == test_x.shape[2]

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

    # Prepare loader
    dataset_params = dict(
        FEATURES = ['Series', 'Store', 'Customers', 'Open', 'Promo', 'StateHoliday', \
            'SchoolHoliday', 'weekday', 'month', 'weekofmonth'],
        CONT_FEATURES = [0,2], # Sales, Customers
        CAT_FEATURES = [1,3,4,5,6,7,8,9]
    )
    # Update dataset params based on config, choose subset of the features
    if 'CONT_FEATURES' in config and 'CAT_FEATURES' in config:
        dataset_params.update({'CONT_FEATURES':config['CONT_FEATURES'],
                                'CAT_FEATURES':config['CAT_FEATURES']})
        feature_idxs = config['CONT_FEATURES'] + config['CAT_FEATURES']
        dataset_params['FEATURES'] = [dataset_params['FEATURES'][feat_idx] for feat_idx in feature_idxs]
        train_x = train_x[:,:,feature_idxs]
        test_x = test_x[:,:,feature_idxs]
    
    dataset = {'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y,
        'scalers':scalers, 'target_scaler': None}
    return dataset_params, dataset