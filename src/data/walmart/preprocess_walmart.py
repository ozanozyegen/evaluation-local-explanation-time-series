#%%
import numpy as np
import pandas as pd
import os
from deprecated import deprecated
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from tqdm import tqdm
from data.helpers import multivariate_data, convert_wandb_config_to_dict

def walmart_split(config, DATA_PATH="data/processed/walmart.csv"):
    data = pd.read_csv(DATA_PATH)
    test_start_idx = 120
    # %%
    scalers = {
        'Weekly_Sales': StandardScaler(),
        'Store':LabelEncoder(),
        'Dept':LabelEncoder(),
        'Size':StandardScaler(),
        'year': LabelEncoder(),
        'Temperature':StandardScaler(),
        'Fuel_Price':StandardScaler(),
        'CPI':StandardScaler(),
        'Unemployment':StandardScaler()
    }
    for feature in scalers:
        data[feature] = scalers[feature].fit_transform(data[feature].values.reshape(-1,1))

    # %%
    train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
    # data.groupby(['Store', 'Dept'], as_index=False).size().index.values.tolist()
    series_ids = data.groupby(['Store', 'Dept'], as_index=False).size()[['Store','Dept']].values.tolist()
    np.random.seed(0)
    np.random.shuffle(series_ids)
    for count, (store_id, dept_id) in enumerate(tqdm(series_ids)):
        if count > config['NUM_SERIES']:
            break
        df = data.loc[(data['Store']==store_id) & (data['Dept']==dept_id)]
        # df.set_index('Date', drop=True, inplace=True)
        df = df.drop(['Unnamed: 0', 'Date'], axis=1)
        
        train_x, train_y = multivariate_data(df.values, df.Weekly_Sales.values, 0, 
            test_start_idx, config['HISTORY_SIZE'], config['TARGET_SIZE'], config['STRIDE'])
        test_x, test_y = multivariate_data(df.values, df.Weekly_Sales.values, test_start_idx-config['HISTORY_SIZE']+config['TARGET_SIZE'], 
            None, config['HISTORY_SIZE'], config['TARGET_SIZE'], config['STRIDE'])

        train_x_all.append(train_x)
        train_y_all.append(train_y)
        test_x_all.append(test_x)
        test_y_all.append(test_y)

    concat = lambda x: np.concatenate(x, axis=0)
    train_x = concat(train_x_all)
    train_y = concat(train_y_all)
    test_x = concat(test_x_all)
    test_y = concat(test_y_all)
    return train_x, train_y, test_x, test_y, data, scalers

@deprecated("Used for testing a similar preprocessing approach \
    to Elect and Rossmann, not used anymore")
def walmart_minmax_split(config, DATA_PATH="data/processed/walmart.csv"):
    """ Scales each series between 0 and 1 similar to Electricity and Rossmann """
    data = pd.read_csv(DATA_PATH)
    test_start_idx = 120
    # %%
    scalers = {
        'Store':LabelEncoder(),
        'Dept':LabelEncoder(),
        'Size':StandardScaler(),
        'year': LabelEncoder(),
        'Temperature':StandardScaler(),
        'Fuel_Price':StandardScaler(),
        'CPI':StandardScaler(),
        'Unemployment':StandardScaler()
    }
    for feature in scalers:
        data[feature] = scalers[feature].fit_transform(data[feature].values.reshape(-1,1))
    scalers['Weekly_Sales'] = dict()
    # %%
    train_x_all, train_y_all, test_x_all, test_y_all = [], [], [], []
    # data.groupby(['Store', 'Dept'], as_index=False).size().index.values.tolist()
    series_ids = data.groupby(['Store', 'Dept'], as_index=False).size()[['Store','Dept']].values.tolist()
    np.random.seed(0)
    # np.random.shuffle(series_ids)
    for count, (store_id, dept_id) in enumerate(tqdm(series_ids)):
        if count > config['NUM_SERIES']:
            break
        df = data.loc[(data['Store']==store_id) & (data['Dept']==dept_id)]
        # df.set_index('Date', drop=True, inplace=True)
        df = df.drop(['Unnamed: 0', 'Date'], axis=1)
        scalers['Weekly_Sales'][(store_id, dept_id)] = MinMaxScaler()
        df['Weekly_Sales'] = scalers['Weekly_Sales'][(store_id, dept_id)].fit_transform(df[['Weekly_Sales']])

        train_x, train_y = multivariate_data(df.values, df.Weekly_Sales.values, 0, 
            test_start_idx, config['HISTORY_SIZE'], config['TARGET_SIZE'], config['STRIDE'])
        test_x, test_y = multivariate_data(df.values, df.Weekly_Sales.values, test_start_idx-config['HISTORY_SIZE']+config['TARGET_SIZE'], 
            None, config['HISTORY_SIZE'], config['TARGET_SIZE'], config['STRIDE'])

        train_x_all.append(train_x)
        train_y_all.append(train_y)
        test_x_all.append(test_x)
        test_y_all.append(test_y)

    concat = lambda x: np.concatenate(x, axis=0)
    train_x = concat(train_x_all)
    train_y = concat(train_y_all)
    test_x = concat(test_x_all)
    test_y = concat(test_y_all)
    return train_x, train_y, test_x, test_y, data, scalers

def walmart_loader(config):
    if 'dataset_loader' in config and config['dataset_loader'] == 'minmax_loader':
        train_x, train_y, test_x, test_y, data, scalers = walmart_minmax_split(config)    
    else:
        train_x, train_y, test_x, test_y, data, scalers = walmart_split(config)
    # %%
    dataset_params = dict(
        FEATURES = ['Weekly_Sales', 'Store', 'Dept', 'Year', 'month', 'weekofmonth', 
        'day', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday'],
        CONT_FEATURES = [0,9,10,11,12,13],
        CAT_FEATURES = [1,2,3,4,5,6,7,8,]
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
        'scalers':scalers, 'data':data, 'target_scaler':scalers['Weekly_Sales']}
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return dataset_params, dataset