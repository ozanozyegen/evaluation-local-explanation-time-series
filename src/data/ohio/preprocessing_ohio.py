import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

from data.ohio.helpers import resample, create_samples
from data.ohio.loading_ohio import load_ohio, scale
from data.ohio.nans_filling import fill_nans
from data.ohio.nans_removal import remove_nans

def preprocessing_subject(subject, target_size, hist, day_len, n_days_test, DATA_DIR='data/raw/ohio-t1dm/2018/'):
    """
    OhioT1DM dataset preprocessing pipeline:
    loading -> samples creation -> cleaning (1st) -> splitting -> cleaning (2nd) -> standardization
    First cleaning is done before splitting to speedup the preprocessing
    :param subject: id of the subject, e.g. "559"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :return: training_old folds, validation folds, testing folds, list of scaler (one per fold)
    """
    df = load_ohio(subject, DATA_DIR)#.iloc[:3000] For debugging
    data = resample(df, freq=5)
    data = create_samples(data, 1, hist, day_len, subject) # Get one step ahead dataset
    data = fill_nans(data, day_len, n_days_test)
    # Create y+1 to y+target_size for multi-step-ahead prediction
    for target_time in range(1, target_size):
        data[f'y_{target_time}'] = data['y'].shift(-target_time)
    data.rename(columns={'y':'y_0'}, inplace=True)
    return data

def preprocessing_all(ph, hist, day_len, n_days_test, standardize, DATA_DIR='data/raw/ohio-t1dm/2018/'):
    """ OhioT1DM dataset preprocessing all patients
    """
    subjects = ["559", "563" , "570", "575", "588", "591"]
    trains, tests = [], []
    for subject in tqdm(subjects):
        df_subject = preprocessing_subject(subject, ph, hist, day_len, n_days_test, DATA_DIR)
        train = df_subject.iloc[:-n_days_test * day_len].copy()
        test = df_subject.iloc[-n_days_test * day_len:].copy()
        trains.append(train); tests.append(test)
    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    [train, test] = remove_nans([train, test])
    scaled_train, scaled_test, scalers = scale(train, test, hist, ph, standardize)
    dataset = {'train':scaled_train, 'test':scaled_test, 'scalers':scalers, 'NUM_SERIES':len(subjects)}
    dataset_params = {'TARGET_SIZE':ph, 'HISTORY_SIZE':hist, 'STANDARDIZE':standardize}
    pickle.dump((dataset, dataset_params), open('data/interim/ohio_df.pickle', 'wb'))
    return dataset, dataset_params


def preprocess_ohio_point_forecast(config={}, PREPROCESSED_PATH='data/interim/ohio_df.pickle'):
    """ Ohio dataset with real features and subject id
    """
    # See if you can use the cached_datasets
    cached_dataset_params = None
    if os.path.exists(PREPROCESSED_PATH):
        cached_dataset, cached_dataset_params = pickle.load(open(PREPROCESSED_PATH, 'rb'))
    # If cached_dataset_params is a subset of the config
    if cached_dataset_params is not None and cached_dataset_params.items() <= config.items():
        dataset, dataset_params = cached_dataset, cached_dataset_params
    else:
        if 'STANDARDIZE' not in config: # Set default value for old models
            config['STANDARDIZE'] = True
        dataset, dataset_params = preprocessing_all(ph=config['TARGET_SIZE'], 
            hist=config['HISTORY_SIZE'], day_len=288, n_days_test=10, 
            standardize=config['STANDARDIZE'])
    
    target_features = [f'y_{i}' for i in range(config['TARGET_SIZE'])]
    train_y = dataset['train'][target_features].to_numpy('float32')
    test_y = dataset['test'][target_features].to_numpy('float32')

    train_x = dataset['train'].drop(['datetime', 'time']+target_features, axis=1).to_numpy('float32').reshape(len(train_y), -1, config['HISTORY_SIZE'])
    test_x = dataset['test'].drop(['datetime', 'time']+target_features, axis=1).to_numpy('float32').reshape(len(test_y), -1, config['HISTORY_SIZE'])
    # Swap time and feature axis
    train_x = np.transpose(train_x, (0,2,1))
    test_x = np.transpose(test_x, (0,2,1))
    # Filter subjects
    subject_scaler = dataset['scalers']['subject']
    if 'SUBJECTS' in config and len(config['SUBJECTS'])<len(subject_scaler.classes_):
        subject_nums = subject_scaler.transform(config['SUBJECTS'])
        train_subject_idxs = np.in1d(train_x[:,0,-1], subject_nums)
        test_subject_idxs = np.in1d(test_x[:,0,-1], subject_nums)
        train_x, train_y = train_x[train_subject_idxs], train_y[train_subject_idxs]
        test_x, test_y = test_x[test_subject_idxs], test_y[test_subject_idxs]
        

    del dataset['NUM_SERIES']
    dataset_params = dict(
        FEATURES = ['Glucose', 'CHO', 'Insulin', 'hour', 'dayofweek', 'meal', 'Subject'],
        CONT_FEATURES=[0,1,2],
        CAT_FEATURES=[-1],
    )

    # Update dataset params based on config, choose subset of the features
    if 'CONT_FEATURES' in config and 'CAT_FEATURES' in config:
        dataset_params.update({'CONT_FEATURES':config['CONT_FEATURES'],
                                'CAT_FEATURES':config['CAT_FEATURES']})
        feature_idxs = config['CONT_FEATURES'] + config['CAT_FEATURES']
        dataset_params['FEATURES'] = [dataset_params['FEATURES'][feat_idx] for feat_idx in feature_idxs]
        train_x = train_x[:,:,feature_idxs]
        test_x = test_x[:,:,feature_idxs]

    dataset.update({'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y})
    dataset.update({'target_scaler':dataset['scalers']['glucose']})
    return dataset_params, dataset

if __name__ == "__main__":
    dataset = preprocessing_all(ph=1, hist=60, day_len=288, n_days_test=10, standardize=True, DATA_DIR='data/raw/ohio-t1dm/2018/')
    # df = preprocessing_subject("559", ph=1, hist=60, day_len=288, n_days_test=10, DATA_DIR='data/raw/ohio-t1dm/2018/')

