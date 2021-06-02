import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Input
from tensorflow.python.keras.layers.core import Dropout
from models.losses import root_mean_squared_error, nd
from deprecated import deprecated

def hyper_lstm(config):
    n_features = len(config['CONT_FEATURES']) + len(config['CAT_FEATURES'])
    
    model = Sequential([])
    model.add(Input(shape=(config['HISTORY_SIZE'], n_features)))
    for _ in range(config['NUM_LAYERS']-1):
        model.add(LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT'],
            return_sequences=True))
    model.add(LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT'],
        return_sequences=False))
    model.add(Dropout(config['DROPOUT']))
    model.add(Dense(units=config['TARGET_SIZE']))
    
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=[nd])
    return model

def hyper_tdnn(config):
    n_features = len(config['CONT_FEATURES']) + len(config['CAT_FEATURES'])
    model = Sequential([
        Reshape((n_features*config['HISTORY_SIZE'],), input_shape=(config['HISTORY_SIZE'], n_features)),
    ])
    for _ in range(config['NUM_LAYERS']):
        model.add(Dense(config['NUM_UNITS'], activation='sigmoid'))
        model.add(Dropout(config['DROPOUT']))
    model.add(Dense(units=config['TARGET_SIZE']))
    
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=[nd])
    return model


@deprecated
def generic_tdnn(config):
    n_features = len(config['CONT_FEATURES']) + len(config['CAT_FEATURES'])
    model = Sequential([
        Reshape((n_features*config['HISTORY_SIZE'],), input_shape=(config['HISTORY_SIZE'], n_features)),
        Dense(units=64, activation='sigmoid'),
        Dense(units=64, activation='sigmoid'),
        Dense(units=config['TARGET_SIZE'])
    ])
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=[nd])
    return model

@deprecated
def generic_lstm(config):
    n_features = len(config['CONT_FEATURES']) + len(config['CAT_FEATURES'])
    model = Sequential([
        LSTM(units=32, return_sequences=True, input_shape=(config['HISTORY_SIZE'], n_features)),
        LSTM(units=32, return_sequences=False, activation='relu'),
        Dense(units=config['TARGET_SIZE'])
    ])
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=[nd])
    return model