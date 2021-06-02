import tensorflow.keras.backend as K
import numpy as np
from deprecated import deprecated

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def nrmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / (K.max(y_true)-K.min(y_true))

def nrmse_mean(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / K.mean(y_true)

def nd(y_true, y_pred):
    y_true, y_pred = K.flatten(y_true), K.flatten(y_pred)
    return K.sum( K.abs(y_true-y_pred) ) / K.sum(K.abs(y_true))

""" Numpy Metrics """
def rmse_np(y_true, y_pred):
    diff = y_pred - y_true # (num_samples, target_size)
    numerator = diff**2
    return np.sqrt( numerator.flatten().mean() )

def nrmse_maxmin_np(y_true, y_pred):
    return rmse_np(y_true, y_pred) / (np.max(y_true) - np.min(y_true))

def nrmse_mean_np(y_true, y_pred):
    return  rmse_np(y_true, y_pred) / np.mean(y_true)

def nrmse_batch_np(y_true, y_pred, divide_by, batch_size=1):
    """ Compute NRMSE for each batch, then compute 
        mean, std, confidence interval of batches
        Returns:
            mean, std, conf (float)
    """
    N = y_true.shape[0]
    num_batches = N//batch_size
    nrmse_scores = np.empty((num_batches,))
    for i in range(num_batches):
        start, end = i*batch_size, (i+1)*batch_size
        rmse_batch = rmse_np(y_true[start:end], y_pred[start:end])
        if divide_by == 'max-min':    
            nrmse_scores[i] = rmse_batch/(np.max(y_true)-np.min(y_true))
        elif divide_by == 'mean':
            nrmse_scores[i] = rmse_batch/np.mean(y_true)
        else: # No denominator
            nrmse_scores[i] = rmse_batch
    mean, std = np.mean(nrmse_scores), np.std(nrmse_scores)
    conf = conf_interval(std, N)
    return mean, std, conf

def nd_np(y_true, y_pred):
    return np.sum( np.abs(y_true - y_pred) ) / np.sum(np.abs(y_true))

def nd_samples_batch_np(y_true, y_pred, batch_size=256):
    """ Compute ND for each batch, then compute 
        mean, std, confidence interval of batches
        Returns:
            mean, std, conf (float)
    """
    N = y_true.shape[0]
    num_batches = N // batch_size
    nd_scores = np.empty((num_batches,))
    for i in range(num_batches):
        start, end = i*batch_size, (i+1)*batch_size
        nd_scores[i] = nd_np(y_true[start:end], y_pred[start:end])
    mean, std = np.mean(nd_scores), np.std(nd_scores)
    conf = conf_interval(std, N)
    return mean, std, conf

def conf_interval(sample_std, N, z_val=1.96):
    """ Compute confidence interval for the mean 
    Parameters:
        z_val (float): 1.96 for 95% confidence
    """
    return (z_val*sample_std)/(N**(1/2))
