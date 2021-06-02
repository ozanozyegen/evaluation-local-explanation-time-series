""" The metrics that we wanted to record are changed
    so we are restoring the best trained models 
    for each dataset model pair
    computing new metrics and reporting the results
"""
import pprint
import pandas as pd
import numpy as np
from reports.wandb_queries import get_best_model, get_wandb_df
from reports.restore_models import restore_wandb_online
from data.loaders import dataset_loader
from train.helpers import use_gpu
from models.losses import nrmse_batch_np, nd_samples_batch_np
from configs.defaults import Globs

use_gpu(False)

Globs.TAG_TRAIN = 'trainv2'

results_list = []
df=get_wandb_df(Globs.PROJECT_NAME)
for dataset_name in ['electricity', 'rossmann', 'walmart', 'ohio']:
    for model_name in Globs.MODELS:
        config = {
            "PROJECT_NAME" : Globs.PROJECT_NAME,
            "DATASET" : dataset_name,
            "MODEL" : model_name}
        best_model_id = get_best_model(config, only_id=True, wandb_train_tag=Globs.TAG_TRAIN)
        model, model_config = restore_wandb_online(config['PROJECT_NAME'], best_model_id)
        dataset_params, dataset = dataset_loader[model_config['DATASET']](model_config)
        train_x, train_y, test_x, test_y = dataset['train_x'], dataset['train_y'], dataset['test_x'], dataset['test_y']

        print("{}, {}, {}".format(dataset_name, model_name, best_model_id))
        # Below is copied from train/helpers/log_errors function
        y_pred = model.predict(dataset['test_x'])
        y_true = dataset['test_y']
        # Descale (Electricity and Rossmann doesn't have scaler)
        if dataset['target_scaler'] is not None:
            y_pred = dataset['target_scaler'].inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)
            y_true = dataset['target_scaler'].inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)
        else:
            print('Target scaler not found, reporting errors for scaled values')

        nrmse_maxmin = nrmse_batch_np(y_true, y_pred, divide_by='max-min', batch_size=256)
        nrmse_mean = nrmse_batch_np(y_true, y_pred, divide_by='mean', batch_size=256)
        nd = nd_samples_batch_np(y_true, y_pred, batch_size=256)
        conf_range_gen = lambda score, interval: [np.round(score-interval,4), np.round(score+interval,4)]

        prefix=''
        results={
            'DATASET': dataset_name,
            'MODEL': model_name,
            
            prefix+'nrmse_maxmin': nrmse_maxmin[0],
            prefix+'nrmse_maxmin_std': nrmse_maxmin[1],
            prefix+'nrmse_maxmin_95': conf_range_gen(nrmse_maxmin[0], nrmse_maxmin[2]),

            prefix + "nd" : nd[0],
            prefix + "nd_std": nd[1],
            prefix + "nd_95": conf_range_gen(nd[0],nd[2]),

            prefix+'nrmse_mean': nrmse_mean[0],
            prefix+'nrmse_mean_std': nrmse_mean[1],
            prefix+'nrmse_mean_95': conf_range_gen(nrmse_mean[0], nrmse_mean[2]),
        }
        pprint.pprint(results)
        results_list.append(results)
        
df_results = pd.DataFrame(results_list)
df_results.to_csv('reports/best_model_results.csv')