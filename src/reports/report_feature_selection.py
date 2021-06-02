""" Report feature removal experiment results
experiments trained in train_feature_selection.py
"""
from configs.defaults import Globs
from reports.wandb_queries import get_models_feat_removal
import numpy as np
import pandas as pd
# Configuration
Globs.TAG_FEATREM = 'feat_remv2'

results_list=[]
conf_range_gen = lambda score, interval: [np.round(score-interval,4), np.round(score+interval,4)]
for dataset in Globs.DATASETS:
    for selection_method in Globs.SELECTION_METHODS:
        model_dict = get_models_feat_removal(dataset, selection_method,
            wandb_train_tag=Globs.TAG_FEATREM,
            model_name='gbr', sort_metric='best-nrmse_maxmin')
        results={
            'DATASET': dataset,
            'selection_method': selection_method,
            
            'nrmse_maxmin': model_dict['best-nrmse_maxmin'],
            'nrmse_maxmin_std': model_dict['best-nrmse_maxmin_std'],
            'nrmse_maxmin_95': conf_range_gen(model_dict['best-nrmse_maxmin'], 
                model_dict['best-nrmse_maxmin_95']),

            'nd': model_dict['best-nd'],
            'nd_std': model_dict['best-nd_std'],
            'nd_95': conf_range_gen( model_dict['best-nd'], model_dict['best-nd_95'])
        }
        results_list.append(results)

df_results = pd.DataFrame(results_list)
df_results.to_csv('reports/feature_selection.csv')
