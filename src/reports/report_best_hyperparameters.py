""" Report best hyperparameter combination for the top performing models
"""
from configs.model_hyper import _lstm_configs, _tdnn_configs, _gbr_configs
from configs.defaults import Globs
from reports.wandb_queries import get_best_model
from reports.restore_models import restore_wandb_online
import pandas as pd

Globs.TAG_TRAIN = 'trainv2'
dataset_names = Globs.DATASETS

model_hypers = dict(
    lstm = list(_lstm_configs.keys()),
    tdnn = list(_tdnn_configs.keys()),
    gbr = list(_gbr_configs.keys())
)

results = []
for dataset_name in dataset_names:
    print(dataset_name)
    for model_name, model_params in model_hypers.items():
        print(model_name)
        config={'PROJECT_NAME':Globs.PROJECT_NAME, 'DATASET':dataset_name, 'MODEL':model_name }
        model_config = get_best_model(config, only_id=False, wandb_train_tag=Globs.TAG_TRAIN)
        config.update({hyper:model_config[hyper] for hyper in model_params})
        results.append(config)

df_results = pd.DataFrame(results)
df_results.to_csv('reports/best_hyperparameters.csv')