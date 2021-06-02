""" Compute eval_metrics for different hyperparameters, 
    eval robustness of the proposed metrics
    Results will then be calculated in reports/measure_robustness.py
"""
from train.eval_metrics import eval_metric
from joblib import Parallel, delayed
from reports.wandb_queries import get_wandb_df
from configs.defaults import Globs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', help='delimited list input', type=str,
    default=",".join(Globs.DATASETS))
parser.add_argument('-m', '--models', help='delimited list input', type=str,
    default=",".join(['lstm', 'gbr']))
args = parser.parse_args()

# Configuration
Globs.TAG_EVAL = 'robv7'
Globs.TAG_TRAIN = 'trainv7'
datasets = args.datasets.split(',')
models = args.models.split(',')
methods = ['shap','global_mean_replacement','local_mean_replacement'] # No need to compute random
sort_feature = 'best-nrmse_maxmin'

def get_best_model_ids(df, config, top_n=9999):
    """ Return the top N models ids for given architecture
    """
    df = df[df.tags.apply(lambda x: Globs.TAG_TRAIN in x)]
    df = df[df['val_loss'].notna()]
    df_models = df.loc[(df['MODEL'] == config['MODEL']) & (df['DATASET'] == config['DATASET'])]
    df_models.sort_values(sort_feature, inplace=True)
    return df_models.id.tolist()[:top_n]
df_all = get_wandb_df(Globs.PROJECT_NAME)

# Compute all for lstm and gbr models
# Configuration
configs = []
# best_models_ids = { model:get_best_model_ids(model, top_n=5) for model in models}
# Create exp configs
# ~120 exps
for dataset in datasets:
    for model in models:
        for model_id in get_best_model_ids(df_all, {'MODEL':model, 'DATASET':dataset}, top_n=5):
            for method_name in methods:
                for eval_method_name in Globs.EVAL_METHODS:
                    configs.append({
                        'DATASET':dataset,
                        'MODEL':model,
                        'MODEL_ID':model_id,
                        'METHOD_NAME':method_name,
                        'EVAL_METHOD_NAME':eval_method_name
                    })

# Get crashed models and reevaluate
# configs = []
# df = df_all[df_all.tags.apply(lambda x: Globs.TAG_EVAL in x)]
# crashed_models, trained_models = df[df.bottom.isna()], df[~df.bottom.isna()]
# # Remove the crash models that have trained versions
# for crash_model_dict in crashed_models.to_dict('records'):
#     # match_keys = ['DATASET', 'MODEL', 'MODEL_ID', 'METHOD_NAME', 'EVAL_METHOD_NAME']
#     # query = " & ".join([f"{match_key} == {crash_model_dict[match_key]}" for match_key in match_keys])
#     similar_trained_models = trained_models[(trained_models['DATASET'] == crash_model_dict['DATASET']) & 
#                         (trained_models['MODEL'] == crash_model_dict['MODEL']) & 
#                         (trained_models['MODEL_ID'] == crash_model_dict['MODEL_ID']) & 
#                         (trained_models['METHOD_NAME'] == crash_model_dict['METHOD_NAME']) & 
#                         (trained_models['EVAL_METHOD_NAME'] == crash_model_dict['EVAL_METHOD_NAME'])]
#     if len(similar_trained_models) == 0:
#         configs.append({
#                         'DATASET':crash_model_dict['DATASET'],
#                         'MODEL':crash_model_dict['MODEL'],
#                         'MODEL_ID':crash_model_dict['MODEL_ID'],
#                         'METHOD_NAME':crash_model_dict['METHOD_NAME'],
#                         'EVAL_METHOD_NAME':crash_model_dict['EVAL_METHOD_NAME']
#                     })
#     else:
#         print(f"""Previously crashed model {crash_model_dict['id']} retrained. You can delete it from Wandb""")



Parallel(n_jobs=8)(delayed(eval_metric)(config, config['MODEL_ID'], \
    wandb_eval_tag=Globs.TAG_EVAL, wandb_train_tag=Globs.TAG_TRAIN) for config in configs)