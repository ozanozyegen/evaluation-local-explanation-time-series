from train.eval_metrics import eval_metric
from joblib import Parallel, delayed
from reports.wandb_queries import get_wandb_df, get_scores
from configs.defaults import Globs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', help='delimited list input', type=str,
    default=",".join(Globs.DATASETS))
parser.add_argument('-m', '--models', help='delimited list input', type=str,
    default=",".join(Globs.MODELS))
args = parser.parse_args()

Globs.TAG_EVAL = 'evalv7'
Globs.TAG_TRAIN = 'trainv7'
datasets = args.datasets.split(',')
models = args.models.split(',')

configs = []
# Compute all
# for dataset in datasets:
#     for model in models:
#         for method_name in ['random', 'shap', 
#             'global_mean_replacement', 'local_mean_replacement']:
#             for eval_method_name in ['aopcr', 'spr']:
#                 configs.append({
#                     'DATASET':dataset,
#                     'MODEL':model,
#                     'METHOD_NAME':method_name,
#                     'EVAL_METHOD_NAME':eval_method_name
#                 })

# Parallel(n_jobs=8)(delayed(eval_metric)(config, \
#     wandb_eval_tag=Globs.TAG_EVAL, wandb_train_tag=Globs.TAG_TRAIN) for config in configs)

# Report results
all_df = get_wandb_df(Globs.PROJECT_NAME)
df = all_df[all_df.tags.apply(lambda x: Globs.TAG_EVAL in x)]
dataset_names = ['ohio']#['electricity','rossmann', 'walmart', 'ohio']
eval_method_names = ['aopcr', 'spr']
models = ['tdnn', 'lstm', 'gbr']
method_names = ['random', 'global_mean_replacement', 'local_mean_replacement', 'shap']
tidy_method_names = ['Random', 'Omission (Global)', 'Omission (Local)', 'SHAP']

# Print for latex table
for dataset in dataset_names:
    for eval_method_name in eval_method_names:
        print(dataset, ' ', eval_method_name)
        for method_name, tidy_method_name in zip(method_names, tidy_method_names):
            print(tidy_method_name, end='')
            for model in models:
                exp = get_scores(df, dataset, model, method_name, eval_method_name)
                if exp is None:
                    print(' & - & - ', end='')
                else:
                    if eval_method_name == 'aopcr':
                        print(" & {0:.5f} & {1:.5f} ".format(abs(exp['top']), abs(exp['bottom'])), end='')
                    elif eval_method_name == 'spr':
                        print(" & {0:.3f} & {1:.3f} ".format(abs(exp['top']), abs(exp['bottom'])), end='')
            
            print(r" \\ \hline")

            


        
