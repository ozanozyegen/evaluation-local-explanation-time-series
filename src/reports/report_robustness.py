""" Measure robustness in the experiments

"""
#%%
from configs.defaults import Globs
from reports.wandb_queries import get_wandb_df
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os
matplotlib.rcParams.update({
    'font.size': 14,
    'legend.fontsize':10
    })

# Configuration
Globs.TAG_EVAL = 'robv2'
# Remove Random from methods
Globs.MODELS = ['lstm','gbr']
Globs.METHODS = ['global_mean_replacement', 'local_mean_replacement', 'shap']
Globs.METHOD_NAMES = ['Global Mean Replacement', 'Local Mean Replacement', 'SHAP']

# Save information
SAVE_DIR = 'reports/figures/sensitivity/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
#%%
def _filter_nans(lst):
    """ Remove null idxs from list """
    for idx, item in enumerate(lst):
        if pd.isnull(item):
            lst.pop(idx)
    return lst

def get_scores(df, eval_method, dataset, method, model, top_n=3):
    """ Get the model scores for different hyperparameters for the given configuration 
    Average the scores
    """
    df_models = df.loc[(df['MODEL'] == model) & (df['DATASET'] == dataset) & \
        (df['METHOD_NAME'] == method) & (df['EVAL_METHOD_NAME'] == eval_method)]
    df_models_top_n = df_models.iloc[::-1][:top_n]
    tops =    _filter_nans(df_models_top_n.top.tolist())
    bottoms = _filter_nans(df_models_top_n.bottom.tolist())
    # Replace with zeros if exp not complete
    if len(tops)<top_n or len(bottoms)<top_n: # If no exp return 0
        print('Zero experiments detected')
        return [0]*top_n
    else:
        return [(np.abs(top)+np.abs(bottom))/2 for top, bottom in zip(tops, bottoms)]


df = get_wandb_df(Globs.PROJECT_NAME)
# df = df[df.tags.apply(lambda x: Globs.TAG_EVAL in x)]

scores = {}
for eval_method in Globs.EVAL_METHODS:
    for dataset in Globs.DATASETS:
        dfs = df[df.tags.apply(lambda x: 'robv7' in x)]
        for method in Globs.METHODS:
            for model in Globs.MODELS:
                scores[(eval_method, dataset, method, model)] = \
                    get_scores(dfs, eval_method, dataset, method, model)

#%%
model = 'lstm'
for eval_method, eval_method_name in zip(Globs.EVAL_METHODS, ['AOPCR', 'APT']):
    dfs = []
    for dataset in Globs.DATASETS:
        methods_scores = [scores[(eval_method, dataset, method, model)] for method in Globs.METHODS]
        df = pd.DataFrame(np.stack(methods_scores, axis=-1), \
            columns=Globs.METHOD_NAMES).assign(DATASET=dataset.capitalize())
        dfs.append(df)
    cdf = pd.concat(dfs)
    mdf = pd.melt(cdf, id_vars=['DATASET'], var_name='Method')
    ax = sns.boxplot(x="DATASET", y="value", hue="Method", data=mdf)  
    plt.title(eval_method_name)
    plt.subplots_adjust(left=0.13)
    if eval_method == 'aopcr':
        plt.ylim(0, 0.3)
    else:
        plt.ylim(0, 1)
    plt.ylabel(f"{eval_method_name} Score")
    plt.savefig(f"{SAVE_DIR}{model}_{eval_method_name}.pdf", format='pdf')
    plt.show()
    plt.clf()
    plt.close()
# %%
model = 'gbr'
for eval_method, eval_method_name in zip(Globs.EVAL_METHODS, ['AOPCR', 'APT']):
    dfs = []
    for dataset in Globs.DATASETS:
        methods_scores = [scores[(eval_method, dataset, method, model)] for method in Globs.METHODS]
        df = pd.DataFrame(np.stack(methods_scores, axis=-1), \
            columns=Globs.METHOD_NAMES).assign(DATASET=dataset.capitalize())
        dfs.append(df)
    cdf = pd.concat(dfs)
    mdf = pd.melt(cdf, id_vars=['DATASET'], var_name='Method')
    ax = sns.boxplot(x="DATASET", y="value", hue="Method", data=mdf)  
    plt.title(eval_method_name)
    plt.subplots_adjust(left=0.13)
    if eval_method == 'aopcr':
        plt.ylim(0, 0.3)
    else:
        plt.ylim(0, 1)
    plt.ylabel(f"{eval_method_name} Score")
    plt.savefig(f"{SAVE_DIR}{model}_{eval_method_name}.pdf", format='pdf')
    plt.show()
    plt.clf()
    plt.close()

# %%
# %%
