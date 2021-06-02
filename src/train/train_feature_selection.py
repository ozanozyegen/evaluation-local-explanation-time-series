""" For GBR and SHAP pick top 10 features and retrain
    Report training time NRMSE and ND with mean and std
"""
from configs.defaults import model_defaults, dataset_defaults, Globs
from reports.wandb_queries import get_best_model
from reports.restore_models import restore_wandb_online
from metrics.helpers import get_feature_importances, select_samples
from train.helpers import log_errors
from data.loaders import dataset_loader
import numpy as np
import matplotlib.pyplot as plt
from models.gbr_model import MultiGradientBoostingRegressorRemoval
import wandb, os, time, pprint, argparse
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', help='delimited list input', type=str,
    default=",".join(Globs.DATASETS))
args = parser.parse_args()

# Wandb run local config
# os.environ['WANDB_SILENT']='true'
# os.environ['WANDB_MODE'] = 'dryrun'

Globs.TAG_TRAIN = 'trainv2'
Globs.TAG_FEATREM = 'feat_remv2'
datasets = args.datasets.split(',')

def train_model(config, imp_features, dataset, selection_method):
    """ Train GBR model on full or removed dataset
    Parameters:
        config: GBR model configuration
        imp_features (np.array): (target_size, n_samples, n_features),
            set None if selection_method is None
        selection_method (str): None, or the feature selection method name

    """
    if selection_method == 'full':
        model = MultiGradientBoostingRegressorRemoval(config)
    else:
        assert imp_features is not None
        model = MultiGradientBoostingRegressorRemoval(config, imp_features)

    run=wandb.init(project=Globs.PROJECT_NAME, config=config,
        tags=[Globs.TAG_FEATREM], reinit=True)
    wandb.config.update({'selection_method':selection_method})
    model.fit(dataset['train_x'], dataset['train_y'],
        validation_data=(dataset['test_x'], dataset['test_y']), wandb=wandb)
    log_errors(dataset, model, wandb)


def f_regression_wrapper(train_x, train_y):
    a = f_regression(train_x, train_y)
    return np.nan_to_num(a[0])

def mutual_info_wrapper(train_x, train_y, random_state=0):
    return mutual_info_regression(train_x, train_y, random_state=random_state)

def get_tree_feature_importance_(model:MultiGradientBoostingRegressorRemoval):
    feat_vals=[gbr_model.feature_importances_ for gbr_model in model._gbr]
    return np.stack(feat_vals, axis=0)

def get_alt_feature_importances(model, train_x, train_y):
    """ Compute important features using alternative feature selection methods
    Returns:
        feat_methods (dict): keys method_names
            values imp_scores (target_size, history_size, n_features)
    """
    target_size = train_y.shape[1]
    org_shape = train_x.shape
    flatten = lambda x: x.reshape(x.shape[0], -1)
    feat_methods = {'tree_imp':get_tree_feature_importance_(model).reshape(target_size,org_shape[1],org_shape[2])}
    for feat_imp_method in [mutual_info_wrapper, f_regression_wrapper]:
        feat_vals = []
        for target_time in range(target_size):
            feat_vals.append( feat_imp_method(flatten(train_x), train_y[:,target_time]).reshape(org_shape[1:]) )
        feat_vals = np.stack(feat_vals, axis=0)
        feat_methods[feat_imp_method.__name__] = feat_vals
    return feat_methods

def rank_features(feat_imp_method_dict, feature_names):
    """ For each imp method, ranks features by importance
    Parameters:
        feat_imp_method_dict: imp_features (target_size, history_size, n_features)
        feature_names: Feature names in the dataset same order
    """
    rank_dict = {}
    for feat_method_name, imp_features in feat_imp_method_dict.items():
        assert imp_features.ndim == 3
        sum_features = imp_features.sum(axis=0).sum(axis=0)
        sorted_idxs = np.argsort(sum_features)[::-1]
        sorted_features = [feature_names[idx] for idx in sorted_idxs]
        rank_dict[feat_method_name] = sorted_features, np.sort(sum_features)[::-1]
    return rank_dict


if __name__ == "__main__":
    model_name = 'gbr'
    results = {}
    for dataset_name in datasets:
        config = {'PROJECT_NAME':Globs.PROJECT_NAME,
            'MODEL':model_name, 'DATASET':dataset_name}
        best_model_id = get_best_model(config, only_id=True, wandb_train_tag=Globs.TAG_TRAIN)
        best_model, config = restore_wandb_online(Globs.PROJECT_NAME, best_model_id)
        dataset_params, dataset = dataset_loader[dataset_name](config)
        train_x, train_y, test_x, test_y = dataset['train_x'], dataset['train_y'], dataset['test_x'], dataset['test_y']


        n_samples, dataset_type = 100, 'train'
        params = {'n_samples':n_samples,'debug':False, 'dataset_type':dataset_type}
        config.update(params)

        if params['dataset_type'] == 'train':
            X_sub, y_sub = select_samples(train_x, train_y, n_samples=params['n_samples'])
        elif params['dataset_type'] == 'test':
            X_sub, y_sub = select_samples(test_x, test_y, n_samples=params['n_samples'])

        feat_imp_method_dict = {}
        feat_imp_method_dict = get_alt_feature_importances(best_model, X_sub, y_sub)
        feat_imp_method_dict['shap'] = np.abs(get_feature_importances('shap', model_name, params, \
            best_model, X_sub, y_sub)) # 4D (target_size, n_samples, history_size, n_features)
        # # Train with full dataset
        # feat_imp_method_dict['full'] = None
        
        # Train models
        # for feat_method_name, imp_features in feat_imp_method_dict.items():
        #     train_model(config, imp_features, dataset, feat_method_name)
        
        # Rank global features
        print(dataset_name)
        feat_imp_method_dict['shap'] = feat_imp_method_dict['shap'].mean(axis=1) # Mean over samples
        feature_rank_dict = rank_features(feat_imp_method_dict, dataset_params['FEATURES'])
        pprint.pprint(feature_rank_dict)
        results[dataset_name] = feature_rank_dict

    # Normalize for each feature imp method and show
    SAVE_DIR = 'reports/figures/feature_rank/'
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    for dataset_name, feat_rank_dict in results.items():
        print(dataset_name)
        fig, ax = plt.subplots()
        sorted_features = np.array([val[0] for val in feat_rank_dict.values()]).T
        min_max_norm = lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
        sorted_features_scores = min_max_norm(np.array([val[1] for val in feat_rank_dict.values()]).T)
        ax = sns.heatmap(sorted_features_scores, annot=sorted_features, fmt='', xticklabels=list(feature_rank_dict.keys()))
        fig.subplots_adjust(bottom=0.45)
        plt.savefig(f'{SAVE_DIR}{dataset_name}.jpg', format='jpg')
        plt.clf()
        



