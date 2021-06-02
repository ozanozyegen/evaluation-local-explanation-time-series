import numpy as np
from metrics.omission import omission_mean_replacement
from metrics.shapdeep import get_deep_shap_feature_importances, get_tree_shap_feature_importances
from metrics.eval_methods import aopc_regression, switching_point


def select_samples(train_x, train_y, n_samples: int, random_seed=0):
    """ 
    Selects n_samples randomly from the training set
    random seed: Select 0 to reproducability, 
        choose a different value to run the evaluation with different set of samples
    """
    n_samples = train_x.shape[0] if n_samples > train_x.shape[0] else n_samples
    if random_seed == 0: # Fixed results
        np.random.seed(0)
        random_row_indices = np.random.choice(train_x.shape[0], size=n_samples, replace=False)
    else: # Different results at each run
        # Use temporary random state to select the same sample states
        rng = np.random.RandomState(random_seed)
        random_row_indices = rng.choice(train_x.shape[0], size=n_samples, replace=False)

    train_x, train_y = train_x[random_row_indices], train_y[random_row_indices]
    return train_x, train_y

def get_feature_importances(method_name, model_name, params, model, train_x, train_y):
    target_size = train_y.shape[1]
    if method_name == 'random': # (target_size, sample_size, history_size, n_features)
        return np.zeros((train_y.shape[1], train_x.shape[0], train_x.shape[1], train_x.shape[2]))
    if method_name == 'shap' and (model_name == 'tdnn' or model_name == 'lstm'):
        imp_features = get_deep_shap_feature_importances(model, train_x)
    elif method_name == 'global_mean_replacement':
        imp_features = omission_mean_replacement(model, train_x, train_y, target_size, 'global')
    elif method_name == 'local_mean_replacement':
        imp_features = omission_mean_replacement(model, train_x, train_y, target_size, 'local')
    elif method_name == 'shap' and model_name == 'gbr':
        imp_features = get_tree_shap_feature_importances(model, train_x, target_size)
    else:
        raise ValueError('Unknown method_name')
    print(f"Loaded {method_name} features")
    assert imp_features.ndim == 4
    return imp_features

def get_scores(model, train_x, imp_features, method_name, eval_method_name, params, descending):
    target_size = imp_features.shape[0]
    if eval_method_name == 'aopcr':
        return aopc_regression(model, train_x, imp_features, method_name, params['aopcr'], params['aopcr']['top_k'], target_size, descending)
    elif eval_method_name == 'spr':
        return switching_point(model, train_x, imp_features, method_name, params['spr'], threshold=params['spr']['threshold'], 
        target_size=target_size, descending=descending)
    else:
        raise ValueError('Unknown method')

def get_scores_average(scores):
    assert scores.ndim == 2
    avg_over_rows = np.mean(scores, axis=0) # Averaging over mc samples
    assert avg_over_rows.ndim == 1 and avg_over_rows.shape[0] == scores.shape[1]
    avg_over_targets = np.mean(avg_over_rows, axis=0)
    return avg_over_targets