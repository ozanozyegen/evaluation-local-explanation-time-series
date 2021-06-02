import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time

def sort_shap_vals_for_sample(shap_vals: np.array, sample_id: int, method_name: str, 
                                selected_target = None, descending: bool = True) -> np.array:
    """ Sorts the SHAP values of a selected sample and a selected target
        Return: sorted shap values (num_timesteps, num_features) 
    """
    if method_name == 'random':
        size = shap_vals.shape[2] * shap_vals.shape[3]
        idxs = np.random.choice(size, size=size, replace=False)
        rows, cols = idxs//shap_vals.shape[3], idxs%shap_vals.shape[3]
        return np.stack((rows, cols), axis=-1)
    else:
        target_shap = shap_vals[selected_target] if isinstance(selected_target, int) else shap_vals.mean(axis=0)
        sample_shap = target_shap[sample_id]
        sorted_indices = np.unravel_index(np.argsort(sample_shap, axis=None), sample_shap.shape)
        if descending:
            sorted_indices = sorted_indices[0][::-1], sorted_indices[1][::-1]
        else:
            sorted_indices = sorted_indices[0], sorted_indices[1]
        sorted_indices_arr = np.array(sorted_indices).T
        return sorted_indices_arr


def get_data_dist(train_x):
    """ !!! Deprecated !!!
        Generate a dict where keys are features and values are the data distribution 
        train_x: Numpy array of shape (samples, timesteps, features)
    """
    data_dist = {}
    samples, timesteps, features = train_x.shape
    for timestep in range(timesteps):
        for feature in range(features):
            data_dist[(timestep, feature)] = train_x[:, timestep, feature]
    assert len(data_dist) == timesteps * features
    return data_dist

def check_condition(monte_carlo_list, z_score=1.96, tolerance=0.005):
    """ Checks confidence intervals for each target time 
        monte_carlo_list: (num samples, target times)
        z_score: Z score of STD for confidence intervals
    """
    mc_arr = np.array(monte_carlo_list)
    mu = mc_arr.mean(axis=0)
    std = mc_arr.std(axis=0)
    n = len(mc_arr)
    margin = (z_score * std) / np.sqrt(n)
    return np.all(margin < tolerance)

def sample_replacement(train_x, timestep_idx, feature_idx, sampling):
    if sampling == 'uniform':
        replacement_set = np.unique(train_x[:, timestep_idx, feature_idx])
    elif sampling == 'data':
        replacement_set = train_x[:, timestep_idx, feature_idx]
    else:
        raise ValueError('Unknown params')
    return np.random.choice(replacement_set)



def aopc_regression(model, train_x, shap_vals,  method_name, params, top_k=10, target_size=12, descending=True):
    """ Calculates AOPC for multi-target time series regression using Monte Carlo Samples
        model: Trained Tensorflow model
        train_x: Training input
        shap_vals: Importance values for the model in shape (target_size, num_samples, num_timesteps, num_features)
        top_k: Top k values to remove from the model
        target_size: Time Horizon size that model predicts
        descending: Set False to get the metric by removing negatively contributing features
        Return: AOPC for each target time
    """
    if method_name == 'random': print('Warning: Running AOPC with random selection\n')
    if params['sampling'] == 'uniform':
        unique_features = {i:np.unique(train_x[:,:,i]) for i in range(train_x.shape[2])} # For sampling from uniform 
    
    monte_carlo_list, MAX_ITER_LIMIT = [], 1000000
    timeout = time.time() + params['timeout']
    for count_iter in tqdm(range(MAX_ITER_LIMIT), mininterval=60, maxinterval=120):
        select_sample_idx = np.random.choice(train_x.shape[0])
        preds_arr, mod_preds_arr = np.empty((top_k, target_size)), np.empty((top_k, target_size))
        preds_list, mod_preds_list = [], []

        for selected_target in range(target_size):
            mod_train_x = train_x.copy()
            sorted_imp_idxs = sort_shap_vals_for_sample(shap_vals, select_sample_idx, method_name, selected_target, descending) # (timesteps, features)
            select_sample = mod_train_x[select_sample_idx]
            for k in range(top_k):
                feat_k_idx = sorted_imp_idxs[k, 0], sorted_imp_idxs[k, 1] # (timestep: int, feature: int)
                # select_sample[feat_k_idx[0], feat_k_idx[1]] = np.random.choice(train_x[:, feat_k_idx[0], feat_k_idx[1]])
                select_sample[feat_k_idx[0], feat_k_idx[1]] = sample_replacement(train_x, feat_k_idx[0], feat_k_idx[1], params['sampling'])

                mod_sample, org_sample = select_sample.copy(), train_x[select_sample_idx]
                preds_list.append(org_sample); mod_preds_list.append(mod_sample)

                
        preds, mod_preds = model.predict(np.stack(preds_list)), model.predict(np.stack(mod_preds_list))
        for idx in range(len(preds)):
            k, selected_target = idx % top_k, idx // top_k
            preds_arr[k, selected_target], mod_preds_arr[k, selected_target] = preds[idx, selected_target], mod_preds[idx, selected_target]

        diff_k = preds_arr - mod_preds_arr
        monte_carlo_list.append( diff_k.mean(axis=0) ) # Average over k samples, not target times
        
        if count_iter%100==99 and (check_condition(monte_carlo_list, params['z_score'], params['margin']) or time.time()>timeout):
            print(f'Stopped after {count_iter} mc samples\n')
            if time.time()>timeout:
                print('Stopped due to timeout')
            return np.array(monte_carlo_list)
    raise ValueError(f'Passed max iteration limit: {MAX_ITER_LIMIT}')

def calculate_switching_point(mod_samples, org_preds, model, expected_change, descending, total_features):
    """ Calculate switching points for multiple samples and target times
        Switching point is defined as pred -+ expected_change * pred
    """
    target_scores = {}
    mod_preds = model.predict(np.stack([tup[0] for tup in mod_samples]))
    
    k_found, temp_target = False, None
    for mod_pred, (_, k, selected_target) in zip(mod_preds, mod_samples):
        if k_found and temp_target == selected_target:
            continue
        elif temp_target is not None and temp_target != selected_target and not k_found:
            target_scores[temp_target] = total_features
        temp_target = selected_target
        k_found = False

        if descending and mod_pred[selected_target] < org_preds[selected_target] - org_preds[selected_target] * expected_change or \
        not descending and mod_pred[selected_target] > org_preds[selected_target] + org_preds[selected_target] * expected_change:
            target_scores[selected_target] = k + 1
            k_found = True
    if not k_found: # SP for last target (11) not set
        target_scores[temp_target] = total_features
    return target_scores
    
def switching_point(model, train_x, shap_vals, method_name, params, threshold, target_size, descending):
    if method_name == 'random': print('Warning: Running SPR with random selection\n')
    total_features = shap_vals.shape[2] * shap_vals.shape[3]

    monte_carlo_list, MAX_ITER_LIMIT = [], 1000000
    timeout = time.time() + params['timeout']
    expected_change = threshold

    for count_iter in tqdm(range(MAX_ITER_LIMIT), mininterval=60, maxinterval=120):
        select_sample_idx = np.random.choice(train_x.shape[0])
        mod_train_x = train_x.copy()
        org_sample = train_x[select_sample_idx]
        org_preds = model.predict(org_sample[None, :, :])[0]
        mod_samples = []
        for selected_target in range(target_size):
            mod_sample = mod_train_x[select_sample_idx].copy()
            sorted_imp_idxs = sort_shap_vals_for_sample(shap_vals, select_sample_idx, method_name, selected_target, descending) # (timesteps, features)
            for k in range(total_features):
                feat_k_idx = sorted_imp_idxs[k, 0], sorted_imp_idxs[k, 1] # (timestep: int, feature: int)
                mod_sample[feat_k_idx[0], feat_k_idx[1]] = sample_replacement(train_x, feat_k_idx[0], feat_k_idx[1], params['sampling'])

                mod_samples.append((mod_sample.copy(), k, selected_target))
    
        target_scores = calculate_switching_point(mod_samples, org_preds, model, expected_change, descending, total_features)
        assert len(target_scores) == target_size and list(target_scores.keys())[5] == 5 # Assert target score dict keys are ordered, Python >=3.7 dicts have insertion order
        monte_carlo_list.append(list(target_scores.values()))

        if count_iter%100==99:
            if check_condition(np.array(monte_carlo_list) / total_features, params['z_score'], params['margin']) or \
                time.time()>timeout: # check_condition requires (num_samples, target_times)
                print(f'Stopped after {count_iter} mc samples\n')
                if time.time()>timeout:
                    print('Stopped due to timeout')
                break
    
    return np.array(monte_carlo_list) / total_features
        