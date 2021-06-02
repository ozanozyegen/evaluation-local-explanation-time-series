import numpy as np
from tqdm import tqdm

def omission_mean_replacement(model, train_x, train_y, target_size, type_replacement):
    """ Apply global_mean_replacement to each sample to get feature importance
    model: Keras model that have output size of target_size
    train_x: (num_samples, num_timesteps, num_features)
    train_y: (num_samples, target_size)
    target_size: Model's output size
    type_replacement: local or global mean replacement
    """
    num_features, num_timesteps = train_x.shape[2], train_x.shape[1]
    if type_replacement == 'global':
        feature_mean_list = [ np.mean(train_x[:, :, i]) for i in range(num_features) ]

    feature_importances = np.empty((target_size, *train_x.shape)) # (target_size, sample_size, history_size, feature_size)
    for i, sample in enumerate(train_x):
        if type_replacement == 'local':
            feature_mean_list = [ np.mean(train_x[i, :, j]) for j in range(num_features)]
        modified_samples = np.repeat(sample[None, :, :], repeats=num_features*num_timesteps ,axis=0)

        lag_idxs = np.repeat(np.arange(num_timesteps), repeats=num_features)
        feature_idxs = np.tile(np.arange(num_features), num_timesteps)
        mod_sample_idxs = np.arange(len(modified_samples))
        update_idxs = np.concatenate((mod_sample_idxs[:, None], lag_idxs[:, None], feature_idxs[:, None]), axis=1)
        feature_values = np.tile(feature_mean_list, num_timesteps) # (sample_size, history_size, feature_size)
        modified_samples[tuple(update_idxs.T)] = feature_values
        
        preds, true = model.predict(modified_samples), train_y[i][None, :] # (mod_sample_size, target_size), (1, target_size)
        change = true - preds
        feature_idxs = tuple(update_idxs[:, 1:].T)
        feature_importances[:, i, feature_idxs[0], feature_idxs[1]] = change.T # (12, 336)
    return feature_importances