import numpy as np
import pickle, os
from sklearn.ensemble import GradientBoostingRegressor
from models.losses import nd_np, nrmse_maxmin_np

class MultiGradientBoostingRegressorRemoval:
    def __init__(self, config, imp_features=None):
        """ Multi-out Gradient Boosted Regressor model
        """
        self._conf = config
        self._gbr = [GradientBoostingRegressor(n_estimators=config['NUM_TREES'], 
                                        max_depth=config['MAX_DEPTH'], 
                                        max_leaf_nodes=config['MAX_LEAF_NODES'], 
                                        min_samples_split=config['MIN_SAMPLES_SPLIT'])
                        for i in range(config['TARGET_SIZE'])]
        
        self._target_size = config['TARGET_SIZE']
        self._flatten = lambda x: x.reshape(x.shape[0], -1)
        self._imp_features = imp_features
        if imp_features is not None:
            if imp_features.ndim == 4: # For SHAP average over samples
                self._imp_features = imp_features.mean(axis=1)
            self._importance_target = self._get_features_to_remove(self._imp_features)
    
    def fit(self, train_x, train_y, validation_data=None, wandb=None):
        if self._imp_features is not None: # For feature removal
            train_x_generator = self._create_removed_dataset(train_x)
            for target_time, train_x_rem in enumerate(train_x_generator):
                self._gbr[target_time].fit(
                    train_x_rem, train_y[:, target_time])
        else: # With all features
            train_x = train_x.reshape(train_x.shape[0], -1)
            for target_time in range(self._target_size):
                self._gbr[target_time].fit(train_x, train_y[:, target_time])
        if wandb is not None:
            y_pred = self.predict(train_x)
            wandb.log(self._compute_errors(train_y, y_pred))
            if validation_data is not None:
                y_pred = self.predict(validation_data[0])
                wandb.log(self._compute_errors(validation_data[1], y_pred, prefix='val_'))
            # Save model
            pickle.dump(self._gbr, open(os.path.join(wandb.run.dir, 'model-best.h5'), 'wb'))
            pickle.dump(self._imp_features, open(os.path.join(wandb.run.dir, 'imp_features.npy'), 'wb'))
            
        
    def predict(self, x: np.ndarray):
        """ Predict and combine predictions of gbr models
        """
        preds = []
        if self._imp_features is not None: # For feature removal
            x_generator = self._create_removed_dataset(x)
            for target_time, x_rem in enumerate(x_generator):
                preds.append(self._gbr[target_time].predict(x_rem))
        else: # With all features
            x = x.reshape(x.shape[0], -1)
            for target_time in range(self._target_size):
                preds.append(self._gbr[target_time].predict(x))
        pred_y = np.stack(preds, axis=-1)    
        assert pred_y.shape[0] == x.shape[0] and pred_y.shape[1] == self._target_size
        return pred_y
    
    @staticmethod
    def _compute_errors(y_true, y_pred, prefix=''):
        return {
            prefix + "nd" : nd_np(y_true, y_pred),
            prefix + "loss" : nrmse_maxmin_np(y_true, y_pred)
        }

    def _create_removed_dataset(self, x):
        """ Generator that yields dataset with important features removed for each target """
        n_feat_remove = x.shape[1] * x.shape[2] - 10  # Only top 10 significant features remain
        for target_time in range(self._target_size):
            sorted_idxs = self._importance_target[target_time]
            remove_feature_list = []
            # Remove top N insignificant features
            for timestep, feat_idx in sorted_idxs[-n_feat_remove:]:
                remove_feature_list.append(timestep*x.shape[2] + feat_idx)
            # Flaten and remove features from the dataset
            new_x = x.reshape(x.shape[0], -1)
            new_x_removed = np.delete(new_x, remove_feature_list, axis=1)
            yield new_x_removed

    def _get_features_to_remove(self, imp_features):
        """ Find important feat idxs for each target 
        Arguments:
            imp_features (np.array):
        Returns:
            importance_target (np.array): 2d idxs sorted by absolute importance
            (target_size,history_size*feature_size,history_size*feature_size)
        """
        importance_target = {}

        def argsort(x): return np.dstack(
            np.unravel_index(np.argsort(x.ravel()), x.shape))
        for target_time in range(self._target_size):
            target_feat_imp = imp_features[target_time]
            sorted_idxs = argsort(np.abs(target_feat_imp))[0][::-1]
            importance_target[target_time] = sorted_idxs
        return importance_target

    def load_weights(self, MODEL_PICKLE_PATH):
        self._gbr = pickle.load(open(MODEL_PICKLE_PATH, 'rb'))

    def load_feat_imp(self, PICKLE_PATH):
        self._imp_features = pickle.load(open(PICKLE_PATH, 'rb'))
        if self._imp_features.ndim == 4: # For SHAP average over samples
            self._imp_features = self._imp_features.mean(axis=1)
        self._importance_target = self._get_features_to_remove(self._imp_features)
        
