# SHAP DeepExplainer doesn't support TF 2.0. Lines below make TF act like TF 1.x 
# https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # Hide debugging info
# tf.disable_eager_execution()
tf.disable_v2_behavior()

import shap
import numpy as np

def get_deep_shap_feature_importances(model, train_x):
    e = shap.DeepExplainer(model, train_x)
    shap_val = np.array(e.shap_values(train_x))
    return shap_val

def get_tree_shap_feature_importances(model, train_x, target_size):
    org_shape = train_x.shape
    flatten = lambda x: x.reshape(x.shape[0], -1)
    assert len(model._gbr) == target_size

    shap_vals = []
    for target_time in range(target_size):
        e = shap.TreeExplainer(model._gbr[target_time], flatten(train_x))
        shap_val = np.array(e.shap_values(flatten(train_x), check_additivity=False))
        shap_vals.append( shap_val.reshape(org_shape) )
    return np.stack(shap_vals, axis=0)
    