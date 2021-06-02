class Globs:
    """ Project globals """
    PROJECT_NAME = 'Interpreting_TS'
    DATASETS = ['ohio', 'walmart', 'electricity', 'rossmann']
    MODELS = ['tdnn', 'lstm', 'gbr']
    METHODS = ['random','shap','global_mean_replacement','local_mean_replacement']
    EVAL_METHODS = ['aopcr', 'spr']
    SELECTION_METHODS = ['full','mutual_info_wrapper', 'f_regression_wrapper', 
    'tree_imp', 'shap']

hyperparameter_defaults=dict(
    BATCH_SIZE = 512,
    EPOCHS = 200,
    PATIENCE = 10,
)

_lstm_defaults = dict(
    MODEL = 'lstm',
    NUM_LAYERS = 2,
    NUM_UNITS = 32,
    DROPOUT = 0
)

_tdnn_defaults = dict(
    MODEL = 'tdnn',
    NUM_LAYERS = 2,
    NUM_UNITS = 32,
    DROPOUT = 0,
)

_gbr_defaults = dict(
    MODEL = 'gbr',
    NUM_TREES = 100,
)

model_defaults = dict(
    lstm = _lstm_defaults,
    tdnn = _tdnn_defaults,
    gbr = _gbr_defaults
)

_elect_defaults = dict(
    DATASET = 'electricity',
    NUM_SERIES = 10,
    HISTORY_SIZE = 168,
    TARGET_SIZE = 12,
    STRIDE = 12,
    CONT_FEATURES = [0],
    CAT_FEATURES = [1,2,3,4,5],
    **hyperparameter_defaults
)

_rossmann_defaults = dict(
    DATASET = 'rossmann',
    NUM_SERIES = 100,
    HISTORY_SIZE = 30,
    TARGET_SIZE = 12,
    STRIDE = 12,
    CONT_FEATURES = [0,2], # Sales, Customers
    CAT_FEATURES = [1,3,4,5,6,7,8,9],
    **hyperparameter_defaults
)

_walmart_defaults = dict(
    DATASET = 'walmart',
    NUM_SERIES = 100, # Store, Dept combinations
    HISTORY_SIZE = 30,
    TARGET_SIZE = 6,
    STRIDE = 1,
    CONT_FEATURES = [0,9],
    CAT_FEATURES = [1,2,3,4,5,6,7,8],
    **hyperparameter_defaults
)

_ohio_defaults = dict(
    DATASET = 'ohio',
    NUM_SERIES = 6,
    HISTORY_SIZE = 60,
    TARGET_SIZE = 6,
    CONT_FEATURES = [0,1,2],
    CAT_FEATURES = [-1],
    SUBJECTS = ['559', '563', '570', '575', '588', '591'],
    STANDARDIZE = False, # if False apply Min-Max Normalization instead
    **hyperparameter_defaults
)

dataset_defaults = dict(
    electricity = _elect_defaults,
    rossmann = _rossmann_defaults,
    walmart = _walmart_defaults,
    ohio = _ohio_defaults
)