import itertools

def _get_permutations(hyper_config_dict):
    configs = []
    keys, values = zip(*hyper_config_dict.items())
    for experiment in itertools.product(*values):
        configs.append({key:value for key, value in zip(keys, experiment)})
    return configs

_lstm_configs = dict(
    NUM_LAYERS = [1, 2, 3, 4],
    NUM_UNITS = [16, 32, 64, 128],
    DROPOUT = [0, 0.2, 0.5],
)

_tdnn_configs = dict(
    NUM_LAYERS = [1, 2, 3, 4],
    NUM_UNITS = [16, 32, 64, 128],
    DROPOUT = [0, 0.2, 0.5],
)

_gbr_configs = dict(
    NUM_TREES = [10, 100, 200],
    MAX_DEPTH = [2, 3, 4, 5],
    MIN_SAMPLES_SPLIT = [2, 5, 10, 15],
    MAX_LEAF_NODES = [None]
)

hyper_configs = dict(
    lstm = _get_permutations(_lstm_configs),
    tdnn = _get_permutations(_tdnn_configs),
    gbr = _get_permutations(_gbr_configs)
)