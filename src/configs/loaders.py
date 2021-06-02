from configs.defaults import hyperparameter_defaults
import yaml, os
from typing import Union

def load_config(config: Union[str, dict]):
    """ Add configuration file or dict on top of defaults and return """
    default_config = hyperparameter_defaults
    if isinstance(config, str):
        config = yaml.load(open(os.path.join("src/configs", config)), Loader=yaml.FullLoader)
    default_config.update(config)
    return default_config

