import os
import yaml, wandb
from models.loaders import model_loader

def restore_wandb_online(project_name, model_id, TEMP_DATA_PATH='wandb/'):
    """
    Returns:
    :model: Model file
    :config: Exp config file
    """
    TEMP_DATA_PATH = os.path.join(TEMP_DATA_PATH, model_id)
    best_model_wb = wandb.restore('model-best.h5', 
        run_path=f"oozyegen/{project_name}/{model_id}", root=TEMP_DATA_PATH)
    config_wb = wandb.restore('config.yaml', 
        run_path=f"oozyegen/{project_name}/{model_id}", root=TEMP_DATA_PATH)
    # Restore config as a dict
    config = yaml.load(open(os.path.join(TEMP_DATA_PATH, 'config.yaml')), Loader=yaml.FullLoader)
    for k in list(config.keys()):
        if type(config[k]) is dict and 'value' in config[k]:
            config[k] = config[k]['value']
    model = model_loader[config['MODEL']](config)
    model.load_weights(best_model_wb.name)
    if config['MODEL']=='gbr' and \
        'selection_method' in config and config['selection_method'] != 'full':
        imp_features = wandb.restore('imp_features.npy', 
        run_path=f"oozyegen/{project_name}/{model_id}", root=TEMP_DATA_PATH)
        model.load_feat_imp(imp_features.name)
    return model, config

def restore_wandb_file(WANDB_LOCAL_EXP_FOLDER_DIR):
    """
    Returns:
    :model: Model file
    :config: Exp config file
    """
    config = yaml.load(open(os.path.join(WANDB_LOCAL_EXP_FOLDER_DIR, 'config.yaml')), Loader=yaml.FullLoader)
    for k in list(config.keys()):
        if type(config[k]) is dict and 'value' in config[k]:
            config[k] = config[k]['value']
    model = model_loader[config['MODEL']](config)
    model.load_weights(f"{WANDB_LOCAL_EXP_FOLDER_DIR}/model-best.h5")
    return model, config
    