""" Evaluate a metric based on a trained model 
Steps: 
1- Find the id of a best trained model and dataset
2- Load the model and config
3- Compute a metric based on its own configuration
4- Save results as a seperate experiment
"""
import wandb, os
import numpy as np
from data.loaders import dataset_loader
from reports.restore_models import restore_wandb_online
from reports.wandb_queries import get_best_model
from metrics.helpers import select_samples, get_feature_importances, get_scores, get_scores_average
from train.helpers import use_gpu
use_gpu(False)

def evaluate(dataset_name, method_name, eval_method_name, model, train_x, imp_features, params, wandb):
    try:
        method_top_scores = get_scores(model, train_x, imp_features, method_name, eval_method_name, params, descending=True)
        top_score = get_scores_average(method_top_scores)
        wandb.log({'top':top_score})
        method_bottom_scores = get_scores(model, train_x, imp_features, method_name, eval_method_name, params, descending=False)
        bottom_score = get_scores_average(method_bottom_scores)
        wandb.log({'bottom':bottom_score})
    except:
        print(wandb.config)
        raise RuntimeError('get_scores exception')
    # Save artifacts
    np.save(os.path.join(wandb.dir, 'top.npy'), method_top_scores)
    np.save(os.path.join(wandb.dir, 'bottom.npy'), method_bottom_scores)

def eval_metric(config_update, model_id=None, 
        wandb_eval_tag='eval', wandb_train_tag='train'):
    """ Evaluates AOPCR or SPR for given configuration
    config_update (dict): base configuration DATASET,MODEL,METHOD,EVAL_METHOD
    model_id (str): if not defined, best model picked from wandb
        if defined, defined model is used
    """
    # Base Configuration
    config = dict(
        PROJECT_NAME = 'Interpreting_TS',
        params = {
            'n_samples':100,
            'debug':False,
            'aopcr': {'top_k':10, 'sampling':'data', 'z_score':1.96, 'margin': 0.002, 'timeout':60*60*10},
            'spr': {'threshold':0.1, 'sampling':'data', 'z_score':1.96, 'margin': 0.01, 'timeout':60*60*10}
        }
    )
    config.update(config_update)      
    config['MODEL_ID'] = get_best_model(config, only_id=True, wandb_train_tag=wandb_train_tag) if model_id is None else model_id
    # Wandb config
    # os.environ['WANDB_SILENT']='true'
    # os.environ['WANDB_MODE'] = 'dryrun'
    run = wandb.init(project='Interpreting_TS', config=config, tags=[wandb_eval_tag], reinit=True) #load_config()
    config = wandb.config

    model, model_config = restore_wandb_online(config['PROJECT_NAME'], config['MODEL_ID'])
    dataset_params, dataset = dataset_loader[model_config['DATASET']](model_config)
    train_x, train_y, test_x, test_y = dataset['train_x'], dataset['train_y'], dataset['test_x'], dataset['test_y']

    if config.params['n_samples']:
        train_x, train_y = select_samples(train_x, train_y, config.params['n_samples'])
    print(train_x.shape, train_y.shape)

    imp_features = get_feature_importances(config.METHOD_NAME, config.MODEL, config.params, model, train_x, train_y)
    evaluate(config.DATASET, config.METHOD_NAME, config.EVAL_METHOD_NAME, model, train_x, imp_features, config.params, run)

