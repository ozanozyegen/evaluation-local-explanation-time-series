""" Train the models """
import wandb, yaml, os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from data.loaders import dataset_loader
from models.loaders import model_loader
from models.gbr_model import MultiGradientBoostingRegressorRemoval
from configs.loaders import load_config # File configs
from configs.defaults import Globs, dataset_defaults, model_defaults
from train.helpers import auto_gpu_selection, log_errors
from visualization.vis_preds import visualize_preds
from data.helpers import convert_wandb_config_to_dict
import pickle


# Configuration
# NOTE: Remove the lines below to save experiment results to W&B
# os.environ['WANDB_SILENT']='true'
# os.environ['WANDB_MODE'] = 'dryrun'

def trainer(conf, wandb_tags=['train']):
    auto_gpu_selection()

    wandb.init(project=Globs.PROJECT_NAME, config=conf, tags=wandb_tags, reinit=True) #load_config()
    config = wandb.config

    dataset_params, dataset = dataset_loader[config.DATASET](convert_wandb_config_to_dict(config))
    config.update(dataset_params)
    model = model_loader[config.MODEL](config)
    print(dataset['train_x'].shape, dataset['train_y'].shape,
        dataset['test_x'].shape, dataset['test_y'].shape)

    if isinstance(model, tf.keras.Model):
        print(model.summary())
        callbacks=[ EarlyStopping(patience=config.PATIENCE,
            restore_best_weights=True, monitor='val_loss'),
            wandb.keras.WandbCallback() ]
        model.fit(dataset['train_x'], dataset['train_y'],
            validation_data=(dataset['test_x'], dataset['test_y']),
            batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, callbacks=callbacks)
    elif isinstance(model, MultiGradientBoostingRegressorRemoval):
        model.fit(dataset['train_x'], dataset['train_y'],
            validation_data=(dataset['test_x'], dataset['test_y']),
            wandb=wandb)

    log_errors(dataset, model, wandb)
    visualize_preds(dataset, model, wandb.run.dir)

if __name__ == "__main__":
    # Train a single model
    config = {**dataset_defaults['ohio'], **model_defaults['lstm']} 
    # config.update({'CONT_FEATURES':[0], 'CAT_FEATURES':[3]})
    # config.update({'SUBJECTS':['559', '570', '575', '588'],})
    config.update({'STANDARDIZE': False})
    trainer(config, wandb_tags=['trainv7'])