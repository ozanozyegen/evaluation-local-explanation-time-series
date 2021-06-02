from train.train_model import trainer
from configs.model_hyper import hyper_configs
from reports.wandb_queries import get_best_model
from configs.defaults import Globs, model_defaults, dataset_defaults
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', help='delimited list input', type=str,
    default=",".join(Globs.DATASETS))
parser.add_argument('-m', '--models', help='delimited list input', type=str,
    default=",".join(Globs.MODELS))
args = parser.parse_args()

# 48*6 = 288 = 288 models
# Took ~14 hours on dsl08 with single GPU
# Train models
datasets = args.datasets.split(',')
models = args.models.split(',')
# Create dataset model configurations
model_configs = []
for dataset in datasets:
    for model in models: #models:
        config = {**dataset_defaults[dataset], **model_defaults[model]}
        model_configs.append(config)

# # Train hyperparameters for each dataset model configuration
for i in range(64): # Max 64 permutations per model
    for model_config in model_configs:
        all_permutations = hyper_configs[model_config['MODEL']]
        if i < len(all_permutations):
            permutation_config = all_permutations[i]
            model_config.update(permutation_config)
            trainer(model_config, wandb_tags=['trainv7'])




