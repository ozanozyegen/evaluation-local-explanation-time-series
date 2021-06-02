import wandb
import pandas as pd
from configs.defaults import Globs

def get_wandb_df(project_name):
    """ Extracts all the exps under a project from wandb
    """
    # Change oreilly-class/cifar to <entity/project-name>
    api = wandb.Api()
    runs = api.runs(f"oozyegen/{project_name}")
    summary_list = [] 
    config_list = [] 
    name_list, tags_list = [], []
    for run in runs: 
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 

        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 

        name_list.append(run.id)  # run.name is the name of the run.
        tags_list.append(run.tags) # run.tags are the tags associated with the run

    import pandas as pd 
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'id': name_list, 'tags': tags_list}) 
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    return all_df

def get_best_model(config, only_id=True, wandb_train_tag='train', 
    sort_feature='best-nrmse_maxmin'):
    """ Finds the model with the best nrmse for given dataset and model
    Arguments:
    only_id (bool): If True returns only the id, if False returns all info
    config (dict):
        wandb_train_tag (str): Filter tag before selecting the best model
        if not defined default tag is *train*
        sort_feature (str): The metric used to sort the models
    """
    df = get_wandb_df(config['PROJECT_NAME'])
    df = df[df.tags.apply(lambda x: wandb_train_tag in x)]
    df = df[df['val_loss'].notna()]
    df_models = df.loc[(df['MODEL'] == config['MODEL']) & (df['DATASET'] == config['DATASET'])]
    # sorting metric feature must be available for all models
    assert len(df_models[df_models[sort_feature].notna()]) == len(df_models) 
    df_models.sort_values(sort_feature, inplace=True)
    if only_id:
        model_id = df_models.iloc[0].id
        return model_id
    else:
        return df_models.iloc[0].to_dict()

def get_scores(df, dataset, model, method_name, eval_method_name):
    """ Finds the evaluation experiment for given config
    If none found returns None
    """
    df_models = df.loc[
        (df['MODEL']==model) & (df['DATASET']==dataset) & 
        (df['METHOD_NAME']==method_name) & (df['EVAL_METHOD_NAME']==eval_method_name)
        ]
    if len(df_models) == 0:
        return None
    else:
        return df_models.iloc[0].to_dict()

def get_models_feat_removal(dataset_name, selection_method,
    wandb_train_tag='feat_rem',
    model_name='gbr', sort_metric='best-nrmse_maxmin'):
    """ Get full and removed models for given dataset and model
        These models have dataset_type = 'full' or 'removed' 
        and tag: feat_rem
        Returns:
            (full_model_id, removed_model_id)
    """
    df = get_wandb_df(Globs.PROJECT_NAME)
    df = df[df.tags.apply(lambda x: wandb_train_tag in x)]
    df = df[df['val_loss'].notna()]
    
    removed_models = df.loc[(df['MODEL'] == model_name) & 
        (df['DATASET'] == dataset_name) & 
        (df['selection_method'] == selection_method) & 
        (df['dataset_type'] == 'train') & 
        (df['n_samples'] == 100)]
    removed_models.sort_values(sort_metric, inplace=True)
    return removed_models.iloc[0].to_dict()

