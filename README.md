# Evaluation of local explanations for time series experiments


## Experiments
Evaluation of local explanations is performed in three steps:
1- Train the models for the datasets
2- Generate and save feature importances
3- Evaluate feature importances

### Steps
- Train models
    - make train_models
- Evaluate local explanations
    - make eval_metrics
- Report the training results
    - make report_train

## Data Sources
### Electricity
- https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
### Rossmann
- https://www.kaggle.com/c/rossmann-store-sales
### Walmart
- https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
### Ohio
- You must request access from the following link
- http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

## Requirements
- Python 3.7 and Tensorflow 2.2
- wandb - Weights and Biases is used for tracking the experiments
- xai.yml contains all the package dependencies