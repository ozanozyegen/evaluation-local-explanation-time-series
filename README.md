# Evaluation of local explanations for time series experiments

**Paper link**: https://link.springer.com/article/10.1007/s10489-021-02662-2

## Experiments
Evaluation of local explanations is performed in three steps: <br>
1- Train the models for the datasets <br>
2- Generate feature importances using local explanation methods <br>
3- Evaluate feature importances <br>

### Steps
You can easily reproduce the experiments using the Makefile.
- Train models
    - `make train_models`
- Evaluate local explanations
    - `make eval_metrics`
- Report the training results
    - `make report_train`

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
