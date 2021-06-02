# Welcome to the MakeFile
name := xai

# Setup the folder hierarchy
setup: 
	@mkdir data
	@mkdir data/external
	@mkdir data/processed
	@mkdir data/interim
	@mkdir data/raw
	@mkdir models
	@mkdir notebooks
	@mkdir references
	@mkdir reports
	@mkdir reports/figures
	@mkdir src
	@mkdir src/data
	@mkdir src/features
	@mkdir src/models
	@mkdir src/visualization
	@echo "PYTHONPATH=./src" > .env

# After setting up the datasets run the following files to run the experiments
train_models:
	@python src/train/train_models.py
train_feature_selection: # Feature selection experiments
	@python src/train/train_feature_selection.py

# Evaluates feature selection methods for all models and datasets
# Can take a day to run, parallelization possible across threads
eval_metrics: 
	@python src/train/eval_metrics_multi.py
# Evaluate robustness of the evaluation metrics with respect to model params
eval_robustness: 
	@python src/train/eval_robustness.py
# Report results
report_train:
	@python src/reports/report_train_results.py
	@python src/reports/report_best_hyperparameters.py
report_feature_selection:
	@python src/reports/report_feature_selection.py
report_robustness:
	@python src/reports/report_robustness.py

# Extra
# Clean __pycache___ for unix
clean_unix:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Clean __pycache___ for windows
clean_win:
	@python -c "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
	@python -c "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"

# Anaconda export environment
export:
	@conda env export > $(name).yml

# Anaconda load environment
load:
	@conda env create -f $(name).yml

# Run Tensorboard
tensorboard:
	@tensorboard --logdir outputs/

