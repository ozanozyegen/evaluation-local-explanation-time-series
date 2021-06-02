from models.gbr_model import MultiGradientBoostingRegressorRemoval
from models.tf_models import *

model_loader = dict(
    lstm = hyper_lstm,
    tdnn = hyper_tdnn,
    gbr = MultiGradientBoostingRegressorRemoval
)








