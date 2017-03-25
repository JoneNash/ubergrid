import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

# Set up and serialize the estimator.
booster = GradientBoostingClassifier()

joblib.dump(booster, 'booster.pkl')

# Set up the parameter grid.
parameter_grid = {
    "estimator": "booster.pkl",
    "param_grid": {
        "n_estimators": [100, 500, 1000],
        "max_depth": [2, 4, 6]
    },
    "scoring": ["log_loss", "roc_auc", "accuracy"]
}
with open('booster_grid_params.json', 'w') as params_out:
    params_out.write(
        json.dumps(parameter_grid) + "\n")

 