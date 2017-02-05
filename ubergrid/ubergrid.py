import json
import sys
import os
import numpy as np

from time import time

from pandas import DataFrame, Series, read_csv

from typing import List, Tuple, Dict, Any

from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import SCORERS
from sklearn.base import BaseEstimator


# TODO: Add logging.
# TODO: Validate that the training_set and validation_set frames have the same
# columns.
# TODO: Validate metric names.
# TODO: Add cross validation.
# TODO: Add joblib Parallel to support multiple simultaneous runs.

def _evaluate_model(estimator: BaseEstimator, 
                    X: DataFrame, 
                    y: DataFrame, 
                    metrics: List[str], 
                    prefix: str) -> Dict[str, Any]:
    """ TODO: Docstring.
    """

    predict_times = []
    results = {}

    # Evaluate the score for each metric.
    for metric in metrics:
        # We need the scorer function itself, as well as the type of arguments
        # required by the scorer function.
        metric_fn = SCORERS[metric]

        # The make_scorer function returns a scorer, which takes an estimator,
        # the training set, and the ground truth labels.
        start = time()
        results[prefix + "_" + metric] = metric_fn(estimator, X, y)
        stop = time()

        predict_times += [stop - start]

    results[prefix + "_total_prediction_time"] = \
        sum(predict_times) / len(predict_times)
    results[prefix + "_total_prediction_records"] = X.shape[0]

    return results

def _train_model(estimator: BaseEstimator,
                 X: DataFrame, 
                 y: DataFrame, 
                 metrics: List[str],
                 fit_params: Dict[str, Any]) \
                 -> Tuple[Dict[str, Any], BaseEstimator]:
    """
    TODO: Write docstring.
    """
    fit_start = time()
    estimator.fit(X, y, **fit_params)
    fit_end = time()

    results = _evaluate_model(estimator, X, y, metrics, "training")
    results["training_time_total"] = fit_end - fit_start

    return estimator, results

def _train_and_evaluate(estimator: BaseEstimator,
                        params: Dict[str, Any],
                        model_id: int,
                        training_file: str,
                        output_dir: str,
                        X_train: DataFrame,
                        y_train: DataFrame,
                        target_col: str,
                        metrics: List[str],
                        fit_params:Dict[str, Any],
                        X_validation: DataFrame = None,
                        y_validation: DataFrame = None,
                        validation_file: str = None) -> None:
    """
    TODO: Write docstring.
    """
    model_file = "{}/model_{}.pkl".format(output_dir, model_id)
    results_file = "{}/results_{}.json".format(output_dir, model_id)
        
    # If the results file already exists, skip this pass.
    if os.path.exists(results_file):
        return

    # Initialize the estimator with the params.
    estimator.set_params(**params)

    model, training_results = \
        _train_model(estimator, 
                     X_train, 
                     y_train, 
                     metrics, 
                     fit_params)

    # If the validation set is defined, use _evaluate_model to evaluate the 
    # model. Otherwise this is an empty dict.
    validation_results = \
        _evaluate_model(estimator, 
                        X_validation, 
                        y_validation, 
                        metrics,
                        "validation") \
        if validation_file is not None else {}
    
    # Construct and write the results for this run.
    results = {
        "training_file": training_file,
        "target": target_col,
        "model_file": model_file,
        **training_results,
        **validation_results,
        **params
    }

    # Add the validation set file if present.
    if validation_file:
        results["validation_file"] = validation_file

    # Write the results _after_ the model.
    joblib.dump(estimator, model_file)
    with open(results_file, 'w') as results_out:
        results_out.write(
            json.dumps(results) + "\n")

def _main(search_params_file: str,
          target_col: str,
          training_file: str,
          output_dir: str,
          validation_file: str = None) -> None:
    """
    TODO: Write docstring.
    """

    search_params = json.load(open(search_params_file, 'r'))

    # The output directory could exist, especially if some of the results were
    # completed in a previous run.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    training_set = read_csv(training_file)
    validation_set = read_csv(validation_file) if validation_file else None

    estimator = joblib.load(search_params['estimator'])
    grid = ParameterGrid(search_params['param_grid'])
    fit_params = search_params['fit_params'] \
                 if 'fit_params' in search_params.keys() else {}
    
    # Get the feature columns.
    feature_cols = [c for c in training_set.columns if c != target_col]

    X_train = training_set[feature_cols]
    y_train = training_set[[target_col]]

    # Initialize the validation stuff only if there's a validation set present.
    X_validation = validation_set[feature_cols] if validation_file else None
    y_validation = validation_set[[target_col]] if validation_file else None

    # This is an extremely sophisticated model ID scheme. Do note that things
    # will be overwritten if there's already stuff in the output directory, 
    # possibly. It will be bad if there's stuff from a different run (meaning
    # a run for a different estimator / parameter grid).
    for model_id, params in enumerate(grid):
       _train_and_evaluate(estimator,
                           params,
                           model_id,
                           training_file,
                           output_dir,
                           X_train,
                           y_train,
                           target_col,
                           search_params['scoring'],
                           fit_params,
                           X_validation=X_validation,
                           y_validation=y_validation,
                           validation_file=validation_file)
 
    # Unify all of the results files into one.
    os.system(
        "cat {output_dir}/results_*.json > ".format(output_dir=output_dir) +
        "{output_dir}/results.json".format(output_dir=output_dir))
    # Remove the intermediate results files.
    os.system("rm {output_dir}/results_*.json".format(output_dir=output_dir))