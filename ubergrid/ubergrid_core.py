import json
import os
import logging
import subprocess

import numpy as np

from time import time

from glob import glob

from pandas import DataFrame, Series, read_csv

from typing import List, Tuple, Dict, Any

from toolz import merge_with, identity, keymap, valmap

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import SCORERS
from sklearn.base import BaseEstimator

AVAILABLE_METRICS = {
    "accuracy",
    "f1",
    "recall",
    "precision",
    "log_loss",
    "roc_auc",
    "average_precision",
    "f1_micro",
    "f1_macro",
    "precision_micro",
    "precision_macro",
    "recall_micro",
    "recall_macro",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_median_absolute_error",
    "r2"
}

logging.basicConfig(format="%(asctime)s %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def _evaluate_model(estimator: BaseEstimator, 
                    X: DataFrame,
                    y: DataFrame,
                    grid_search_context: Dict[str, Any],
                    prefix: str) -> Dict[str, Any]:
    metrics = grid_search_context['metrics']
    # Validate that the metrics are in the available list.
    if len(set(metrics) - AVAILABLE_METRICS) != 0:
        logger.critical("{} are not available metrics.".format(
            " ".join(set(metrics) - AVAILABLE_METRICS)))
        raise ValueError(
            "{} are not available metrics.".format(
            set(metrics) - AVAILABLE_METRICS))
    
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
                 grid_search_context: Dict[str, Any]) \
                 -> Tuple[Dict[str, Any], BaseEstimator]:
    X = grid_search_context['X_train']
    y = grid_search_context['y_train']
    fit_params = grid_search_context['fit_params']

    fit_start = time()
    estimator.fit(X, y, **fit_params)
    fit_end = time()

    results = _evaluate_model(estimator, X, y, grid_search_context, "training")
    results["training_time_total"] = fit_end - fit_start

    return estimator, results

def _cross_validate(estimator: BaseEstimator,
                    model_id: int,
                    grid_search_context: Dict[str, Any]) -> Dict[str, Any]:
    n_splits = grid_search_context['cross_validation']
    X_train = grid_search_context['X_train']
    y_train = grid_search_context['y_train']
    fit_params = grid_search_context['fit_params']

    cross_validation_results = []
    k_folds = KFold(n_splits=n_splits)
    for cv_train, cv_test in k_folds.split(X_train):
            
        logger.info("Training model {} on cross validation training set."\
            .format(model_id))
        cv_train_start = time()
        estimator.fit(X_train.iloc[cv_train],
                      y_train.iloc[cv_train],
                      **fit_params)
        cv_train_stop = time()
        logger.info(
            "Completed training model {} on cross validation "\
            .format(model_id) + 
            "training set. Took {:.3f} seconds."\
                .format(cv_train_stop - cv_train_start))
        
        logger.info("Evaluating model {} on cross validation training set."\
            .format(model_id))
        cv_training_results = \
            _evaluate_model(
                estimator,
                X_train.iloc[cv_train],
                y_train.iloc[cv_train],
                grid_search_context,
                "cross_validation_training")
        logger.info("Completed evaluating model {} on cross validation "\
            .format(model_id) +
            "training set. Took {:.3f} seconds for {} records.".format(
                cv_training_results[
                    "cross_validation_training_total_prediction_time"],
                cv_training_results[
                    "cross_validation_training_total_prediction_records"]))
            
        logger.info("Evaluating model {} on cross validation test set."\
            .format(model_id))
        cv_validation_results = \
            _evaluate_model(estimator, 
                            X_train.iloc[cv_test],
                            y_train.iloc[cv_test],
                            grid_search_context, 
                            "cross_validation")
        logger.info("Completed evaluating model {} on cross validation "\
            .format(model_id) +
            "test set. Took {:.3f} seconds for {} records.".format(
                cv_validation_results[
                    "cross_validation_total_prediction_time"],
                cv_validation_results[
                    "cross_validation_total_prediction_records"]))
            
        cross_validation_results.append(
            {   
                "cross_validation_training_time_total": 
                    cv_train_stop - cv_train_start,
                **cv_training_results,
                **cv_validation_results
            })

    # Merge the results.
    cv_results_merged = merge_with(identity, *cross_validation_results)
    cv_results = {
        # These are the results for the individual folds.
        **(keymap(lambda x: x + "_all", cv_results_merged)),
        # These are the average results.
        **(valmap(lambda x: sum(x) / len(x), cv_results_merged))
    }
    logger.info("Cross validation for model {} completed.".format(model_id))
    return cv_results

def _train_and_evaluate(estimator: BaseEstimator,
                        params: Dict[str, Any],
                        model_id: int,
                        grid_search_context: Dict[str, Any]) -> None:
    # Unpack the grid search context.
    output_dir = grid_search_context['output_dir']
    cross_validation = grid_search_context['cross_validation']
    validation_file = grid_search_context['validation_file']
    target_col = grid_search_context['target_col']
    training_file = grid_search_context['training_file']
    
    param_str = ", ".join(
           ["{}={}".format(param_name, param_value)
            for param_name, param_value in params.items()])
    logger.info("Training and evaluating model {}: {}"\
                .format(model_id, param_str))
    
    model_file = "{}/model_{}.pkl".format(output_dir, model_id)
    results_file = "{}/results_{}.json".format(output_dir, model_id)
        
    # If the results file already exists, skip this pass.
    if os.path.exists(results_file):
        logger.info("Model {} already exists, skipping.".format(model_id))
        return

    # Initialize the estimator with the params.
    estimator.set_params(**params)

    cv_results = {}
    # Perform cross validation if selected.
    if cross_validation is not None:
        logger.info("Cross validating model {} for {} folds.".format(
            model_id, cross_validation))

        cv_results = \
            _cross_validate(estimator,
                            model_id,
                            grid_search_context)
       
    logger.info(
        "Training model {} and evaluating the model on the training set."\
        .format(model_id))
    estimator, training_results = \
        _train_model(estimator, grid_search_context)
    
    logger.info(
        "Model {} trained in {:.3f} seconds.".format(
            model_id, training_results["training_time_total"]))
    logger.info(
        "Model {} training set prediction time: {:.3f} for {} records.".format(
            model_id, 
            training_results["training_total_prediction_time"],
            training_results["training_total_prediction_records"]))

    # If the validation set is defined, use _evaluate_model to evaluate the 
    # model. Otherwise this is an empty dict.
    if validation_file is not None:
        logger.info(
            "Evaluating model {} on the validation set.".format(model_id))
    validation_results = \
            _evaluate_model(estimator,
                        grid_search_context['X_validation'],
                        grid_search_context['y_validation'], 
                        grid_search_context, 
                        "validation") \
        if validation_file is not None else {}

    if len(validation_results) > 0:
        logger.info(
            "Model {} validation set evaluation time: {:.3f} for {} records."\
            .format(model_id, 
                    validation_results["validation_total_prediction_time"],
                    validation_results["validation_total_prediction_records"]))
    
    # Construct and write the results for this run.
    results = {
        "training_file": training_file,
        "target": target_col,
        "model_file": model_file,
        "model_id": model_id,
        **cv_results,
        **training_results,
        **validation_results,
        **params
    }

    # Add the validation set file if present.
    if validation_file:
        results["validation_file"] = validation_file

    # Write the results _after_ the model.
    logger.info("Writing estimator for model {} to {}."\
                .format(model_id, model_file))
    joblib.dump(estimator, model_file)
    
    logger.info("Writing results for model {} to {}."\
                .format(model_id, results_file))
    with open(results_file, 'w') as results_out:
        results_out.write(
            json.dumps(results) + "\n")

def _dry_run(grid: ParameterGrid,
             grid_search_context: Dict[str, Any]):
    # Unpack the grid search context.
    output_dir = grid_search_context['output_dir']
    fit_params = grid_search_context['fit_params']
    metrics = grid_search_context['metrics']
    cross_validation = grid_search_context['cross_validation']
    validation_file = grid_search_context['validation_file']

    logger.info("Dry run: output_dir = {}".format(output_dir))
    logger.info("Dry run: Models trained with fit params {}.".format(
        ", ".join(["{}={}".format(fit_param_name, fit_param_value)
         for fit_param_name, fit_param_value in fit_params.items()])))
    logger.info("Dry run: Models evaluated with metrics {}.".format(
        ", ".join(metrics)))
    if cross_validation:
        logger.info("Dry run: Models cross validated with {} folds."\
            .format(cross_validation))
    if validation_file:
        logger.info("Dry run: Models validated on {}.".format(
            validation_file))
    for model_id, params in enumerate(grid):
        param_str = ", ".join(
           ["{}={}".format(param_name, param_value)
            for param_name, param_value in params.items()])
        logger.info("Dry run: Model {} trained and evaluated with {}.".format(
            model_id, param_str))

def _main(search_params_file: str,
          target_col: str,
          training_file: str,
          output_dir: str,
          validation_file: str = None,
          cross_validation: int = None,
          n_jobs: int = 1,
          dry_run: bool = False) -> None:
    # Validate that the search parameter file exists.
    if not os.path.exists(search_params_file):
        logger.critical("{} does not exist.".format(search_params_file))
        raise ValueError(
            "Search params file {} does not exist.".format(search_params_file))

    # Validate that the training file exists.
    if not os.path.exists(training_file):
        logger.critical(
            "Training file {} does not exist.".format(training_file))
        raise ValueError(
            "Training file {} does not exist.".format(training_file))
    
    # Validate that the validation file exists.
    if validation_file and not os.path.exists(validation_file):
        logger.critical(
            "Validation file {} does not exist.".format(validation_file))
        raise ValueError(
            "Validation file {} does not exist.".format(validation_file))
    
    search_params = json.load(open(search_params_file, 'r'))

    # The output directory could exist, especially if some of the results were
    # completed in a previous run.
    if not os.path.exists(output_dir):
        logger.info(
            "{} does not exist. Creating {}.".format(output_dir, output_dir))
        os.mkdir(output_dir)

    training_set = read_csv(training_file)
    validation_set = read_csv(validation_file) if validation_file else None

    # Validate that the training data contains the target column.
    if target_col not in training_set.columns:
        logger.critical(
            "Target column {} is not in the training data.".format(target_col))
        raise ValueError(
            "Target column {} not in training data.".format(target_col))
    
    # Validate that the validation data contains the target column.
    if validation_file and target_col not in validation_set.columns:
        logger.critical(
            "Target column {} is not in the validation data."\
            .format(target_col))
        raise ValueError(
            "Target column {} not in validation data.".format(target_col))

    if validation_file and \
        set(training_set.columns) != set(validation_set.columns):
        logger.critical(
            "Validation set doesn't have the same columns as the training set.")
        raise ValueError("Validation set doesn't have the same columns as "
            "the training set.")

    if "estimator" not in search_params.keys():
        logger.critical(
            "The search params file {} needs an \"estimator\" field."\
            .format(search_params_file))
        raise ValueError(
            "The search params file {} needs an \"estimator\" field."\
            .format(search_params_file))
    
    if "param_grid" not in search_params.keys():
        logger.critical(
            "The search params file {} needs a \"param_grid\" field."\
            .format(search_params_file))
        raise ValueError(
            "The search params file {} needs a \"param_grid\" field."\
            .format(search_params_file))
    
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

    # The grid search context contains information that is held consistent
    # with each run.
    grid_search_context = {
        "fit_params": fit_params,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "output_dir": output_dir,
        "metrics": search_params['scoring'],
        "cross_validation": cross_validation,
        "training_file": training_file,
        "validation_file": validation_file,
        "target_col": target_col
    }

    # Step through the dry run _after_ validating all of the inputs.
    if dry_run:
        _dry_run(grid, grid_search_context)
        # Exit the program.
        return
    
    # This is an extremely sophisticated model ID scheme. Do note that things
    # will be overwritten if there's already stuff in the output directory, 
    # possibly. It will be bad if there's stuff from a different run (meaning
    # a run for a different estimator / parameter grid).
    Parallel(n_jobs=n_jobs)(delayed(_train_and_evaluate)\
        # All the args to _train_and_evaluate.
        (joblib.load(search_params['estimator']),
         params,
         model_id,
         grid_search_context)
        for model_id, params in enumerate(grid))

    # Unify all of the results files into one.
    logger.info("Consolidating results.")
    results_glob = glob("{}/results_*.json".format(output_dir))
    
    with open('{}/results.json'.format(output_dir), 'w') as outfile:
        subprocess.run(['cat'] + results_glob, stdout=outfile)

    logger.info("Deleting intermediate results.")
    subprocess.run(["rm"] + results_glob)