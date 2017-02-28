import json
import sys
import os
import logging

import numpy as np

from time import time

from pandas import DataFrame, Series, read_csv

from typing import List, Tuple, Dict, Any

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import SCORERS
from sklearn.base import BaseEstimator
from toolz import merge_with, identity, keymap, valmap

# TODO: Refactor the arguments to _train_and_evaluate.
# TODO: Add PMML generation.
# TODO: Add PMML timing.

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
                    metrics: List[str], 
                    prefix: str) -> Dict[str, Any]:
    """ Evaluates the performance of the model on the provided data, for the
        provided metrics, and returns a dictionary of results.

        :param estimator: A scikit-learn estimator object (trained).
        
        :param X: The data to evaluate the model on, without the true value.
        
        :param y: The true values for the data in X.
        
        :param metrics: 
            A list of metric names to test. They must be metrics in
            scikit-learn's ``SCORERS`` dict 
            (see `sklearn.metrics <http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`_).
            Some of these metrics require additional data that isn't currently 
            available within the schema of ``search_params``. These are the 
            metrics which `are` available:

            * ``"accuracy"``
            * ``"f1"``
            * ``"recall"``
            * ``"precision"``
            * ``"log_loss"``
            * ``"roc_auc"``
            * ``"average_precision"``
            * ``"f1_micro"``
            * ``"f1_macro"``
            * ``"precision_micro"``
            * ``"precision_macro"``
            * ``"recall_micro"``
            * ``"recall_macro"``
            * ``"neg_mean_absolute_error"``
            * ``"neg_mean_squared_error"``
            * ``"neg_median_absolute_error"``
            * ``"r2"``
        
        :param prefix: A string to prefix the fields in the results dict with.

        :raises ValueError:
            If there's a metric that cannot be calculated from ``SCORERS``.
        
        :returns: 
            The results in a dictionary, with one field for each metric,
            named as ``{prefix}_{metric}``, plus a couple of fields with timing
            information::

                {
                    "{prefix}_{metric1}": value,
                    "{prefix}_{metric2}": value,
                    "{prefix}_total_prediction_time": time_in_seconds,
                    "{prefix}_total_prediction_records": number_of_records
                }
    """

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
                 X: DataFrame, 
                 y: DataFrame, 
                 metrics: List[str],
                 fit_params: Dict[str, Any]) \
                 -> Tuple[Dict[str, Any], BaseEstimator]:
    """ Trains the model on the provided data, and evaluates it for the provided
        metrics.

        :param estimator: A scikit-learn estimator object (untrained).

        :param X: The data on which to train the estimator, without the target.

        :param y: The ground truth target for X.

        :param metrics: 
            A list of strings describing the metrics to calculate
            against the training data. They must be in scikit-learn's SCORERS dict.
            See `sklearn.metrics <http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`_).

        :param fit_params: 
            A dictionary of parameters to send to the estimator's
            ``fit`` method.

        :returns: 
            A tuple containing the results in a dict, with fields named
            as ``training_{metric}``, and the trained estimator. The dict also 
            contains information related to the timing of the model. Here's an
            example::

                {
                    "training_{metric}": value,
                    "training_{metric}": value,
                    "training_total_prediction_time": time_for_predictions,
                    "training_total_prediction_records": number_of_records,
                    "training_time_total": time_for_training
                }
    """
    fit_start = time()
    estimator.fit(X, y, **fit_params)
    fit_end = time()

    results = _evaluate_model(estimator, X, y, metrics, "training")
    results["training_time_total"] = fit_end - fit_start

    return estimator, results

def _cross_validate(estimator: BaseEstimator,
                    model_id: int,
                    X_train: DataFrame,
                    y_train: DataFrame,
                    metrics: List[str],
                    fit_params: Dict[str, Any],
                    n_splits: int) -> Dict[str, Any]:
    """ Performs K-Fold cross validation on the estimator for a given training
        set.

        :param estimator: The estimator to be cross validated.

        :param model_id: The model identifier.

        :param X_train: The training data in a DataFrame.

        :param y_train: The training data ground truth values in a DataFrame.

        :param metrics: A list of scores to evaluate.

        :param fit_params: 
            A dict of parameters to pass to the estimators ``fit`` method.

        :param n_splits: The number of splits in the validation.

        :returns:
            A dict containing evaluation metrics for the cross validation. There
            are two sets of metrics: one set of lists that contain values for
            each validation split, and another that contains the means of the
            values for all of the splits. Metric fields are named 
            ``cross_validation_{metric}_all`` for the lists and 
            ``cross_validation_{metric}`` for the means. These are also repeated
            for the training cycles, prefixed with 
            ``cross_validation_training_...``. The dict also contains
            information about the timing of the training and evaluation of the
            model on the splits. Here's an example::

                {
                    # Training set values.
                    "cross_validation_training_{metric}_all": 
                        list_of_training_set_metric_values,
                    "cross_validation_training_{metric}":
                        mean_of_training_set_metric_list,
                    # ... for each metric.
                    
                    "cross_validation_training_total_prediction_time_all":
                        list_of_training_set_prediction_times,
                    "cross_validation_training_total_prediction_time":
                        mean_of_training_set_prediction_time_list,
                    "cross_validation_training_total_prediction_records_all":
                        list_of_training_record_counts,
                    "cross_validation_training_total_prediction_records":
                        mean_of_training_record_count_list,
                    "cross_validation_training_time_total_all":
                        list_of_training_times,
                    "cross_validation_training_time_total_mean":
                        mean_of_training_time_list,

                    # Test set values.
                    "cross_validation_{metric}_all": list_of_metric_values,
                    "cross_validation_{metric}": mean_of_metric_list,
                    # ... for each metric.
                    
                    "cross_validation_total_prediction_time_all":
                        list_of_prediction_times,
                    "cross_validation_total_prediction_time":
                        mean_of_prediction_time_list,
                    "cross_validation_total_prediction_records_all":
                        list_of_prediction_record_counts,
                    "cross_validation_total_prediction_records":
                        mean_of_prediction_record_count_list
                }
    """
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
            "training set. Took {} seconds."\
                .format(cv_train_stop - cv_train_start))
        
        logger.info("Evaluating model {} on cross validation training set."\
            .format(model_id))
        cv_training_results = \
            _evaluate_model(
                estimator,
                X_train.iloc[cv_train],
                y_train.iloc[cv_train],
                metrics,
                "cross_validation_training")
        logger.info("Completed evaluating model {} on cross validation "\
            .format(model_id) +
            "training set. Took {} seconds for {} records.".format(
                cv_training_results[
                    "cross_validation_training_total_prediction_time"],
                cv_training_results[
                    "cross_validation_training_total_prediction_records"]))
            
        logger.info("Evaluating model {} on cross validation test set."\
            .format(model_id))
        cv_validation_results = \
            _evaluate_model(
                estimator,
                X_train.iloc[cv_test],
                y_train.iloc[cv_test],
                metrics,
                "cross_validation")
        logger.info("Completed evaluating model {} on cross validation "\
            .format(model_id) +
            "test set. Took {} seconds for {} records.".format(
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

# TODO: This _train_and_evaluate function is kind of a mess. It's not a huge
# deal since it's internal to the system only, but it might be worth a refactor
# at some point.
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
                        validation_file: str = None,
                        cross_validation: int = None) -> None:
    """ Performs training and evaluation on a scikit-learn estimator, saving the
        results to disk.
    
        If there are already results in the provided output path,
        this function skips calculations (for larger grids this allows jobs to 
        be resumed if they fail). This function is designed in this way so that 
        it can be executed in parallel. It writes two files: a joblib-pickled 
        model (in``{output_dir}/model_{model_id}.pkl``), and
        a results file that contains a single line JSON object 
        (``output_dir/results_{}.json``). That object contains
        all of the performance and timing information related to the run. 
        Here's an example of that file::

            {
                # Files used to build and evaluate the model.
                "training_file": "/path/to/training.csv",
                "model_file": "/path/to/model.pkl",
                "validation_file": "/path_to_validation.csv", # If used.

                # The name of the target column.
                "target": "target_col_name",

                # Parameters that identify the model
                "param_1": param_value_1,
                "param_2": param_value_2,
                # ...

                # The metrics for training
                "training_time_total": time_for_training,
                "training_total_prediction_time": time_for_predictions,
                "training_total_prediction_records": number_of_records,
                "training_{metric}": value,
                "training_{metric}": value,
                # ...

                # The metrics for cross validation, if cross validation was
                # performed.
                "cross_validation_training_time_total_all": list_of_times,
                "cross_validation_training_time_total_mean": mean_training_time,
                "cross_validation_total_prediction_time_all":
                    list_of_prediction_times,
                "cross_validation_total_prediction_time_mean":
                    mean_total_prediction_time,
                "cross_validation_total_prediction_records_all":
                    list_of_numbers_of_records,
                "cross_validation_total_prediction_records_mean":
                    mean_number_of_records,
                "cross_validation_{metric}_all": list_of_metric_values,
                "cross_validation_{metric}_mean": mean_metric_value,
                "cross_validation_{metric}_all": list_of_metric_values,
                "cross_validation_{metric}_mean": mean_metric_value,
                # ...

                # The metrics for validation, if validation was performed.
                "validation_total_prediction_time": 
                    time_for_predictions_validation,
                "validation_total_prediction_records": 
                    number_of_validation_records,
                "validation_{metric}": value,
                "validation_{metric}": value,
                # ...

            }

        :param estimator: The scikit-learn estimator object.

        :param params: The parameters for building the model, as a dict.

        :param model_id: An integer id for the model.

        :param training_file: The name of the file with the training data.

        :param output_dir: The name of the output directory.

        :param X_train: The training data, without the the target column.

        :param y_train: The training data ground-truth values.

        :param target_col: The name of the target column.

        :param metrics: 
            A list of strings describing the metrics to evaluate. Each
            string must correspond to a value in scikit-learn's ``SCORERS`` 
            dict. See `sklearn.metrics <http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`_

        :param fit_params: Parameters to pass to the estimator's ``fit`` method.

        :param X_validation: 
            The validation data, without the target_column.
            Default: None.

        :param y_validation: 
            The ground truth target values for the validation data.
            Default: None.

        :param validation_file: 
            The name of the file with the validation data.
            Default: None.

        :param cross_validation:
            The number of cross validation folds to perform.
            Default: None.

        :returns: Nothing, writes to the files described above.
        
    """
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
                            X_train,
                            y_train,
                            metrics,
                            fit_params,
                            cross_validation)
       
    logger.info(
        "Training model {} and evaluating the model on the training set."\
        .format(model_id))
    estimator, training_results = \
        _train_model(estimator, 
                     X_train, 
                     y_train, 
                     metrics, 
                     fit_params)
    logger.info(
        "Model {} trained in {} seconds.".format(
            model_id, training_results["training_time_total"]))
    logger.info(
        "Model {} training set prediction time: {} for {} records.".format(
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
                        X_validation, 
                        y_validation, 
                        metrics,
                        "validation") \
        if validation_file is not None else {}

    if len(validation_results) > 0:
        logger.info(
            "Model {} validation set evaluation time: {} for {} records."\
            .format(model_id, 
                    validation_results["validation_total_prediction_time"],
                    validation_results["validation_total_prediction_records"]))
    
    # Construct and write the results for this run.
    results = {
        "training_file": training_file,
        "target": target_col,
        "model_file": model_file,
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
             output_dir: str,
             scoring: List[str],
             fit_params: Dict[str, Any],
             validation_file: str,
             cross_validation: int):
    """ Logs the actions that will execute in the grid search without actually
        executing them.

        :param grid: A scikit-learn ParameterGrid object.
        
        :param output_dir: The name of the output directory for the models.

        :param fit_params: The parameters to send to the model's ``fit`` method.

        :param validation_file: 
            The name of the file with the validation data (if present).
        
        :param cross_validation:
            The number of cross validation folds (if present).
    """
    logger.info("Dry run: output_dir = {}".format(output_dir))
    logger.info("Dry run: Models trained with fit params {}.".format(
        ", ".join(["{}={}".format(fit_param_name, fit_param_value)
         for fit_param_name, fit_param_value in fit_params.items()])))
    logger.info("Dry run: Models evaluated with metrics {}.".format(
        ", ".join(scoring)))
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
    """ Executes an entire parameter grid, saving all results and models to disk.

        This method runs the ``_train_and_evaluate`` function on each combination
        of parameters. It's the core function for the module. It loads the estimator
        from a path specified in the ``search_param_file`` using joblib's pickling
        capabilities. It builds the grid from that JSON file as well. This function
        creates (if necessary) and writes all models into a specified
        directory. It also places a file called ``results.json`` that contains, for
        each model, one JSON object that has all of the information used to build
        the model, and all of its performance characteristics.
        Here's an example of one line::

            {
                # Files used to build and evaluate the model.
                "training_file": "/path/to/training.csv",
                "model_file": "/path/to/model.pkl",
                "validation_file": "/path_to_validation.csv", # If used.

                # The name of the target column.
                "target": "target_col_name",

                # Parameters that identify the model
                "param_1": param_value_1,
                "param_2": param_value_2,
                # ...

                # The metrics for training
                "training_time_total": time_for_training,
                "training_prediction_time": time_for_predictions,
                "training_total_prediction_records": number_of_records,
                "training_{metric}": value,
                "training_{metric}": value,
                # ...

                # The metrics for cross validation, if cross validation was
                # performed.
                "cross_validation_training_time_total_all": list_of_times,
                "cross_validation_training_time_total_mean": mean_training_time,
                "cross_validation_total_prediction_time_all":
                    list_of_prediction_times,
                "cross_validation_total_prediction_time_mean":
                    mean_total_prediction_time,
                "cross_validation_total_prediction_records_all":
                    list_of_numbers_of_records,
                "cross_validation_total_prediction_records_mean":
                    mean_number_of_records,
                "cross_validation_{metric}_all": list_of_metric_values,
                "cross_validation_{metric}_mean": mean_metric_value,
                "cross_validation_{metric}_all": list_of_metric_values,
                "cross_validation_{metric}_mean": mean_metric_value,
                # ...

                # The metrics for validation, if validation was performed.
                "validation_prediction_time": time_for_predictions_validation,
                "validation_total_prediction_records": number_of_validation_records,
                "validation_{metric}": value,
                "validation_{metric}": value,
                # ...

            }

        :param search_params_file: 
            The name of the JSON file with the search 
            parameters. The file itself should have the following structure::
            
                {
                    # These are passed to the fit method of each estimator.
                    "fit_params": {
                        "fit_param_1": value,
                        "fit_param_2": value
                    },
                    "param_grid": {
                        "param_1": [value, value, value],
                        "param_2": [value, value, value]
                    },
                    "scoring": [metric, metric, metric],
                    # The estimator should be pickled with joblib.
                    "estimator": "/path/to/estimator.pkl"
                }
    
        :param target_col: 
            The name of the column containing the target variable.
    
        :param training_file: The name of the file containing the training data.

        :param output_dir: The name of the output directory.

        :param validation_file: 
            The name of the file containing the validation data.
            Default: None.

        :param cross_validation:
            The number of K-Fold cross validations to perform.
            Default: None.

        :param n_jobs: 
            The number of parallel jobs to execute the grid for.
            Default: 1.

        :param dry_run:
            Whether to execute a "dry run", which prints out the steps
            that will be executed.
            Default: False

        :raises ValueError: If the ``search_params_file`` doesn't exist.

        :raises ValueError: If the ``training_set_file`` doesn't exist.

        :raises ValueError: If the ``validation_set_file`` doesn't exist.

        :raises ValueError: If the ``target_col`` is not in the training set.

        :raises ValueError: If the ``target_col`` is not in the validation set.

        :raises ValueError: 
            If the validation set is present and doesn't have the same columns 
            as the training set.


        :returns: 
            Nothing. Writes all of the models in the grid as pickled files in
            ``output_dir`` along with a ``results.json``.
    """

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
        logger.info("{} does not exist. Creating {}.".format(output_dir))
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

    if dry_run:
        _dry_run(grid, 
                 output_dir, 
                 search_params['scoring'], 
                 fit_params, 
                 validation_file, 
                 cross_validation)
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
         ii,
         training_file,
         output_dir,
         X_train,
         y_train,
         target_col,
         search_params['scoring'],
         fit_params,
         X_validation=X_validation,
         y_validation=y_validation,
         validation_file=validation_file,
         cross_validation=cross_validation)
        for ii, params in enumerate(grid))

    # Unify all of the results files into one.
    logger.info("Consolidating results.")
    os.system(
        "cat {output_dir}/results_*.json > ".format(output_dir=output_dir) +
        "{output_dir}/results.json".format(output_dir=output_dir))
    # Remove the intermediate results files.
    logger.info("Deleting intermediate results.")
    os.system("rm {output_dir}/results_*.json".format(output_dir=output_dir))