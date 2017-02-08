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
# TODO: Validate metric names.
# TODO: Add cross validation.
# TODO: Add joblib Parallel to support multiple simultaneous runs.
# TODO: Refactor the arguments to _train_and_evaluate.

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
                    "training_prediction_time": time_for_predictions,
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
                        validation_file: str = None) -> None:
    """ Performs training and evaluation on a scikit-learn estimator, saving the
        results to disk.
    
        If there are already results in the provided output path,
        this function skips calculations (for larger grids this allows jobs to be
        resumed if they fail). This function is designed in this way so that it can
        be executed in parallel. It writes two files: a joblib-pickled model (in
        ``{output_dir}/model_{model_id}.pkl``), and
        a results file that contains a single line JSON object 
        (``output_dir/results_{}.json``). That object contains
        all of the performance and timing information related to the run. Here's an
        example of that file::

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

                # The metrics for validation, if validation was performed.
                "validation_prediction_time": time_for_predictions_validation,
                "validation_total_prediction_records": number_of_validation_records,
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

        :returns: Nothing, writes to the files described above.
        
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

        :raises ValueError: 
            If the validation set is present and doesn't have the same columns 
            as the training set.

        :returns: 
            Nothing. Writes all of the models in the grid as pickled files in
            ``output_dir`` along with a ``results.json``.
    """

    search_params = json.load(open(search_params_file, 'r'))

    # The output directory could exist, especially if some of the results were
    # completed in a previous run.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    training_set = read_csv(training_file)
    validation_set = read_csv(validation_file) if validation_file else None

    if validation_file and \
        set(training_set.columns) != set(validation_set.columns):
        raise ValueError("Validation set doesn't have the same columns as "
            "the training set.")

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