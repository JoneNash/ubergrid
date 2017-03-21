import json
import os

from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from pandas import DataFrame
from typing import Dict, Any, List, Union
from toolz import keyfilter, valfilter, compose, complement, curry

# Helper functions.
listfilter = compose(list, filter)
listmap = compose(list, map)
listels = lambda x: type(x) is list

def _dict_contains(small_dict: Dict[str, Any], 
                   big_dict: Dict[str, Any]) -> bool:
    """ Returns true if ``small_dict`` is contained by ``big_dict``.

        :param small_dict: 
            The smaller dict that may or may not be contained by the bigger 
            dict.
        
        :param big_dict:
            The bigger dict that may or may not contain the smaller dict.

        :returns: True if the bigger dict contains the smaller dict.
    """
    # Remove the list values from big_dict because they are unhashable.
    # The implicit assumption is that small_dict doesn't have list elements.
    # Since this is called in the context of model hyperparameters I think
    # this is a safe assumption.
    return set(small_dict.items()) <= \
           set(valfilter(complement(listels), big_dict).items())

def _frame_exclude_col(col_name: str) -> bool:
    """ Returns true for columns to exclude in a data frame representation of 
        the grid search results. Currently this is only the cross validation
        list fields in ``results.json``.

        :param col_name: The name of the column.

        :returns: Whether to exclude or include it.
    """
    return col_name.startswith("cross_validation") and \
           col_name.endswith("all")

def read_results(output_dir: str) -> List[Dict[str, Any]]:
    """ Reads the ``results.json`` file from the provided output directory into
        a list of dicts.

        :param output_dir: The name of the output directory of the grid search.

        :returns: The results as a list of dicts.

        :raises ValueError: 
            If the output directory doesn't have a ``results.json`` file in it.
    """
    results_file = output_dir + "/results.json"
    
    if not os.path.exists(results_file):
        raise ValueError("Results file {} does not exist.".format(results_file))
    
    results_fin = open(results_file, 'r')
    results = [json.loads(l) for l in results_fin]
    results_fin.close()

    return results

def read_results_frame(output_dir: str) -> DataFrame:
    """ Reads the results.json file into a pandas DataFrame.

        :param output_dir: The name of the output directory of the grid search.

        :returns: 
            The results as a pandas data frame. Excludes the cross validation 
            run lists (but keeps the cross validation mean values).

        :raises ValueError:
            If the output directory doesn't have a ``results.json`` file in it.
    """
    results = read_results(output_dir)
    return DataFrame(
        data = listmap(
            lambda r: keyfilter(complement(_frame_exclude_col), r), results))

def get_model(results: Union[str, Dict[str, Any]], **kwargs) -> BaseEstimator:
    """ Obtains the model with the provided parameters from the results.
        
        :param results: 
            Either the results list of dictionaries or the name of the directory
            with the ``results.json`` file for the run.

        :param **kwargs: 
            The parameters uniquely identifying the model to obtain.

        :returns: The estimator with the corresponding parameters.

        :raises ValueError:
            If the parameters do not match a model in the grid.

        :raises ValueError: 
            If the parameters match more than one model in the grid.
    """
    # Pull the results if the directory is provided.
    if type(results) is str:
        results = read_results(results) 
    
    contains_args = curry(_dict_contains)(kwargs)
    matching_results = listfilter(contains_args, results)

    if len(matching_results) == 0:
        raise ValueError("No results for parameters: {}.".format(
            ",".join(["{}={}".format(param_name, param_value)
                      for param_name, param_value in kwargs.items()])))
    if len(matching_results) > 1:
        raise ValueError("More than one model for parameters: {}.".format(
            ",".join(["{}={}".format(param_name, param_value)
                      for param_name, param_value in kwargs.items()])))

    return joblib.load(matching_results[0]['model_file'])