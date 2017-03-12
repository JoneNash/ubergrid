import subprocess
import logging
import os
import json

from time import time

from typing import Dict, Any

from sklearn.externals import joblib
from sklearn.externals.joblib import delayed

logging.basicConfig(format="%(asctime)s %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn2pmml import PMMLPipeline, sklearn2pmml
except Exception as e:
    logger.critical("sklearn2pmml is not installed. "
                    "This is required for PMML support.")
    raise e

from sklearn_pandas import DataFrameMapper

def _count_lines(filename: str) -> int:
    """ Counts the lines in the file using the shell.

        :param filename: The name of the file to count lines for.

        :returns: The number of lines in the file.
    """
    input_file = subprocess.Popen(["cat", filename], stdout=subprocess.PIPE)
    word_counter = subprocess.Popen(["wc", "-l"], 
        stdin=input_file.stdout,
        stdout=subprocess.PIPE)
    num_lines = int(word_counter.communicate()[0].strip())
    
    return num_lines

def _make_pmml(model_results: Dict[str,Any]) -> str:
    """ Creates the PMML file for the estimator.

        :param model_results: The results dict for the model.

        :returns: The name of the PMML file for the model.
    """
    training_file = model_results['training_file']
    target = model_results['target']
    model_file = model_results['model_file']

    estimator = joblib.load(model_file)

    training_in = open(training_file, 'r')
    header = next(training_in)
    training_in.close()

    # Grab the feature columns for the DataFrameMapper.
    feature_cols = \
        [c.strip() for c in header.strip().split(",") if c.strip() != target]

    df_mapper = DataFrameMapper([(feature_cols, None)])
    estimator_pipeline = PMMLPipeline([
        ("mapper", df_mapper),
        ("estimator", estimator)])

    pmml_file = os.path.splitext(model_file)[0] + '.pmml'
    sklearn2pmml(estimator_pipeline, pmml_file)
    return pmml_file

def _time_pmml(pmml_file: str, pmml_evaluator: str, file_to_evaluate: str) -> \
    Dict[str, Any]:
    """ Evaluates the PMML file and times it.

        :param pmml_file: The name of the pmml file with the model.

        :param pmml_evaluator: The name of the PMML evaluator executable jar.

        :param file_to_evaluate: The name of the file to evaluate timing for.

        :returns:
            A dict with the number of records in the file and the total time
            for prediction.
    """
    start = time()
    subprocess.run([
        "java",
        "-cp",
        pmml_evaluator,
        "org.jpmml.evaluator.EvaluationExample",
        "--model",
        pmml_file,
        "--input",
        file_to_evaluate,
        "--output",
        "output.csv"])
    stop = time()

    subprocess.run(["rm", "-rf", "output.csv"])
    return {
        "pmml_total_prediction_time": stop - start,
        "pmml_total_prediction_records": _count_lines(file_to_evaluate) - 1
    }

def _main(results_dir: str,
          pmml_evaluator: str = None,
          file_to_evaluate: str = None) -> None:

    """ Creates and optionally times a PMML file for each estimator in the
        provided results grid.

        :param results_dir:
            The directory where a complete grid search results.json file lives.
        
        :param pmml_evaluator: 
            The name of the PMML evaluator executable jar. Default: None.

        :param file_to_evaluate:
            The name of the csv containing the values to time the PMML file 
            with. The CSV file must have a header.

        :raises ValueError: 
            If the ``results_dir`` doesn't contain a ``results.json`` file.
        
        :raises ValueError:
            If the ``file_to_evaluate`` option is provided without a 
            corresponding ``pmml_evaluator`` argument.
        
        :raises ValueError: If the ``pmml_evaluator`` argument doesn't exist.

        :raises ValueError: If the ``file_to_evaluate`` argument doesn't exist.

        :returns:
            Nothing. Writes a PMML file for every model in the grid, and 
            modifies the ``results.json`` file to include the following fields::

                {
                    # Everything in results.json PLUS:

                    "pmml_file": "/path/to/model.pmml",

                    # If PMML timing was selected.
                    "pmml_total_prediction_time": time_for_pmml_prediction,
                    "pmml_total_prediction_records": number_of_pmml_predictions
                }
    """
    results_file = results_dir + "/results.json"
    # Validate the inputs.
    if not os.path.exists(results_file):
        logger.critical("Results file {} does not exist.".format(results_file))
        raise ValueError(
            "Results file {} does not exist.".format(results_file))

    if file_to_evaluate and not pmml_evaluator:
        logger.critical("PMML evaluation requires a PMML evaluator jar.")
        raise ValueError("PMML evaluation requires a PMML evaluator jar.")

    if pmml_evaluator and not os.path.exists(pmml_evaluator):
        logger.critical(
            "PMML evaluator {} does not exist.".format(pmml_evaluator))
        raise ValueError(
            "PMML evaluator {} does not exist.".format(pmml_evaluator))

    if file_to_evaluate and not os.path.exists(file_to_evaluate):
        logger.critical(
            "File {} does not exist.".format(file_to_evaluate))
        raise ValueError(
            "File {} does not exist.".format(file_to_evaluate))
    
    logger.info("Reading results file {}.".format(results_file))
    results_in = open(results_file, 'r')
    results = [json.loads(r) for r in results_in]
    results_in.close()

    new_results = []
    for result in results:

        logger.info("Creating PMML file for model {}.".format(
            result["model_id"]))
        pmml_file = _make_pmml(result)
        # Add the PMML file to the result.
        result['pmml_file'] = pmml_file

        if pmml_evaluator:
            logger.info("Timing PMML file for model {}.".format(
                result["model_id"]))
            pmml_results = \
                _time_pmml(pmml_file, pmml_evaluator, file_to_evaluate)
            logger.info("Done timing model {}. Took {} seconds for {} records."\
                .format(result['model_id'], 
                        pmml_results['pmml_total_prediction_time'],
                        pmml_results['pmml_total_prediction_records']))
            # Integrate the results.
            result = { **result, **pmml_results }
            new_results.append(result)
    
    # Now write all of the results back to disk, overwriting the previous file.
    with open(results_file, 'w') as results_out:
        for result in new_results:
            results_out.write(json.dumps(result) + "\n")