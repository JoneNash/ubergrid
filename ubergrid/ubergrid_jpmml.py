import subprocess
import logging
import os
import json
import sys

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
    logger.critical("Install with pip install -e git+https://github.com/jpmml"
                    "/sklearn2pmml#egg=sklearn2pmml")
    sys.exit(1)

from sklearn_pandas import DataFrameMapper

def _count_lines(filename: str) -> int:
    input_file = subprocess.Popen(["cat", filename], stdout=subprocess.PIPE)
    word_counter = subprocess.Popen(["wc", "-l"], 
        stdin=input_file.stdout,
        stdout=subprocess.PIPE)
    num_lines = int(word_counter.communicate()[0].strip())
    
    return num_lines

def _make_pmml(model_results: Dict[str,Any]) -> str:
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
            logger.info(
                "Done timing model {}. Took {:.3f} seconds for {} records."\
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