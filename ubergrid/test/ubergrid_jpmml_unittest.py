import os
import json
import subprocess

import numpy as np
import pandas as pd

from unittest import TestCase
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import ubergrid_jpmml as ugp
import ubergrid as ug

TEST_OUTPUT_DIR = "classification_test"
TEST_INPUT_DIR = "classification"
PMML_EVALUATOR = "example-1.3-SNAPSHOT.jar"

def setUpModule():
    os.mkdir(TEST_OUTPUT_DIR)
    os.mkdir(TEST_INPUT_DIR)

    classification_X, classification_y = make_classification()
    feature_cols = \
        ["feature_{}".format(ii) for ii in range(classification_X.shape[1])]
    
    classification_train = \
        DataFrame(data = np.c_[classification_X, classification_y],
                  columns = feature_cols + ['target'])

    classification_train.to_csv(TEST_INPUT_DIR + '/train.csv', index=False)
    classification_model = GradientBoostingClassifier()
    joblib.dump(classification_model, TEST_INPUT_DIR + '/classifier.pkl')

    search_params = {
        "fit_params": {
            "sample_weight": None
        },
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [2, 4]
        },
        "scoring": [
            "accuracy",
            "roc_auc",
            "log_loss"
        ],
        "estimator": TEST_INPUT_DIR + '/classifier.pkl'
    }

    param_file = open(TEST_INPUT_DIR + "/search_params.json", "w")
    json.dump(search_params, param_file)
    param_file.close()

    # Now execute a grid search on the provided inputs. This is the input to
    # the JPMML main function.
    search_params_file = TEST_INPUT_DIR + "/search_params.json"
    training_file = TEST_INPUT_DIR + "/train.csv"
    target_col = "target"
    output_dir = TEST_OUTPUT_DIR

    ug._main(search_params_file,
             target_col,
             training_file,
             output_dir)

def tearDownModule():
    subprocess.run(["rm", "-rf", TEST_INPUT_DIR, TEST_OUTPUT_DIR])

class UbergridJPMMLUnitTest(TestCase):

    def test_count_lines(self) -> None:
        """ Tests that the _count_lines function returns the correct value.
        """
        result = ugp._count_lines(TEST_OUTPUT_DIR + '/results.json')
        self.assertEqual(4, result)
    
    def test_make_pmml(self) -> None:
        """ Tests that the _make_pmml function returns the correct value and
            executes the correct side effects.
        """
        model_results_file = TEST_OUTPUT_DIR + '/results.json'
        fin = open(model_results_file, 'r')
        model_results_all = [json.loads(l) for l in fin]
        fin.close()

        model_results = model_results_all[0]
        pmml_file = ugp._make_pmml(model_results)

        # Test that the correct name is returned.
        model_file = model_results['model_file']
        pmml_file_truth = os.path.splitext(model_file)[0] + '.pmml'

        self.assertEqual(pmml_file_truth, pmml_file)

        # Test that the file exists.
        self.assertTrue(os.path.exists(pmml_file))

        # Test that the file returns the correct predictions.
        subprocess.run([
            "java",
            "-cp",
            PMML_EVALUATOR,
            "org.jpmml.evaluator.EvaluationExample",
            "--model",
            pmml_file,
            "--input",
            TEST_INPUT_DIR + "/train.csv",
            "--output",
            "output.csv"])

        pmml_predictions = pd.read_csv("output.csv")
        estimator = joblib.load(model_results['model_file'])
        estimator_input = \
            pd.read_csv(model_results['training_file'])\
              .drop('target', 1) # Drop the target column.
        estimator_predictions = estimator.predict_proba(estimator_input)

        # Technically we only need to test one of these but it feels weird.
        self.assertTrue(
            ((estimator_predictions[:,0] - pmml_predictions.probability_0)
                < 1e-14).all())
        self.assertTrue(
            ((estimator_predictions[:,1] - pmml_predictions.probability_1) 
                < 1e-14).all())

        subprocess.run(["rm", "-rf", "output.csv"])

    def test_time_pmml(self) -> None:
        """ Tests that the _time_pmml function returns the correct values.
        """
        model_results_file = TEST_OUTPUT_DIR + '/results.json'
        fin = open(model_results_file, 'r')
        model_results_all = [json.loads(l) for l in fin]
        fin.close()

        model_results = model_results_all[0]
        pmml_file = ugp._make_pmml(model_results)

        # Test that the correct result fields are returned.
        pmml_timing_results = \
            ugp._time_pmml(pmml_file, 
                           PMML_EVALUATOR, 
                           TEST_INPUT_DIR + '/train.csv')

        pmml_timing_truth_keys = set(
            [
                "pmml_total_prediction_time",
                "pmml_total_prediction_records"
            ])
        
        self.assertEqual(
            pmml_timing_truth_keys, set(pmml_timing_results.keys()))
        self.assertEqual(
            100, pmml_timing_results['pmml_total_prediction_records'])

    def test_main(self) -> None:

        ugp._main(TEST_OUTPUT_DIR,
                  PMML_EVALUATOR,
                  TEST_INPUT_DIR + "/train.csv")

        result_keys_truth = set(
            [
                "pmml_total_prediction_time",
                "pmml_total_prediction_records",
                "pmml_file",

                "training_file",
                "target",
                "model_file",
                "model_id",
                "max_depth",
                "n_estimators",

                "training_accuracy",
                "training_log_loss",
                "training_roc_auc",
                "training_total_prediction_time",
                "training_total_prediction_records",
                "training_time_total"
            ])
        
        results = open(TEST_OUTPUT_DIR + '/results.json', 'r')
        for result_line in results:
            result = json.loads(result_line)

            # Test that each result set has the correct fields.
            self.assertEqual(result_keys_truth, set(result.keys()))
            
            # Test that the PMML files exist.
            self.assertTrue(os.path.exists(result['pmml_file']))
            
            # Test that the PMML files make the correct predictions.
            subprocess.run([
                "java",
                "-cp",
                PMML_EVALUATOR,
                "org.jpmml.evaluator.EvaluationExample",
                "--model",
                result['pmml_file'],
                "--input",
                TEST_INPUT_DIR + "/train.csv",
                "--output",
                "output.csv"])

            pmml_predictions = pd.read_csv("output.csv")
            estimator = joblib.load(result['model_file'])
            estimator_input = \
                pd.read_csv(result['training_file'])\
                  .drop('target', 1) # Drop the target column.
            estimator_predictions = estimator.predict_proba(estimator_input)

            # Technically we only need to test one of these but it feels weird.
            self.assertTrue(
                ((estimator_predictions[:,0] - pmml_predictions.probability_0)
                    < 1e-14).all())
            self.assertTrue(
                ((estimator_predictions[:,1] - pmml_predictions.probability_1) 
                    < 1e-14).all())

            subprocess.run(["rm", "-rf", "output.csv"])
        
        results.close()
        
        # Test the exceptions.
        with self.assertRaises(ValueError):
            ugp._main("/not/a/dir")
        with self.assertRaises(ValueError):
            ugp._main(TEST_OUTPUT_DIR,
                      file_to_evaluate = TEST_INPUT_DIR + "/train.csv")
        with self.assertRaises(ValueError):
            ugp._main(TEST_OUTPUT_DIR,
                      pmml_evaluator = "not/a/jar",
                      file_to_evaluate = TEST_INPUT_DIR + "/train.csv")
        with self.assertRaises(ValueError):
            ugp._main(TEST_OUTPUT_DIR,
                      pmml_evaluator = PMML_EVALUATOR,
                      file_to_evaluate = "not/a/file")