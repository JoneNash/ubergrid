from unittest import TestCase

from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np
import json
import os
import subprocess

import ubergrid as ug
import ubergrid_core as ugc

TEST_OUTPUT_DIR = "classification_test"
CLASSIFICATION_DIR = "classification"

def setUpModule():
    """ Creates a test fixture by building parameter grid for a gradient
        boosting classifier on generated binary classification data, then
        running ``ubergrid_core._main( ... )`` on it.
    """
    os.mkdir(TEST_OUTPUT_DIR)
    os.mkdir(CLASSIFICATION_DIR)

    classification_X, classification_y = make_classification()
    feature_cols = \
        ["feature_{}".format(ii)
         for ii in range(classification_X.shape[1])]
    
    classification_train = \
        DataFrame(data = np.c_[classification_X, classification_y],
                  columns = feature_cols + ['target'])

    classification_train.to_csv(CLASSIFICATION_DIR + "/train.csv", index=False)

    classification_model = GradientBoostingClassifier()
    joblib.dump(classification_model, CLASSIFICATION_DIR + "/classifier.pkl")

    classification_search_params = {
        "param_grid": {
            "n_estimators": [100, 200]
        },
        "scoring": ["accuracy"],
        "estimator": CLASSIFICATION_DIR + "/classifier.pkl"
    }

    param_file = open(CLASSIFICATION_DIR + '/search_params.json', 'w')
    json.dump(classification_search_params, param_file)
    param_file.close()

    # Run the grid search.
    # Do it with cross validation turned on so we can see the effect of the
    # excluded columns.
    ugc._main(
        CLASSIFICATION_DIR + "/search_params.json",
        "target",
        CLASSIFICATION_DIR + "/train.csv",
        TEST_OUTPUT_DIR,
        cross_validation = 3
    )

def tearDownModule():
    """ Deletes the test fixtures created in setUpModule().
    """
    subprocess.run(['rm', '-rf', CLASSIFICATION_DIR])
    subprocess.run(['rm', '-rf', TEST_OUTPUT_DIR])

class UbergridUnitTest(TestCase):

    def test_dict_contains(self):
        small_dict = {"a": 1, "b": 2}
        big_dict = {"a": 1, "b": 2, "c": 3}
        other_big_dict = {"a": 2, "b": 2, "c": 4}

        self.assertTrue(ug._dict_contains(small_dict, big_dict))
        self.assertFalse(ug._dict_contains(small_dict, other_big_dict))

    def test_frame_exclude_col(self):

        self.assertTrue(ug._frame_exclude_col("cross_validation_log_loss_all"))
        self.assertFalse(
            ug._frame_exclude_col("cross_validation_log_loss_mean"))

    def test_read_results(self):
        # Test that an exception is thrown when the results dir doesn't exist.
        with self.assertRaises(ValueError):
            ug.read_results("/not/a/dir/")
        
        results = ug.read_results(TEST_OUTPUT_DIR)

        true_results_keys = {
            "training_file",
            "model_file",
            "model_id",
            "target",
            "n_estimators",
            "training_time_total",
            "training_total_prediction_time",
            "training_total_prediction_records",
            "training_accuracy",
            
            "cross_validation_accuracy",
            "cross_validation_total_prediction_time",
            "cross_validation_total_prediction_records",

            "cross_validation_accuracy_all",
            "cross_validation_total_prediction_time_all",
            "cross_validation_total_prediction_records_all",

            "cross_validation_training_accuracy",
            "cross_validation_training_total_prediction_time",
            "cross_validation_training_total_prediction_records",
            "cross_validation_training_time_total",
            
            "cross_validation_training_accuracy_all",
            "cross_validation_training_total_prediction_time_all",
            "cross_validation_training_total_prediction_records_all",
            "cross_validation_training_time_total_all"
        }

        self.assertEqual(2, len(results))
        for result in results:
            self.assertEqual(true_results_keys, set(result.keys()))

    def test_get_results_frame(self):
        results_frame = ug.read_results_frame(TEST_OUTPUT_DIR)
        
        true_columns = {
            "training_file",
            "model_file",
            "model_id",
            "target",
            "n_estimators",
            "training_time_total",
            "training_total_prediction_time",
            "training_total_prediction_records",
            "training_accuracy",
            
            "cross_validation_accuracy",
            "cross_validation_total_prediction_time",
            "cross_validation_total_prediction_records",

            "cross_validation_training_accuracy",
            "cross_validation_training_total_prediction_time",
            "cross_validation_training_total_prediction_records",
            "cross_validation_training_time_total"
        }

        self.assertEqual(true_columns, set(results_frame.columns))
        self.assertEqual(2, len(results_frame))

    def test_get_model(self):

        results = ug.read_results(TEST_OUTPUT_DIR)

        # Test that the function throws an exception when no models match.
        with self.assertRaises(ValueError):
            ug.get_model(results, n_estimators = 500)
        
        # Test that the function throws an exception when too many models 
        # match.
        with self.assertRaises(ValueError):
            ug.get_model(results)

        # Test that the model is retrieved properly when the results struct
        # is used.
        model = ug.get_model(results, n_estimators = 200)

        self.assertEqual(200, model.get_params()['n_estimators'])

        # Test that the model is retrieved properly when the results directory
        # is used.
        model2 = ug.get_model(TEST_OUTPUT_DIR, n_estimators = 100)

        self.assertEqual(100, model2.get_params()['n_estimators'])