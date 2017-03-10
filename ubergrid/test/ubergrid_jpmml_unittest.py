import os
import json
import subprocess

import numpy as np

from unittest import TestCase
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import ubergrid_jpmml as ugp
import ubergrid as ug

TEST_OUTPUT_DIR = "classification_test"
TEST_INPUT_DIR = "classification"

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
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 4, 6]
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
        self.assertEqual(9, result)
    
    def test_make_pmml(self) -> None:
        pass # TODO: Implement.
    def test_time_pmml(self) -> None:
        pass # TODO: Implement.
    def test_main(self) -> None:
        pass # TODO: Implement.