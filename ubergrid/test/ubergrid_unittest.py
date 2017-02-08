import os
import subprocess
import json

from unittest import TestCase

import numpy as np
from pandas import DataFrame, read_csv
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, ParameterGrid

import ubergrid as ug

TEST_OUTPUT_DIR = "classification_test"
CLASSIFICATION_DIR = "classification"
MULTICLASS_DIR = "multiclass"
REGRESSION_DIR = "regression"

def setUpModule():
    os.mkdir(TEST_OUTPUT_DIR)

    # Make a directory to store the classification data, model, and
    # results.
    os.mkdir(CLASSIFICATION_DIR)
    classification_X, classification_y = make_classification()
    feature_cols = \
        ["feature_{}".format(ii) 
         for ii in range(classification_X.shape[1])]

    classification_X_train, classification_X_test,\
    classification_y_train, classification_y_test = \
        train_test_split(classification_X, classification_y, test_size=0.33)

    # Make data frames out of the training and test sets to send them to csv
    # files.
    classification_train = \
        DataFrame(data = np.c_[classification_X_train, classification_y_train],
                  columns = feature_cols + ['target'])
                  
    classification_test = \
        DataFrame(data = np.c_[classification_X_test, classification_y_test],
                  columns = feature_cols + ['target'])

    classification_train.to_csv(CLASSIFICATION_DIR + '/train.csv')
    classification_test.to_csv(CLASSIFICATION_DIR + '/test.csv')
    
    # Make the binary classification model.
    classification_model = GradientBoostingClassifier()
    joblib.dump(classification_model, CLASSIFICATION_DIR + '/classifier.pkl')

    # Make the parameter grid for binary classification.
    classification_search_params = {
        "fit_params": {
            "sample_weight": None
        },
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 4, 6]
        },
        "scoring": [
            "accuracy", 
            "f1", 
            "recall", 
            "precision", 
            "log_loss",
            "roc_auc",
            "average_precision"
        ],
        "estimator": CLASSIFICATION_DIR + "/classifier.pkl"
    }

    param_file = open(CLASSIFICATION_DIR + '/search_params.json', 'w')
    json.dump(classification_search_params, param_file)
    param_file.close()
        
    # Make a directory to store the multiclass data, model, and
    # results.
    os.mkdir(MULTICLASS_DIR)
    multiclass_X, multiclass_y = make_classification(n_classes=3, 
                                                     n_informative=3)
    feature_cols = \
        ["feature_{}".format(ii) 
         for ii in range(multiclass_X.shape[1])]

    multiclass_X_train, multiclass_X_test,\
    multiclass_y_train, multiclass_y_test = \
        train_test_split(multiclass_X, multiclass_y, test_size=0.33)

    # Make data frames out of the training and test sets to send them to csv
    # files.
    multiclass_train = \
        DataFrame(data = np.c_[multiclass_X_train, multiclass_y_train],
                  columns = feature_cols + ['target'])
                  
    multiclass_test = \
        DataFrame(data = np.c_[multiclass_X_test, multiclass_y_test],
                  columns = feature_cols + ['target'])

    multiclass_train.to_csv(MULTICLASS_DIR + '/train.csv')
    multiclass_test.to_csv(MULTICLASS_DIR + '/test.csv')
    
    # Make the binary classification model.
    multiclass_model = GradientBoostingClassifier()
    joblib.dump(multiclass_model, MULTICLASS_DIR + '/classifier.pkl')

    # Make the parameter grid for binary classification.
    multiclass_search_params = {
        "fit_params": {
            "sample_weight": None
        },
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 4, 6]
        },
        "scoring": [
            "f1_micro",
            "f1_macro",
            "precision_micro",
            "precision_macro",
            "recall_micro",
            "recall_macro"
        ],
        "estimator": MULTICLASS_DIR + "/classifier.pkl"
    }

    multiclass_param_file = open(MULTICLASS_DIR + '/search_params.json', 'w')
    json.dump(multiclass_search_params, multiclass_param_file)
    multiclass_param_file.close()

    os.mkdir(REGRESSION_DIR)
    regression_X, regression_y = make_regression()

    feature_cols = \
        ["feature_{}".format(ii) 
         for ii in range(regression_X.shape[1])]

    regression_X_train, regression_X_test,\
    regression_y_train, regression_y_test = \
        train_test_split(regression_X, regression_y, test_size=0.33)

    # Make data frames out of the training and test sets to send them to csv
    # files.
    regression_train = \
        DataFrame(data = np.c_[regression_X_train, regression_y_train],
                  columns = feature_cols + ['target'])
                  
    regression_test = \
        DataFrame(data = np.c_[regression_X_test, regression_y_test],
                  columns = feature_cols + ['target'])

    regression_train.to_csv(REGRESSION_DIR + '/train.csv')
    regression_test.to_csv(REGRESSION_DIR + '/test.csv')
    
    # Make the binary classification model.
    regression_model = SGDRegressor()
    joblib.dump(regression_model, REGRESSION_DIR + '/regressor.pkl')

    # Make the parameter grid for binary classification.
    regression_search_params = {
        "fit_params": {
            "coef_init": 0.1,
            "intercept_init": 0.1
        },
        "param_grid": {
            "alpha": [0.00001, 0.001, 0.01],
            "n_iter":[2, 5, 10]
        },
        "scoring": [
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_median_absolute_error",
            "r2"
        ],
        "estimator": REGRESSION_DIR + "/regressor.pkl"
    }

    regression_param_file = open(REGRESSION_DIR + '/search_params.json', 'w')
    json.dump(regression_search_params, regression_param_file)
    regression_param_file.close()


def tearDownModule():
    # Delete the data.
    subprocess.run(['rm', '-rf', CLASSIFICATION_DIR])
    subprocess.run(['rm', '-rf', MULTICLASS_DIR])
    subprocess.run(['rm', '-rf', REGRESSION_DIR])
    subprocess.run(['rm', '-rf', TEST_OUTPUT_DIR])

class UbergridUnitTest(TestCase):
    
    def test_evaluate_model_classifier(self) -> None:
        """ Tests the _evaluate_model function for the binary classification
            problem.
        """
        # Read in all the stuff we want.
        estimator = joblib.load('classification/classifier.pkl')
        training_data = read_csv('classification/train.csv')
        X = training_data[[col for col in training_data.columns 
                           if col != 'target']]
        y = training_data[['target']]
        param_file = open('classification/search_params.json','r')
        search_params = \
            json.load(param_file)
        param_file.close()
        
        metrics = search_params['scoring']
        prefix = "train"

        # Fit the estimator in preparation for evaluation.
        estimator.fit(X,y)
        
        results = ug._evaluate_model(estimator, X, y, metrics, prefix)
        
        # Values will change each time, we only care that the right keys
        # end up in the results dict.
        result_keys_truth = [
            "train_accuracy",
            "train_f1",
            "train_precision",
            "train_recall",
            "train_log_loss",
            "train_roc_auc",
            "train_average_precision",
            "train_total_prediction_time",
            "train_total_prediction_records"
        ]

        self.assertEqual(sorted(list(results.keys())), 
                         sorted(result_keys_truth))

        # Validate that the ValueError is raised.
        with self.assertRaises(ValueError):
            ug._evaluate_model(estimator, X, y, metrics + ["nope"], prefix)

    def test_evaluate_model_multiclass(self):
        # Read in all the stuff we want.
        estimator = joblib.load('multiclass/classifier.pkl')
        training_data = read_csv('multiclass/train.csv')
        X = training_data[[c for c in training_data.columns if c != 'target']]
        y = training_data[['target']]
        param_file = open('multiclass/search_params.json','r')
        search_params = json.load(param_file)
        metrics = search_params['scoring']
        prefix = 'train'

        # Fit the estimator.
        estimator.fit(X,y)

        results = ug._evaluate_model(estimator, X, y, metrics, prefix)

        # The values will change each time - making sure we can call all the
        # metrics successfully is the goal.
        result_keys_truth = [
            "train_total_prediction_time",
            "train_total_prediction_records",
            "train_f1_micro",
            "train_f1_macro",
            "train_precision_micro",
            "train_precision_macro",
            "train_recall_micro",
            "train_recall_macro"
        ]

        self.assertEqual(sorted(list(results.keys())),
                         sorted(result_keys_truth))

        param_file.close()
    
    def test_evaluate_model_regression(self):
        # Read in all the stuff we want.
        estimator = joblib.load('regression/regressor.pkl')
        training_data = read_csv('regression/train.csv')
        X = training_data[[c for c in training_data.columns if c != "target"]]
        y = training_data[['target']]
        param_file = open('regression/search_params.json', 'r')
        search_params = json.load(param_file)
        metrics = search_params['scoring']
        prefix = "train"

        estimator.fit(X,y)

        results = ug._evaluate_model(estimator, X, y, metrics, prefix)

        # The values will vary from run to run. Here we want to test that all
        # of the fields in the result set are present.

        result_keys_truth = [
            "train_total_prediction_time",
            "train_total_prediction_records",
            "train_neg_mean_absolute_error",
            "train_neg_mean_squared_error",
            "train_neg_median_absolute_error",
            "train_r2"

        ]

        self.assertEqual(sorted(list(results.keys())), 
                         sorted(result_keys_truth))

        param_file.close()

    def test_train_model(self):
        # Read in the stuff we need.
        estimator = joblib.load('classification/classifier.pkl')
        training_data = read_csv('classification/train.csv')
        X = training_data[[c for c in training_data.columns if c != 'target']]
        y = training_data[['target']]
        search_param_file = open('classification/search_params.json','r')
        search_params = json.load(search_param_file)
        metrics = search_params['scoring']
        fit_params = search_params['fit_params']

        estimator, results = \
            ug._train_model(estimator, X, y, metrics, fit_params)

        # The estimator object will crash if fit isn't called, so the test
        # will fail if that's incorrect. This tests to ensure _evaluate_model
        # is called properly.

        result_keys_truth = [
           "training_accuracy",
           "training_f1",
           "training_precision",
           "training_recall",
           "training_log_loss",
           "training_roc_auc",
           "training_average_precision",
           "training_total_prediction_time",
           "training_total_prediction_records",
           "training_time_total"
        ]

        self.assertEqual(sorted(list(results.keys())),
                         sorted(result_keys_truth))
        
        search_param_file.close()

    def test_train_and_evaluate(self):
        # Read the stuff we need.
        estimator = joblib.load('classification/classifier.pkl')
        
        search_param_file = open('classification/search_params.json', 'r')
        search_params = json.load(search_param_file)
        params = ParameterGrid(search_params['param_grid'])[0]
        
        model_id = 0
        
        training_file = 'classification/train.csv'
        validation_file = 'classification/test.csv'

        training_data = read_csv(training_file)
        validation_data = read_csv(validation_file)

        X_train = \
            training_data[[c for c in training_data.columns if c != 'target']]
        X_validation = \
            validation_data[[c for c in validation_data.columns 
                             if c != 'target']]
        y_train = training_data[['target']]
        y_validation = validation_data[['target']]

        metrics = search_params['scoring']
        fit_params = search_params['fit_params']
        target_col = "target"
        
        output_dir = TEST_OUTPUT_DIR

        ug._train_and_evaluate(
            estimator,
            params,
            model_id,
            training_file,
            output_dir,
            X_train,
            y_train,
            target_col,
            metrics,
            fit_params,
            X_validation = X_validation,
            y_validation = y_validation,
            validation_file = validation_file)

        result_keys_truth = [
           "training_accuracy",
           "training_f1",
           "training_precision",
           "training_recall",
           "training_log_loss",
           "training_roc_auc",
           "training_average_precision",
           "training_total_prediction_time",
           "training_total_prediction_records",
           "training_time_total",
           "validation_accuracy",
           "validation_f1",
           "validation_precision",
           "validation_recall",
           "validation_log_loss",
           "validation_roc_auc",
           "validation_average_precision",
           "validation_total_prediction_time",
           "validation_total_prediction_records",
           "training_file",
           "target",
           "model_file",
           "validation_file",
           "max_depth",
           "n_estimators"
        ]

        # Check that the estimator file exists.
        self.assertTrue(
            os.path.exists("{}/model_{}.pkl".format(output_dir, model_id)))

        # Check that the results file exists.
        self.assertTrue(
            os.path.exists("{}/results_{}.json".format(output_dir, model_id)))
        
        # Read in the results file.
        result_file = open("{}/results_{}.json".format(output_dir, model_id))
        results = json.load(result_file)

        self.assertEqual(
            sorted(list(results.keys())),
            sorted(result_keys_truth))
        
        self.assertEqual(results["training_file"], 
                         CLASSIFICATION_DIR + "/train.csv")

        self.assertEqual(results["target"], "target")
        
        self.assertEqual(
            results["model_file"],
            "{}/model_{}.pkl".format(output_dir, model_id))

        self.assertEqual(results["validation_file"], 
                         CLASSIFICATION_DIR + "/test.csv")

        # Cleanup.
        search_param_file.close()
        result_file.close()
        pass

    def test_main(self):
        # Set up the inputs. Thankfully this is a little simpler than the other
        # methods.
        search_params_file = CLASSIFICATION_DIR + "/search_params.json"
        training_file = CLASSIFICATION_DIR + "/train.csv"
        target_col = "target"
        output_dir = TEST_OUTPUT_DIR
        validation_file = CLASSIFICATION_DIR + "/test.csv"

        ug._main(search_params_file,
                 target_col,
                 training_file,
                 output_dir,
                 validation_file=validation_file)
        
        result_keys_truth = [
           "training_accuracy",
           "training_f1",
           "training_precision",
           "training_recall",
           "training_log_loss",
           "training_roc_auc",
           "training_average_precision",
           "training_total_prediction_time",
           "training_total_prediction_records",
           "training_time_total",
           "validation_accuracy",
           "validation_f1",
           "validation_precision",
           "validation_recall",
           "validation_log_loss",
           "validation_roc_auc",
           "validation_average_precision",
           "validation_total_prediction_time",
           "validation_total_prediction_records",
           "training_file",
           "target",
           "model_file",
           "validation_file",
           "max_depth",
           "n_estimators"
        ]
        
        model_ids = range(9)
        self.assertTrue(os.path.exists(TEST_OUTPUT_DIR + "/results.json"))
        for ii in model_ids:
            self.assertTrue(
                os.path.exists(TEST_OUTPUT_DIR + "/model_{}.pkl".format(ii)))
        
        results = [json.loads(l) for l 
                   in open(TEST_OUTPUT_DIR + "/results.json", "r").readlines()]

        # Assert that we have the expected number of runs.
        self.assertEqual(len(results), len(model_ids))

        # For each model id, make sure we have the right columns in the 
        # associated results structure, as well as the right file names.
        for model_id, result in zip(model_ids, results):
            # Assure the file names are filled in correctly.
            # Training data should be in the initial source directory for the 
            # job.
            self.assertEqual(result["training_file"], 
                              CLASSIFICATION_DIR + "/train.csv")
            self.assertEqual(result["target"], "target")
            # Models are stored in the output directory for the grid.
            self.assertEqual(result["model_file"], 
                              TEST_OUTPUT_DIR + 
                              "/model_{}.pkl".format(model_id))
            # Validation data should be in the initial source directory for the
            # job.
            self.assertEqual(result["validation_file"],
                              CLASSIFICATION_DIR + "/test.csv")
            # Make sure the metrics are all present.
            # Technically this is tested in _test_and_evaluate, but checking
            # twice isn't the worst idea in the world.
            self.assertEqual(sorted(list(result.keys())),
                              sorted(result_keys_truth))

        # Test that the _main function raises a ValueError when the validation
        # set has different columns.
        with self.assertRaises(ValueError):
            ug._main(search_params_file,
                     target_col,
                     training_file,
                     output_dir,
                     validation_file = MULTICLASS_DIR + "/test.csv")

        # Test that the _main function raises a ValueError when the 
        # search_params_file doesn't exist.
        with self.assertRaises(ValueError):
            ug._main("not/a/file.json",
                     target_col,
                     training_file,
                     output_dir,
                     validation_file = validation_file)

        # Test that the _main function raises a ValueError when the training
        # file doesn't exist.
        with self.assertRaises(ValueError):
            ug._main(search_params_file,
                     target_col,
                     "not/a/file.csv",
                     output_dir,
                     validation_file = validation_file)

        # Tests that the _main function raises a ValueError when the validation
        # file doesn't exist.
        with self.assertRaises(ValueError):
            ug._main(search_params_file,
                     target_col,
                     training_file,
                     output_dir,
                     validation_file = "not/a/file.csv")

        # Tests that the _main function raises a ValueError when the target col
        # isn't in the training set.
        with self.assertRaises(ValueError):
            ug._main(search_params_file,
                     "not_a_target",
                     training_file,
                     output_dir)

        # Tests that the _main function raises a ValueError when the target col
        # isn't in the validation set.
        with self.assertRaises(ValueError):
            training_data = read_csv(training_file)\
                            .rename(columns={"target":"new_target"})\
                            .to_csv(CLASSIFICATION_DIR + "/other_training.csv")

            ug._main(search_params_file,
                     "new_target",
                     CLASSIFICATION_DIR + "/other_training.csv",
                     output_dir,
                     validation_file = validation_file)