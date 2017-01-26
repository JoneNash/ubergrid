import os
import subprocess
import json

from unittest import TestCase

import numpy as np
from pandas import DataFrame, read_csv
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import ubergrid as ug

def setUpModule():
    # Make a directory to store the classification data, model, and
    # results.
    os.mkdir('classification')
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

    classification_train.to_csv('classification/train.csv')
    classification_test.to_csv('classification/test.csv')
    
    # Make the binary classification model.
    classification_model = GradientBoostingClassifier()
    joblib.dump(classification_model, 'classification/classifier.pkl')

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
        ]
    }

    param_file = open('classification/search_params.json', 'w')
    json.dump(classification_search_params, param_file)
    param_file.close()
        
    # Make a directory to store the multiclass data, model, and
    # results.
    os.mkdir('multiclass')
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

    multiclass_train.to_csv('multiclass/train.csv')
    multiclass_test.to_csv('multiclass/test.csv')
    
    # Make the binary classification model.
    multiclass_model = GradientBoostingClassifier()
    joblib.dump(multiclass_model, 'multiclass/classifier.pkl')

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
        ]
    }

    multiclass_param_file = open('multiclass/search_params.json', 'w')
    json.dump(multiclass_search_params, multiclass_param_file)
    multiclass_param_file.close()
    # TODO: Create grid + model + data for regression.

def tearDownModule():
    # Delete the data.
    subprocess.run(['rm', '-rf', 'classification/'])
    subprocess.run(['rm', '-rf', 'multiclass/'])

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
        param_file.close()

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

        pass