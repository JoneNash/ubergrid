# Ubergrid

Ubergrid is a command line tool for executing large or long running hyperparameter grid searches with scikit-learn.
It builds, tests, times and persists every model in the grid, and is used for cases when a `GridSearchCV` is not appropriate:

* Training time is fifteen minutes and you don't want to wait ten hours to complete the search with 3-fold cross validation.
* The parameter grid is large and you don't want to guess what'll happen if your computer goes to sleep with your Jupyter notebook running.
* You want to assess the tradeoff between predictive performance and runtime (10x-ing the prediction time of a model for a 0.002 gain in AUC is only appropriate on Kaggle).
* You have to serialize models to PMML and want a straightforward way to do that on the parameter grid results.

If any of these cases apply to you, ubergrid can help.
It's designed to treat hyperparameter tuning as a process, rather than just one more thing to overfit, by presenting a complete picture of the model you're building so it's easy to see what the tradeoffs in the parameter space are.
It's also well suited for searches that take a while.
For example, it will resume where it left off if the process is terminated before it completes (i.e. computer goes to sleep, AWS goes down, ...) - because there's no summarization or selection, models in the grid are built and persisted incrementally.
This also allows models to be trained and evaluated in parallel, just like `GridSearchCV`.

**This is a hobby project.**
I've been working on this for a couple of months at home on my own time, which amounts to about an hour or so a night because toddlers.
I can't make any guarantees around roadmap timelines and new features, but I think what's here is solid enough to release.
I can say I will try my best fix bugs as quickly as possible.

## Installation

Installing ubergrid is straightforward.
Clone this repo, then `cd` into it.

```shell
pip install --editable .
```

You're done. Pip will install the dependencies, which are:

1. [click](http://click.pocoo.org/5/)
2. [scikit-learn](http://scikit-learn.org/)
3. [toolz](http://toolz.readthedocs.io/en/latest/api.html)
4. [pandas](http://pandas.pydata.org/)
5. [numpy](http://www.numpy.org/)
6. [sklearn-pandas](https://github.com/paulgb/sklearn-pandas)

The default installation not include JPMML support.
To enable it you also need to install [sklearn2pmml](https://github.com/jpmml/sklearn2pmml), which you can do with pip.

```shell
pip install -e git+https://github.com/jpmml/sklearn2pmml.git#egg=sklearn2pmml
```

This isn't included in `setup.py` for ubergrid because it's not a PyPI package, and seems to be in flux a bit.
I wanted to keep the JPMML support optional because it's not batteries included and not everyone will need it.

To evaluate and time the PMMLs, you'll need the [JPMML Evaluator](https://github.com/jpmml/jpmml-evaluator) jar.

## Quick Start

For the impatient, this repo has some scripts that pull sample data and set up a model for running.
After you've installed ubergrid, cd into the `sample` directory.
From there, execute the script `setup_sample.py`, which creates the estimator (`booster.pkl`) and the parameter grid file (`booster_grid_params.json`).
Then, cd into the `data` directory and execute the shell script `get_blood_donations.sh`, which downloads the blood donation dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center) and changes the headers a little bit to produce `data/donations.csv`.
Now you're ready to run ubergrid.

```shell
ubergrid run \
    booster_grid_params.json \
    donated \
    data/donations.csv \
    models \
    --cross-validation 3
```

`booster_grid_params.json` is the file with the grid search parameters.
`donated` is the name of the target column is the training set.
`data/donations.csv` is the name of the training file.
`models` is the name of the (future) output directory. 
If it doesn't exist, ubergrid will make it.

If you check out `models/results.json` you'll see line separated JSON dictionaries with the model results.
You don't need to do that with vim though.
Fire up a python interpreter.

```python
import ubergrid as ug

# This has all of the results from the run in a data frame.
# You can also get a list of JSON dicts if you want (ug.get_results( ... )).
results_frame = ug.read_results_frame('models')

# Calculate the mean cross validation AUC for the grid.
max_auc = results_frame.cross_validation_roc_auc.max()
>>> 0.609

# Let's find the best model.
results_frame[results_frame.cross_validation_roc_auc == max_auc]\
    [['max_depth', 'n_estimators']]


>>> max_depth = 2
>>> n_estimators = 100

# Use get_model to retrieve the model with matching parameters.
best_ish_model = ug.get_model('models', n_estimators=100, max_depth=2)

>>> GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=2,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=100, presort='auto', random_state=None,
              subsample=1.0, verbose=0, warm_start=False)

```

In the above example, only the `ubergrid run` command is used.
There's also `ubergrid jpmml` that you can run on already existing results to serialize the models to PMML.
There are more details on that later in the documentation.

## Run

The `ubergrid run` command executes the grid search.
These are the arguments and options.

```shell
Usage: ubergrid run [OPTIONS] SEARCH_PARAMS_FILE TARGET_COL TRAINING_FILE
                    OUTPUT_DIR

  Runs the grid search.

Arguments:

    SEARCH_PARAMS_FILE - The JSON file containing the grid search
    parameters.

    TARGET_COL - The name of the target variable column.

    TRAINING_FILE - The name of the training file (csv with headers).

    OUTPUT_DIR - The name of the directory that will hold the results.
    If it does not exist, ubergrid will make it.

Options:
  -v, --validation-file TEXT      The name of the file with the validation
                                  set.
  -c, --cross-validation INTEGER  The number of cross validation folds to
                                  apply.
  -j, --n-jobs INTEGER            The number of jobs (in parallel) to run.
  -d, --dry-run                   Run with only logging.
  --help                          Show this message and exit.
```

`SEARCH_PARAMS_FILE` is the JSON file that defines the grid.
It's a dict that requires three fields:

```javascript
{
    // Required.

    // The estimator field should hold the path to the untrained estimator.
    "estimator": "/path/to/estimator.pkl",
    // Scoring holds a list of metrics to score. See below for the avaliable
    // metrics.
    "scoring": ["metrics", "to", "score"],
    // param_grid holds a dict that maps the parameter names to the values you
    // want to search.
    "param_grid": {
        "param_1": ["param_1_value", "other_param_1_value"],
        "param_2": ["param_2_value", "other_param_2_value"]
    },

    // Optional.
    // fit_params holds a dict mapping the names and values of parameters to
    // be passed to the estimator's fit method.
    "fit_params": {
        "fit_param_1": "value_of_fit_param_1"
    }
}
```

### Available Scorers

The scorers that are available to ubergrid are the ones in scikit-learn's [sklearn.metrics.SCORERS](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) dict.

For binary classification:
* `accuracy`
* `f1`
* `recall`
* `precision`
* `log_loss`
* `roc_auc`

For multiclass classification:
* `average_precision`
* `f1_micro`
* `f1_macro`
* `precision_micro`
* `precision_macro`
* `recall_micro`
* `recall_macro`

For regression:
* `neg_mean_absolute_error`
* `neg_mean_squared_error`
* `neg_median_absolute_error`
* `r2`

### Output

The output of ubergrid's run command is a directory containing all of the models and a file with the results.

```
ubergrid_output/
    results.json
    model_0.pkl
    model_1.pkl
    ...
```

The `results.json` file contains everything needed to evaluate and retrieve the best model.
It's a line separated file of JSON objects, with one object per model.
Each of those objects has the following fields:

```javascript
{
    // Files used to build and evaluate the model.
    "training_file": "/path/to/training.csv",
    "model_file": "/path/to/model.pkl",
    "validation_file": "/path_to_validation.csv", // If used.

    // The name of the target column.
    "target": "target_col_name",

    // Parameters that identify the model
    "param_1": param_value_1,
    "param_2": param_value_2,
    // ...

    // The metrics for training
    "training_time_total": time_for_training,
    "training_prediction_time": time_for_predictions,
    "training_total_prediction_records": number_of_records,
    "training_{metric}": value,
    "training_{metric}": value,
    // ...

    // The metrics for cross validation, if cross validation was
    // performed.

    // The cross validation metrics, summarized by mean.
    "cross_validation_{metric}": mean_value_metric,
    // ... other metrics.
    "cross_validation_total_prediction_time": mean_prediction_time,
    "cross_validation_total_prediction_records": 
        mean_number_of_records,

    // All cross validation runs.    
    "cross_validation_{metric}_all": list_of_metric_values,
    // ... other metrics.
    "cross_validation_total_prediction_time_all":
        list_of_prediction_times,
    "cross_validation_total_prediction_records_all":
        list_of_prediction_records,

    // The cross validation runs summarized with mean for training. 
    "cross_validation_training_{metric}": mean_metric_on_training,
    // ... other metrics.
    "cross_validation_training_total_prediction_time":
        mean_total_prediction_time,
    "cross_validation_training_total_prediction_records":
        mean_total_prediction_records,
    "cross_validation_training_time_total":
        mean_time_to_train,

    // All cross validation training runs.
    "cross_validation_training_{metric}_all": list_of_metric_values,
    // ... other metrics.
    "cross_validation_training_total_prediction_time_all":
        list_of_training_prediction_times,
    "cross_validation_training_total_prediction_records_all":
        list_of_training_prediction_records,
    "cross_validation_training_time_total_all":
        list_of_training_times,

    // The metrics for validation, if validation was performed.
    "validation_prediction_time": time_for_predictions_validation,
    "validation_total_prediction_records": number_of_validation_records,
    "validation_{metric}": value,
    "validation_{metric}": value,
    // ...
}
```

### Dry Run

Before executing a full grid search, it might be usefult to simply log what ubergrid is _going_ to do without actually doing it.
This way you can see precisely how many models you're building and what parameters they're being built with.
Simply add the `--dry-run` option to the `ubergrid run` command to get just the logging.

```shell
ubergrid run params.json target train.csv output --dry-run
```

## Analyze

Ubergrid also comes with a library that has a few utility functions that are useful when analyzing grid search results.

```python
import ubergrid as ug

# Reads the results.json file into a list of dicts.
results = ug.read_results('path/to/output')

# Reads the results.json file into a pandas dataframe, ignoring the fields that
# are lists.
results_frame = ug.read_results_frame('/path/to/output')

# Obtains the model with the specified parameters from the results.
estimator = ug.get_model(results, param_1=val1, param_2=val2)

# You can also pass it the path instead of the results list.
same_estimator = ug.get_model('path/to/output', param_1=val1, param_2=val2)

# If you don't provide enough parameters to uniquely specify a model, an
# exception is thrown.
nope = ug.get_model(results, param_1=val1)
```

## JPMML

The ubergrid jpmml command takes an already solved parameter grid and serializes all of the models to PMML.
It will also optionally time the predictions of the PMML evaluator against a provided dataset.

```
Usage: ubergrid jpmml [OPTIONS] RESULTS_DIR

  Takes an existing ubergrid search and builds PMML files with JPMML.

  Arguments:

      RESULTS_DIR - The name of the directory with a completed ubergrid run.

Options:
  -p, --pmml-evaluator TEXT    The name of the JPMML evaluator jar.
  -f, --file-to-evaluate TEXT  The name of the file to evaluate. Must be
                               specified with --pmml-evaluator option.
  --help                       Show this message and exit.
```

Assuming an already run grid like this:

```
ubergrid_output/
    results.json
    model_0.pkl
    model_1.pkl
    ...
```

Running the command

```shell
ubergrid jpmml ubergrid_output
```

will change that directory to

```
ubergrid_output/
    results.json
    model_0.pkl
    model_0.pmml
    model_1.pkl
    model_1.pmml
    ...
```

It also adds a field to all of the `results.json` dicts.

```javascript
{
    "pmml_file": "ubergrid_output/model_0.pmml",

    // All the other stuff.
    // ...
}
```

To run the timing, specify two additional optional arguments:

```shell
ubergrid jpmml \
    ubergrid_output \
    --pmml-evaluator example-1.3-SNAPSHOT.jar \
    --file-to-evaluate training.csv
```

The `--pmml-evaluator` argument requires a [JPMML Evaluator](https://github.com/jpmml/jpmml-evaluator) jar.
The version as of this writing is 1.3 but it's slottable as long as it's called in the same way, which may or may not be true for future releases.
I'll do my best to keep up.
When you run the jpmml command you get another couple of fields added to `results.json`:

```javascript
{
    "pmml_file": "ubergrid_output/model_0.pmml",

    // If PMML evaluation was selected.
    "pmml_total_prediction_time": time_for_pmml_prediction,
    "pmml_total_prediction_records": number_of_pmml_predictions,

    // Plus everything else.
    // ...
}
```

It's important to note that there are a couple of issues with the timing.
Ubergrid calls the evaluator _per model_ as a subprocess, which means the JVM startup time is included.
Roughly speaking you can compensate for this by subtracting 1 or 2 seconds from the total prediction time before dividing by the number of records to get the time per prediction.
If the file's large enough (but not so large as to overfill the JVM heap) this won't change the results all that much.
In addition to the JVM warmup the JPMML evaluator program must read and deserialize the csv file it's predicting.
As of 1.3, this is done on multiple cores, but it's still a non-trivial contribution to the timing.
The timing information is still valuable because both JVM warmup and csv deserialization are constant across runs, so the relative runtime changes between different sets of model parameters should be unaffected.