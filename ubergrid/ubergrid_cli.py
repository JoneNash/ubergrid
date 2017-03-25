import click

import ubergrid.ubergrid_core as ugc
import ubergrid.ubergrid_jpmml as ugj

@click.group()
def cli():
    pass

@cli.command()
@click.argument("search_params_file", type=str)
@click.argument("target_col", type=str)
@click.argument("training_file", type=str)
@click.argument("output_dir", type=str)
@click.option("--validation-file", "-v", 
              type=str, 
              default=None,
              help="The name of the file with the validation set.")
@click.option("--cross-validation", "-c", 
              type=int, 
              default=None,
              help="The number of cross validation folds to apply.")
@click.option("--n-jobs", "-j", 
              type=int, 
              default=1,
              help="The number of jobs (in parallel) to run.")
@click.option("--dry-run", "-d", 
              is_flag=True,
              help="Run with only logging.")
def run(search_params_file: str,
        target_col: str,
        training_file: str,
        output_dir: str,
        validation_file: str,
        cross_validation: int,
        n_jobs: int,
        dry_run: bool):
    """ 
    Runs the grid search.

    Usage: 

    ubergrid search_params_file target_col training_file output_dir [OPTIONS]

    Positional arguments:

        search_params_file - The JSON file containing the grid search
            parameters.

        target_col - The name of the target variable column.

        training_file - The name of the training file (csv with headers).

        output_dir - The name of the directory that will hold the results.
        If it does not exist, ubergrid will make it.
    """
    ugc._main(search_params_file,
              target_col,
              training_file,
              output_dir,
              validation_file = validation_file,
              cross_validation = cross_validation,
              n_jobs = n_jobs,
              dry_run = dry_run)

@cli.command()
@click.argument("results_dir", type=str)
@click.option("--pmml-evaluator", "-p",
              default=None,
              type=str,
              help="The name of the JPMML evaluator jar.")
@click.option("--file-to-evaluate", "-f",
              default=None,
              type=str,
              help="The name of the file to evaluate. "
                    "Must be specified with --pmml-evaluator option.")
def jpmml(results_dir: str,
          pmml_evaluator: str,
          file_to_evaluate: str):
    """ 
    Takes an existing ubergrid search and builds PMML files with JPMML.

    Usage:

        ubergrid jpmml results [OPTIONS]

    Positional arguments:

        results - The name of the directory with a completed ubergrid run.
    """
    ugj._main(results_dir, pmml_evaluator, file_to_evaluate)