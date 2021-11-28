"""
Testing script - run inference of model against test data set
author: Ben E
date: 2021-11-24
"""
import argparse
import logging
import mlflow
import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    logger.info("RUNNING TEST SECTION")
    # Get all variables used for training
    with open(args.yaml_variables) as fp:
        variables_yaml = yaml.safe_load(fp)

    variables_numeric = variables_yaml['variables']['valid_numeric_variables']
    variables_categorical = list(variables_yaml['variables']['valid_categorical_variables'])
    processed_features = variables_numeric + variables_categorical

    logger.info("Loading test data")
    X_test = pd.read_csv(args.infer_data)

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(args.mlflow_model)
    y_pred = pd.DataFrame(sk_pipe.predict(X_test))

    y_pred.to_csv(args.csv_output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")
    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )
    parser.add_argument(
        "--infer_data",
        type=str,
        help="New data to run inference on",
        required=True
    )
    parser.add_argument(
        "--yaml_variables",
        type=str,
        help="List of variables used in model",
        required=True
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        help="Output file with predictions",
        required=True
    )
    args = parser.parse_args()
    go(args)
