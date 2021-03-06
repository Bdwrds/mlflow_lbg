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
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score

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
    # load name of target
    target = list(variables_yaml['variables']['target'])

    logger.info("Loading test data")
    X_test = pd.read_csv(args.test_data)
    y_test = X_test.pop(target[0])

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(args.mlflow_model)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    metric_auc = sk_pipe.score(X_test, y_test)
    metric_precision = precision_score(y_test, y_pred)
    metric_recall = recall_score(y_test, y_pred)
    metric_f1 = f1_score(y_test, y_pred)
    metric_f2 = fbeta_score(y_test, y_pred, beta=2)

    logger.info(f"AUC: {metric_auc}")
    logger.info(f"Precision: {metric_precision}")
    logger.info(f"Recall: {metric_recall}")
    logger.info(f"F1 Score: {metric_f1}")
    logger.info(f"F2 Score: {metric_f2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")
    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="Test dataset",
        required=True
    )
    parser.add_argument(
        "--yaml_variables",
        type=str,
        help="List of variables used in model",
        required=True
    )
    args = parser.parse_args()
    go(args)
