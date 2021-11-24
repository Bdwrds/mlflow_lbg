"""
This script trains a Random Forest
author: Ben E
date: 2021-11-23
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import yaml
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Get the Random Forest configuration
    with open(args.rf_config) as fp:
        rf_config = yaml.safe_load(fp)['random_forest']
    # Get all variables used for training
    with open(args.yaml_variables) as fp:
        variables_yaml = yaml.safe_load(fp)

    # Fix the random seed for the Random Forest for reproducible results
    rf_config['random_state'] = args.random_seed

    variables_numeric = variables_yaml['variables']['valid_numeric_variables']
    variables_categorical = list(variables_yaml['variables']['valid_categorical_variables'])
    processed_features = variables_numeric + variables_categorical

    # load name of target
    target = list(variables_yaml['variables']['target'])

    X = pd.read_csv(args.trainval_data)
    y = X.pop(target[0])

    # split data into train/ valid
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size= args.val_size, stratify=y, random_state= args.random_seed
    )

    logger.info("Preparing sklearn pipeline")
    sk_pipe = get_inference_pipeline(rf_config, variables_numeric, variables_categorical)

    logger.info("Fitting")
    ######################################
    # Fit the pipeline sk_pipe - applies transforms -> modelling
    sk_pipe.fit(X_train, y_train)
    ######################################

    # Compute AUC/ F1-score
    logger.info("Scoring")

    metric_auc = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    metric_f1 = f1_score(y_val, y_pred)

    logger.info(f"AUC Score: {metric_auc}")
    logger.info(f"F1 Score: {metric_f1}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    ######################################
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    export_path = os.path.join(os.getcwd(), "random_forest_dir")
    #signature = infer_signature(X_val, y_val)
    mlflow.sklearn.save_model(
        sk_pipe,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        #signature=signature,
        input_example=X_val.iloc[:2],
    )

    ######################################
    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    fig_feat_imp.savefig('var_imp.png')
    ######################################


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest_model"].feature_importances_
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.barh(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_yticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_yticklabels(np.array(feat_names), rotation=0)
    sub_feat_imp.invert_yaxis()
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, variables_numeric, variables_categorical):

    ######################################
    # Impute with the most frequent field
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    # Let's impute the numerical columns to make sure we can handle missing values
    zero_imputer = SimpleImputer(strategy="median")

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", categorical_preproc, variables_categorical),
            ("impute_zero", zero_imputer, variables_numeric)
        ],
        remainder="drop",
    )

    # create model
    random_forest_model = RandomForestClassifier(**rf_config)

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest_model", random_forest_model),
            ]
         )
    ######################################
    return sk_pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")
    parser.add_argument(
        "--trainval_data",
        type=str,
        help="CSV containing the training dataset. It will be split into train and validation"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )
    parser.add_argument(
        "--rf_config",
        help="Random forest configuration.",
        default="none",
    )
    parser.add_argument(
        "--yaml_variables",
        help="Variables used for modelling.",
        default="none",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )
    args = parser.parse_args()
    go(args)
