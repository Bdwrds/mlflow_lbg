"""
This script trains two tree based models
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    logger.info("RUNNING TRAINING SECTION")
    # Get the Random Forest configuration
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
        rf_config = model_config['random_forest']
        dt_config = model_config['decision_tree']
    # Get all variables used for training
    with open(args.yaml_variables) as fp:
        variables_yaml = yaml.safe_load(fp)

    # Fix the random seed for the Random Forest for reproducible results
    rf_config['random_state'] = args.random_seed
    dt_config['random_state'] = args.random_seed

    variables_numeric = variables_yaml['variables']['valid_numeric_variables']
    variables_categorical = list(variables_yaml['variables']['valid_categorical_variables'])
    processed_features = variables_numeric + variables_categorical

    # load name of target
    target = list(variables_yaml['variables']['target'])

    X = pd.read_csv(args.trainval_data)
    y = X.pop(target[0])
    X_cat = pd.get_dummies(X.loc[:, variables_categorical])
    variables_categorical_onehot = list(X_cat.columns.values)
    processed_features = variables_numeric + variables_categorical_onehot

    # split data into train/ valid
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size= args.val_size, stratify=y, random_state= args.random_seed
    )

    # create model
    model_rf = RandomForestClassifier(**rf_config)
    model_dt = DecisionTreeClassifier(**dt_config)

    logger.info("Preparing sklearn pipeline")
    sk_pipe_rf = get_inference_pipeline(rf_config, model_rf, variables_numeric, variables_categorical)
    sk_pipe_dt = get_inference_pipeline(dt_config, model_dt, variables_numeric, variables_categorical)

    logger.info("Fitting")
    ######################################
    # Fit the pipeline sk_pipe - applies transforms -> modelling
    sk_pipe_rf.fit(X_train, y_train)
    sk_pipe_dt.fit(X_train, y_train)
    ######################################

    # Compute AUC/ F2-score
    logger.info("Scoring")
    metric_auc = sk_pipe_rf.score(X_val, y_val)
    y_pred = sk_pipe_rf.predict(X_val)
    metric_recall= recall_score(y_val, y_pred)
    metric_f2 = fbeta_score(y_val, y_pred, beta=2)
    logger.info(f"DT Recall Score: {metric_recall}")
    logger.info(f"RF F2 Score: {metric_f2}")
    logger.info("Exporting RF model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("../random_forest_dir"):
        shutil.rmtree("../random_forest_dir")
    export_path = os.path.join(os.getcwd(), "../random_forest_dir")
    mlflow.sklearn.save_model(
        sk_pipe_rf,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2],
    )

    fig_feat_imp = plot_feature_importance(sk_pipe_rf, processed_features)
    fig_feat_imp.savefig(os.path.join(os.getcwd(),'results/var_imp_rf.png'))
    ######################################

    ######################################
    metric_auc = sk_pipe_dt.score(X_val, y_val)
    y_pred = sk_pipe_dt.predict(X_val)
    metric_recall = recall_score(y_val, y_pred)
    metric_f2 = fbeta_score(y_val, y_pred, beta=2)
    logger.info(f"DT Recall Score: {metric_recall}")
    logger.info(f"DT F2 Score: {metric_f2}")
    logger.info("Exporting DT model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("../decision_tree_dir"):
        shutil.rmtree("../decision_tree_dir")
    export_path = os.path.join(os.getcwd(), "../decision_tree_dir")
    mlflow.sklearn.save_model(
        sk_pipe_dt,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2],
    )
    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe_dt, processed_features)
    fig_feat_imp.savefig(os.path.join(os.getcwd(),'results/var_imp_dt.png'))

    # export the tree structure as a log file
    text_representation = export_text(sk_pipe_dt['model'], feature_names=processed_features)
    with open(os.path.join(os.getcwd(), 'results/tree_text.log'), 'w') as dtree:
        dtree.write(text_representation)

    # export the tree structure as a plot
    plt.figure(figsize=(60,30))
    dt_tree = plot_tree(sk_pipe_dt['model'], feature_names=processed_features, \
                        fontsize=5, filled=True)
    plt.savefig(os.path.join(os.getcwd(), 'results/tree_dt.png'))
    ######################################

def plot_feature_importance(pipe, feat_names):
    #feat_names = pipe.feature_names_in_
    feat_imp = pipe["model"].feature_importances_
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.barh(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_yticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_yticklabels(np.array(feat_names), rotation=0)
    sub_feat_imp.invert_yaxis()
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(model_config, model_type, variables_numeric, variables_categorical):

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
            ("impute_zero", zero_imputer, variables_numeric),
            ("ordinal_cat", categorical_preproc, variables_categorical)
        ],
        #remainder = 'passthrough'
    )

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_type),
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
        "--model_config",
        help="Model configuration.",
        default="none",
    )
    parser.add_argument(
        "--yaml_variables",
        help="Variables used for modelling.",
        default="none",
    )
    args = parser.parse_args()
    go(args)
