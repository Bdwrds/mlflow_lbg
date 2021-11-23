"""
Primary file for running all steps in project
author: Ben E
Date:: 2021-11-23
"""
import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "preprocessing",
    "data_check",
    "data_split",
    "train_models",
    "test_model"
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "preprocessing" in active_steps:
            # Download data, clean and upload
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "preprocessing"),
                "main",
                parameters={
                    "csv_campaign": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_campaign']),
                    "csv_mortgage": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_mortgage']),
                    "csv_output": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_output']),
                    "csv_output_rem": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_output_rem']),
                    "yaml_file": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_file'])
                },
            )

        if "data_check" in active_steps:
            # Check data
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "data_check"),
                "main",
                parameters={
                    "csv_input": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_output']),
                    "csv_checked_output": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_checked_output']),
                    "yaml_file": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_file'])
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "data_split"),
                "main",
                parameters={
                    "csv_clean": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_checked_output']),
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                }
            )

        if "train_models" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": "random_forest_export",
                    "transform_artifact": "transform_artifact",
                }
            )

        if "test_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset":  "test_data.csv:latest",
                }
            )

if __name__ == "__main__":
    go()
