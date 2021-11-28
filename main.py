"""
Primary file for running all steps in this project
author: Ben E
Date:: 2021-11-23
"""
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "preprocessing",
    "data_check",
    "data_split",
    "train_model",
    "test_model",
    "inference"
]

# Reads in the configuration
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
                    "yaml_file": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_variables'])
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
                    "yaml_file": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_variables'])
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
                    "stratify_by": config['modeling']['stratify_by'],
                    "csv_train": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_train']),
                    "csv_test": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_test'])
                }
            )

        if "train_model" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "modelling/train_model"),
                "main",
                parameters={
                    "trainval_data": os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_train']),
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "model_config": os.path.join(hydra.utils.get_original_cwd(),config['modeling']['model_param']),
                    "yaml_variables": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_variables']),
                    "output_artifact": "random_forest_export",
                }
            )

        if "test_model" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "modelling/test_model"),
                "main",
                parameters={
                    "mlflow_model": os.path.join(hydra.utils.get_original_cwd(),
                                                 config['production']['model_dir']),
                    "test_data":  os.path.join(hydra.utils.get_original_cwd(), config['etl']['csv_test']),
                    "yaml_variables": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_variables']),
                }
            )

        if "inference" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "inference"),
                "main",
                parameters={
                    "mlflow_model": os.path.join(hydra.utils.get_original_cwd(),
                                                 config['production']['model_dir']),
                    "infer_data":  os.path.join(hydra.utils.get_original_cwd(), config['production']['csv_test']),
                    "yaml_variables": os.path.join(hydra.utils.get_original_cwd(), config['etl']['yaml_variables']),
                    "csv_output": os.path.join(hydra.utils.get_original_cwd(), config['production']['csv_output']),
                }
            )

if __name__ == "__main__":
    go()