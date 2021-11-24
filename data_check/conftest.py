"""
Validation step for the data
author: Ben E
date: 2021-11-23
"""
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
import pytest


def pytest_addoption(parser):
    parser.addoption("--csv_input", action="store")
    parser.addoption("--csv_checked_output", action="store")
    parser.addoption("--yaml_file", action="store")

@pytest.fixture(scope='session')
def data(request):
    data_path = request.config.option.csv_input
    if data_path is None:
        pytest.fail("You must provide the --csv_input option on the command line")
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope='session')
def data_output_name(request):
    output_filename = request.config.option.csv_checked_output
    if output_filename is None:
        pytest.fail("You must provide --csv_checked_output on the command line")
    return output_filename

@pytest.fixture(scope='session')
def variables_yaml(request):
    yaml_path = request.config.option.yaml_file
    with open(yaml_path, "r") as yml:
        try:
            yaml_file = yaml.safe_load(yml)
        except yaml.YAMLError as exc:
            pytest.fail("You must provide the --yaml_file option on the command line")
    return yaml_file


