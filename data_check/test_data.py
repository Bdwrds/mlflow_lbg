"""
Testing against dataset
author: Ben E
date: 2021-11-22
"""
import pandas as pd

def test_column_names(data, variables_yaml):
    """
    Check all required columns defined in yaml exist in data file
    """
    features_numeric = variables_yaml['valid_numeric_features']
    features_categorical = list(variables_yaml['valid_categorical_features'])
    target = list(variables_yaml['target'])
    all_required_columns = features_numeric + features_categorical + target
    filtered_cols = data.columns[data.columns.isin(all_required_columns)].values
    assert set(filtered_cols) == set(all_required_columns)
    return filtered_cols

def test_row_count(data):
    """
    Checking the size of the dataset - not too small or large
    """
    assert 15000 < data.shape[0] < 1000000

def test_saving_down_file(data, data_output_name, variables_yaml):
    all_required_columns = test_column_names(data, variables_yaml)
    data_output = data.loc[:, all_required_columns]
    data_output.to_csv(data_output_name, index=False)