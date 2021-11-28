"""
Testing against dataset
author: Ben E
date: 2021-11-22
"""
import pandas as pd

def get_variables_numeric(variables_yaml):
    return variables_yaml['variables']['valid_numeric_variables']

def get_variables_categorical(variables_yaml):
    return list(variables_yaml['variables']['valid_categorical_variables'])

def get_target(variables_yaml):
    return list(variables_yaml['variables']['target'])

def get_variable_keys(variables_yaml):
    return list(variables_yaml['variables']['keys'])

def test_column_names(data, variables_yaml):
    """
    Check all required columns defined in yaml exist in data file
    """
    variables_keys = get_variable_keys(variables_yaml)
    variables_numeric = get_variables_numeric(variables_yaml)
    variables_categorical = get_variables_categorical(variables_yaml)
    target = get_target(variables_yaml)
    all_required_columns = variables_keys + variables_numeric + variables_categorical + target
    filtered_cols = data.columns[data.columns.isin(all_required_columns)].values
    assert set(filtered_cols) == set(all_required_columns)
    return filtered_cols

def test_row_count(data):
    """
    Checking the size of the dataset - not too small or large
    """
    assert 1 < data.shape[0] < 1000000

def test_variable_numeric(data, variables_yaml):
    """
    Check all the numeric variables are numeric
    """
    variables_numeric = get_variables_numeric(variables_yaml)
    for variable in variables_numeric:
        assert pd.api.types.is_numeric_dtype(data.loc[:,variable].dtypes)

def test_variable_category(data, variables_yaml):
    """
    Check that none of the categorical variables are numeric
    """
    variables_categorical = get_variables_categorical(variables_yaml)
    for variable in variables_categorical:
        assert pd.api.types.is_numeric_dtype(data.loc[:, variable].dtypes) != True

def test_saving_down_file(data, data_output_name, variables_yaml):
    """
    Save down the dataset with all features and targets
    """
    all_required_columns = test_column_names(data, variables_yaml)
    data_output = data.loc[:, all_required_columns]
    data_output.to_csv(data_output_name, index=False)