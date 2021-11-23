#!/usr/bin/env python
"""
Basic clean and preprocess
author: Ben E
date: 2021-11-23
"""
import argparse
import logging
import pandas as pd
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def tidy_fields(series: pd.Series, valid_options, other_field):
    for key, value in valid_options.items():
        series.loc[series.str.contains(key)] = value
    series.loc[~series.isin(list(valid_options.values()))] = other_field
    return series

def update_valid_fields(series, valid_options):
    series[~series.isin(valid_options)] = 'other'
    return series

def convert_salary_band_to_numeric(sband: pd.Series):
    # get base value
    sband['number_value'] = sband.salary_band.str.extract('(\d+)').astype('float')
    # obtain midpoint from ranged values
    mix_range = \
    sband.loc[sband.salary_band.str.contains('range')].salary_band.str.extractall('(\d+)').astype('float')
    range_midpoint = (mix_range[0].reset_index().pivot(index='level_0',columns=['match'])[0][0] + \
                      mix_range[0].reset_index().pivot(index='level_0',columns=['match'])[0][1]) / 2
    # update range, weekly and monthly payments
    sband.loc[sband.salary_band.str.contains('range'),'number_value'] = range_midpoint
    sband.loc[sband.salary_band.str.contains('pw'),'number_value'] *= 52
    sband.loc[sband.salary_band.str.contains('month'),'number_value'] *= 12
    return sband.number_value.values


def go(args):
    logger.info("Load data and apply basic clean on price")
    # loading both csv files
    df_campaign = pd.read_csv(args.csv_campaign)
    df_mortgage = pd.read_csv(args.csv_mortgage)

    logger.info("Loading yaml file")
    with open(args.yaml_file, "r") as yml:
        try:
            yaml_file = yaml.safe_load(yml)['variables']
            valid_company_email_adj_top_5 = yaml_file['valid_categorical_features']['company_email_adj_top_5']
            valid_town_adj_top_5 = yaml_file['valid_categorical_features']['town_adj_top_5']
            valid_salary_band_text = yaml_file['mappings']['salary_band_text']
            valid_workclass = yaml_file['mappings']['workclass']
            valid_education_order = yaml_file['mappings']['education_mapping_order']
        except yaml.YAMLError as exc:
            logger.info("Failed loading yaml file", exc)


    # filter for the same customer base and combine
    df_mortgage_sub = df_mortgage.iloc[0: df_campaign.shape[0], :].copy()
    df_combined = pd.merge(df_campaign, df_mortgage_sub, left_index=True, right_index=True)

    ## NEW FEATURES
    logger.info("Create new features")
    ## has_married
    df_combined['has_married'] = np.where(df_combined.marital_status.isin(['Divorced', 'Never-married']), 'No', 'Yes')

    ## salary_band
    df_combined['salary_band_text'] = df_combined.salary_band.str.replace('(\d+)', '', regex=True)
    # update and remove the long tail or foreign currencies
    df_combined['salary_band_text_adj'] = tidy_fields(df_combined['salary_band_text'].copy(), valid_salary_band_text, 'foreign_ccy')

    # tidy the salary variable to an annual payment
    df_combined['annual_salary'] = convert_salary_band_to_numeric(df_combined.loc[:, ['salary_band']].copy())
    # cap the upper limit at the 97.5 percentile
    upper_bound = df_combined.annual_salary.quantile(0.975)
    df_combined.loc[df_combined.annual_salary > upper_bound, ['annual_salary']] = upper_bound

    # combine years/months_with_employer to have total months at employer
    df_combined['total_months_with_employer'] = \
        df_combined['years_with_employer'] * 12 + df_combined['months_with_employer']

    # new feature with ordered number from category
    df_combined['education_order'] = df_combined.education.map(valid_education_order)

    ## Most popular towns
    df_combined['company_email_address'] = df_combined.company_email.str.split('@', expand=True)[1]
    df_combined['company_email_adj_top_5'] = \
        update_valid_fields(df_combined['company_email_address'].copy(), valid_company_email_adj_top_5)

    ## Most popular towns
    df_combined['town_adj_top_5'] = update_valid_fields(df_combined['town'].copy(), valid_town_adj_top_5)

    ## workclass
    df_combined['workclass_adj'] = tidy_fields(df_combined.workclass.copy(), valid_workclass, 'other')

    # convert binary flags from numeric to string
    df_combined['interested_insurance'] = np.where(df_combined['interested_insurance'] == 1, 'Yes', 'No')

    ## finally write new df to csv
    logger.info("Write cleaned df to csv")
    df_combined.to_csv(args.csv_output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This step cleans the data")
    parser.add_argument(
        "--csv_campaign",
        type=str,
        help="Name of campaign csv file",
        required=True
    )
    parser.add_argument(
        "--csv_mortgage",
        type=str,
        help="Name of mortgage csv file",
        required=True
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        help="Name of the cleaned csv file",
        required=True
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        help="Location of a yaml file covering input variables",
        required=True
    )
    args = parser.parse_args()
    go(args)
