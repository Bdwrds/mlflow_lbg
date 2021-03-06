"""
This script splits the provided dataframe in test and remainder
author: Ben E
date: 2021-11-23
"""
import argparse
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    logger.info("RUNNING DATA SPLIT SECTION")
    logger.info(f"Fetching cleaned dataset {args.csv_clean}")
    df = pd.read_csv(args.csv_clean)
    base_path = os.path.dirname(args.csv_clean)

    logger.info("Splitting trainval and test")
    df_trainval, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    logger.info("Saving datasets to data folder")
    df_trainval.to_csv(args.csv_train, index=False)
    df_test.to_csv(args.csv_test, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")
    parser.add_argument("--csv_clean", type=str, help="Input csv to split")
    parser.add_argument(
        "--test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )
    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )
    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )
    parser.add_argument("--csv_train", type=str, help="Output csv to name training data", required=True)
    parser.add_argument("--csv_test", type=str, help="Output csv to name test data", required=True)
    args = parser.parse_args()
    go(args)
