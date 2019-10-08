"""
Generates tsv files containing the ID of the sessions for a single split.
The two datasets have non-significant differences in terms of distribution for age, sex and site.
"""
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ttest_ind


sex_dict = {'m': 0, 'f': 1}


def chi2(x_test, x_train):
    # Look for chi2 computation
    p_expectedF = np.sum(x_train) / len(x_train)
    p_expectedM = 1 - p_expectedF

    expectedF = p_expectedF * len(x_test)
    expectedM = p_expectedM * len(x_test)
    observedF = np.sum(x_test)
    observedM = len(x_test) - np.sum(x_test)

    T = (expectedF - observedF) ** 2 / expectedF + (expectedM - observedM) ** 2 / expectedM

    return T


if __name__ == "__main__":

    import argparse
    import pandas as pd
    import os
    from os import path
    import numpy as np

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("merged_tsv", type=str,
                        help="Path to the tsv file to split.")

    # Data management
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Size of the test set.")
    parser.add_argument("--subset_name", type=str, default="validation",
                        help="Name of the subset that is complementary to train.")

    # Thresholds for the selection
    parser.add_argument("--p_val_threshold", "-p", default=0.90, type=float,
                        help="The threshold used for the T-test.")
    parser.add_argument("--t_val_threshold", "-t", default=0.0642, type=float,
                        help="The threshold used for the chi2 test.")

    args = parser.parse_args()

    # Read files
    merged_df = pd.read_csv(args.merged_tsv, sep='\t')

    results_path = os.path.abspath(os.path.join(args.merged_tsv, os.pardir))
    filename = path.splitext(path.basename(args.merged_tsv))[0]

    train_path = path.join(results_path, filename + '_train.tsv')
    test_path = path.join(results_path, filename + '_' + args.subset_name + '.tsv')

    flag_selection = True

    sex = list(merged_df.gender.values)
    site = list(merged_df.site.values)
    age = list(merged_df.age.values)

    train_index, test_index = None, None

    while flag_selection:

        splits = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size)

        for train_index, test_index in splits.split(np.zeros(len(site)), site):

            age_test = [float(age[idx]) for idx in test_index]
            age_train = [float(age[idx]) for idx in train_index]

            sex_test = [sex_dict[sex[idx]] for idx in test_index]
            sex_train = [sex_dict[sex[idx]] for idx in train_index]

            t_age, p_age = ttest_ind(age_test, age_train)
            T_sex = chi2(sex_test, sex_train)

            print(p_age, T_sex)
            if p_age > args.p_val_threshold and T_sex < args.t_val_threshold:
                flag_selection = False

            test_df = merged_df.iloc[test_index]
            train_df = merged_df.iloc[train_index]

            train_df.to_csv(train_path, sep='\t', index=False)
            test_df.to_csv(test_path, sep='\t', index=False)
