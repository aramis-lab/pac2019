"""
Generates plots to compare the different folds created by kfold_split.py .
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from os import path
import numpy as np
import shutil
import os

sex_dict = {'m': 0, 'f': 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("merged_tsv", type=str,
                        help="Path to the file obtained by the command clinica iotools merge-tsv.")
    parser.add_argument("formatted_data_path", type=str,
                        help="Path to the folder containing formatted data.")

    # Modality selection
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Define the number of subjects to put in test set."
                             "If 0, there is no training set and the whole dataset is considered as a test set.")
    parser.add_argument("--subset_name", type=str, default="validation",
                        help="Name of the subset that is complementary to train.")

    args = parser.parse_args()

    # Read files
    merged_df = pd.read_csv(args.merged_tsv, sep=',')
    results_path = path.join(args.formatted_data_path, 'QC_split-' + str(args.n_splits))
    if path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    train_path = path.join(args.formatted_data_path, 'train_splits-' + str(args.n_splits))
    if not path.exists(train_path):
        raise ValueError('Train path %s does not exist' % train_path)

    test_path = path.join(args.formatted_data_path, args.subset_name + '_splits-' + str(args.n_splits))
    if not path.exists(test_path):
        raise ValueError('Test path %s does not exist' % test_path)

    for split in range(args.n_splits):
        train_split_df = pd.read_csv(path.join(train_path, 'split-' + str(split) + '.tsv'), sep='\t')
        test_split_df = pd.read_csv(path.join(test_path, 'split-' + str(split) + '.tsv'), sep='\t')

        plt.figure()
        plt.title("Age distribution")
        bins = np.arange(10, 100, 10)
        plt.hist(train_split_df.age.values, bins=bins, label='train', alpha=0.6)
        plt.hist(test_split_df.age.values, bins=bins, label=args.subset_name, alpha=0.6)
        plt.xlabel("Age")
        plt.ylabel("Number of subjects")
        plt.legend()
        plt.savefig(path.join(results_path, "ageDist_split" + str(split)))

        plt.figure()
        plt.title("Sex distribution")
        sex_train = list(train_split_df.gender.values)
        sex_train = [sex_dict[sex] for sex in sex_train]
        sex_test = list(test_split_df.gender.values)
        sex_test = [sex_dict[sex] for sex in sex_test]
        plt.hist(sex_train, bins=2, label="train", alpha=0.6)
        plt.hist(sex_test, bins=2, label=args.subset_name, alpha=0.6)
        plt.xlabel("Sex")
        plt.ylabel("Number of subjects")
        plt.legend()
        plt.savefig(path.join(results_path, "sexDist_split" + str(split)))

        plt.figure()
        plt.title("Site distribution")
        site_train = list(train_split_df.site.values)
        site_test = list(test_split_df.site.values)
        plt.hist(site_train, bins=np.max(site_train), label="train", alpha=0.6)
        plt.hist(site_test, bins=np.max(site_train), label=args.subset_name, alpha=0.6)
        plt.xlabel("Site")
        plt.ylabel("Number of subjects")
        plt.legend()
        plt.savefig(path.join(results_path, "siteDist_split" + str(split)))
