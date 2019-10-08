"""
This code was used to generate the prediction of the specialized 6-layer CNNs
See paper at section 3.2.4 Model 4: Specialized 6-layer CNNs for younger and older subjects
Two models must have been trained, one on the whole data (generalist model) and one on all subjects
older than 40 (specialist model).
"""

import pandas as pd
import numpy as np
from os import path
from copy import deepcopy
import argparse
import os

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("generalist_model_path", type=str,
                    help="Path to the model trained on the whole data.")
parser.add_argument("specialist_model_path", type=str,
                    help="Path to the model trained on the eldest subjects.")
parser.add_argument("fusion_path", type=str,
                    help="Path to the output directory containing the fused results.")
parser.add_argument("--split", type=int, default=None, nargs='+',
                    help="Will process the list of folds wanted. Default behaviour will process all available folds "
                         "in the generalist model directory.")


def mean_fuse_results(gen_df, spe_df, age_limit=40):
    fusion_df = deepcopy(gen_df)
    for participant in gen_df.index.values:
        gen_age = gen_df.loc[participant, 'predicted_age']
        if gen_age > age_limit:
            spe_age = spe_df.loc[participant, 'predicted_age']
            fusion_df.loc[participant, 'predicted_age'] = (spe_age + gen_age) / 2 
            
    return fusion_df


def evaluate_spearman(df):
    from scipy.stats import spearmanr

    simple_correlation, _ = spearmanr(df.true_age, df.predicted_age)
    difference_correlation, _ = spearmanr(df.true_age, df.predicted_age.astype(float) - df.true_age.astype(float))

    return simple_correlation, difference_correlation


def evaluate_age_limits(gen_df, spe_df):
    age_limits = np.arange(30, 89)
    results_df = pd.DataFrame(index=age_limits, columns=['MAE_mean', 'Spearman_mean'])
    for age_limit in age_limits:
        mean_df = mean_fuse_results(gen_df, spe_df, age_limit=age_limit)
        
        MAE = np.mean(np.abs(mean_df.predicted_age - mean_df.true_age))
        _, diff_Spearman = evaluate_spearman(mean_df)
        results_df.loc[age_limit, 'MAE_mean'] = MAE
        results_df.loc[age_limit, 'Spearman_mean'] = diff_Spearman
        
    return results_df


def main(options):

    if options.split is None:

        split_dirs = [split_dir for split_dir in os.listdir(options.generalist_model_path)
                      if split_dir.split('_')[0] == "fold"]
        options.split = sorted([int(split_dir.split('_')[1]) for split_dir in split_dirs])

    for fold in options.split:
        print(fold, type(fold))
        gen_tv_df = pd.read_csv(path.join(options.generalist_model_path, 'fold_%i' % fold, 'performances_train',
                                          'best_loss', 'valid_subject_level_result.tsv'), sep='\t')
        spe_tv_df = pd.read_csv(path.join(options.specialist_model_path, 'fold_%i' % fold, 'performances_train',
                                          'best_loss', 'valid_subject_level_result.tsv'), sep='\t')
        gen_tv_df.set_index('participant_id', inplace=True)
        spe_tv_df.set_index('participant_id', inplace=True)
        results_df = evaluate_age_limits(gen_tv_df, spe_tv_df)
        MAE_mean = results_df.MAE_mean.astype(float)
        age_limit = MAE_mean.idxmin()
        print("Min MAE %.2f for age %i" %(MAE_mean.min(), age_limit))

        gen_v_df = pd.read_csv(path.join(options.generalist_model_path, 'fold_%i' % fold, 'performances_val',
                                         'best_loss', 'valid_subject_level_result.tsv'), sep='\t')
        spe_v_df = pd.read_csv(path.join(options.specialist_model_path, 'fold_%i' % fold, 'performances_val',
                                         'best_loss', 'valid_subject_level_result.tsv'), sep='\t')
        gen_v_df.set_index('participant_id', inplace=True)
        spe_v_df.set_index('participant_id', inplace=True)

        fusion_df = mean_fuse_results(gen_v_df, spe_v_df, age_limit=age_limit)
        MAE = np.mean(np.abs(fusion_df.predicted_age - fusion_df.true_age))
        _, r = evaluate_spearman(fusion_df)
        print("Fusion, MAE: %.2f, r: %.2f" % (MAE, r))
        MAE = np.mean(np.abs(gen_v_df.predicted_age - gen_v_df.true_age))
        _, r = evaluate_spearman(gen_v_df)
        print("Generalist, MAE: %.2f, r: %2f" % (MAE, r))

        results_path = path.join(options.fusion_path, 'fold_%i' % fold, 'performances_val', 'best_loss')
        if not path.exists(results_path):
            os.makedirs(results_path)
        fusion_df.to_csv(path.join(results_path, 'valid_subject_level_result.tsv'), sep='\t')


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
