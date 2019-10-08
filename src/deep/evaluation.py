import argparse
from torch.utils.data import DataLoader
from os import path
import numpy as np
import pandas as pd
import torch
import os

from structures.classification_utils import test, load_model, read_json
from structures.data_utils import MRIDataset, Normalization, load_data

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("output_dir", type=str,
                    help="Path where the outputs of the training were saved.")

# Data management
parser.add_argument("--train_mode", action="store_true", default=False,
                    help="Loads data used for the training phase")
parser.add_argument("--split_list", type=int, nargs="+", default=None,
                    help="List of splits to evaluate. If None all detected folds will be evaluated.")

# Computational issues
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--num_threads', type=int, default=0,
                    help='Number of threads used.')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def evaluate_Spearman(df):
    from scipy.stats import spearmanr

    simple_correlation, _ = spearmanr(df.true_age, df.predicted_age)
    difference_correlation, _ = spearmanr(df.true_age, df.predicted_age.astype(float) - df.true_age.astype(float))

    return simple_correlation, difference_correlation


def main(options):

    # Check if model is implemented
    import sys
    import models
    import inspect

    torch.set_num_threads(options.num_threads)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    valid_transformations = Normalization(options.data_normalization)

    text_file = open(path.join(options.output_dir, 'python_version.txt'), 'w')
    text_file.write('Version of python: %s \n' % sys.version)
    text_file.write('Version of pytorch: %s \n' % torch.__version__)
    text_file.close()

    # Initialize the model
    print('Initialization of the model')
    model = models.create_model(options)
    print(model)
    criterion = eval("torch.nn." + options.loss)()

    # Choose folds for evaluation
    if options.split_list is None:
        folds_list = os.listdir(options.output_dir)
        folds_list = [fold for fold in folds_list if fold[:5:] == "fold_"]
    else:
        folds_list = ["fold_" + str(fold) for fold in options.split_list]

    for fold_name in folds_list:
        fold = fold_name[5:]

        # Computing the mean and std age to have the same normalization as in the training phase
        true_training_tsv, _ = load_data(options.data_path, fold, options.n_splits, train_mode=True,
                                         selection=options.selection)
        true_data_train = MRIDataset(options.input_dir, true_training_tsv, transform=valid_transformations,
                                     normalization=options.output_normalization,
                                     list_file_extension=options.file_extension, n_site=options.n_covars - 1)
        age_mean = true_data_train.age_mean
        age_std = true_data_train.age_std
        
        # Get the data.
        training_tsv, valid_tsv = load_data(options.data_path, fold, options.n_splits, train_mode=options.train_mode)

        data_train = MRIDataset(options.input_dir, training_tsv, transform=valid_transformations,
                                normalization=options.output_normalization, list_file_extension=options.file_extension,
                                n_site=options.n_covars - 1, age_mean=age_mean, age_std=age_std)
        data_valid = MRIDataset(options.input_dir, valid_tsv, transform=valid_transformations,
                                normalization=options.output_normalization, list_file_extension=options.file_extension,
                                age_mean=age_mean, age_std=age_std,
                                n_site=data_train.n_site)

        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        best_model_dir = path.join(options.output_dir, 'fold_' + str(fold), 'best_model_dir')
        if options.train_mode:
            evaluation_path = path.join(options.output_dir, 'fold_' + str(fold), 'performances_train')
        else:
            evaluation_path = path.join(options.output_dir, 'fold_' + str(fold), 'performances_val')
        for selection in ['loss', 'mae']:
            model_dir = os.path.join(best_model_dir, 'best_' + selection)
            folder_name = 'best_' + selection

            best_model, best_epoch = load_model(model, model_dir, filename='model_best.pth.tar')

            loss_train, mae_train, train_df = test(best_model, train_loader, options.gpu, criterion, full_return=True)
            loss_valid, mae_valid, valid_df = test(best_model, valid_loader, options.gpu, criterion, full_return=True)
            simplespearman_train, diffspearman_train = evaluate_Spearman(train_df)
            simplespearman_valid, diffspearman_valid = evaluate_Spearman(valid_df)

            if not path.exists(path.join(evaluation_path, folder_name)):
                os.makedirs(path.join(evaluation_path, folder_name))

            text_file = open(path.join(evaluation_path, 'evaluation_' + selection + '.txt'), 'w')
            text_file.write('Best epoch: %i \n' % best_epoch)
            text_file.write('Mean loss on training set: %f \n' % (loss_train / len(data_train) * options.batch_size))
            text_file.write('MAE on training set: %f \n' % mae_train)
            text_file.write('Simple Spearman correlation on training set: %f \n' % simplespearman_train)
            text_file.write('Difference Spearman correlation on training set: %f \n' % diffspearman_train)
            text_file.write('Mean loss on validation set: %f \n' % (loss_valid / len(data_valid) * options.batch_size))
            text_file.write('MAE on validation set: %f \n' % mae_valid)
            text_file.write('Simple Spearman correlation on validation set: %f \n' % simplespearman_valid)
            text_file.write('Difference Spearman correlation on validation set: %f \n' % diffspearman_valid)
            text_file.close()

            train_df.to_csv(path.join(evaluation_path, folder_name, 'train_subject_level_result.tsv'), sep='\t', index=False)
            valid_df.to_csv(path.join(evaluation_path, folder_name, 'valid_subject_level_result.tsv'), sep='\t', index=False)

        # Graphic part
        if options.graphics:
            import matplotlib.pyplot as plt

            plt.switch_backend('agg')

            training_df = pd.read_csv(path.join(options.output_dir, 'fold_' + str(fold),
                                                'log_dir', 'training.tsv'), sep='\t')
            epochs = training_df.epoch.values
            iterations = training_df.iteration.values
            iterations = iterations / np.max(iterations)
            x = epochs + iterations

            plt.title('Fold ' + str(fold))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(x, training_df.mean_loss_train.values, color='orange', label='training')
            plt.plot(x, training_df.mean_loss_valid.values, color='blue', label='validation')
            plt.legend()
            plt.savefig(path.join(evaluation_path, 'loss.png'))
            plt.close()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    options = read_json(options, path.join(options.output_dir, 'commandline.json'))
    main(options)
