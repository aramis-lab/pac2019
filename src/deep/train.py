import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import create_model
from structures.classification_utils import load_model, commandline_to_json
from structures.data_utils import Normalization, load_data, MRIDataset, generate_sampler

from pytorchtrainer import create_default_trainer
from pytorchtrainer.stop_condition import EarlyStopping
from pytorchtrainer.callback import ValidationCallback, SaveBestCheckpointCallback, SaveCheckpointCallback, CsvWriter, LoadCheckpointCallback
from pytorchtrainer.callback.checkpoint import default_best_filename
from pytorchtrainer.metric import TorchLoss


parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("data_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be subject_ID, gender, age and site.")
parser.add_argument("output_dir", type=str,
                    help="Path to save the outputs of the training.")
parser.add_argument("input_dir", type=str, nargs='+',
                    help="Path to input directories of the MRI.")
parser.add_argument("model", type=str,
                    help="Model used for classification.")

# Data Management
parser.add_argument("--sampler", "-s", default="random", type=str, choices=["random", "weighted"],
                    help="Sampler choice (random, or weighted for imbalanced datasets)")
parser.add_argument("--data_normalization", "-dn", default=None, type=str, choices=["Gaussian", "MinMax", None],
                    help="Type of data normalization to use for the input data.")
parser.add_argument("--output_normalization", "-on", default="", type=str, choices=["m", "mv", ""],
                    help="Type of data normalization to use for the output data.")
parser.add_argument("--blacklist", "-bl", default=None, type=str,
                    help="Path to the blacklist file. This excludes subjects from the dataloader.")
parser.add_argument("--file_extension", "-file_ext", type=str, nargs='+', default=None,
                    help="Extensions of the image file names. If None they are deduced from the list of input_dir")
parser.add_argument("--selection", type=str, default=None, choices=["old-" + str(i) for i in range(17, 90)] +
                                                                   ["young-" + str(i) for i in range(17, 90)],
                    help="Allow to choose among several datasets to train a specialized network.")

# Data split management
parser.add_argument("--n_splits", type=int, default=5,
                    help="If a value is given will load data of a k-fold CV")
parser.add_argument("--split", type=int, default=None, nargs='+',
                    help="Will load the list of folds wanted.")

# Training arguments
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--patience", type=int, default=10,
                    help="Waiting time for early stopping.")
parser.add_argument("--tolerance", type=float, default=0.05,
                    help="Tolerance value for the early stopping.")
parser.add_argument("--dropout", default=0.0, type=float,
                    help="Dropout rate for all dropout layers of a CNN.")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.01)
parser.add_argument("--weight_decay", "-wd", default=1e-4, type=float)
parser.add_argument("--loss", "-l", default="L1Loss", choices=["L1Loss", "MSELoss"], type=str,
                    help="Loss chosen for the training phase.\n"
                         "- L1Loss: Mean Absolute Error (used in Cole et al)\n"
                         "- MSELoss: Mean Squared Error (to better learn old people)")
parser.add_argument("--accumulation_steps", "-asteps", default=1, type=int,
                    help="Accumulates gradients in order to increase the size of the batch.")

# Output management
parser.add_argument("--evaluation_steps", "-esteps", default=100, type=int,
                    help="Fix the number of batches to use before computing validation and training performances.")

# Computational issues
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--num_threads', type=int, default=0,
                    help='Number of threads used.')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=4, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--training_evaluation", default='whole_set', type=str, choices=['whole_set', 'n_batches'],
                    help="Choose the way training evaluation is performed.")


def main(options):
    # Check if model is implemented
    import sys
    import models
    import inspect

    torch.set_num_threads(options.num_threads)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps not in [1, 0]:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    transformations = Normalization(options.data_normalization)

    text_file = open(os.path.join(options.output_dir, 'python_version.txt'), 'w')
    text_file.write('Version of python: %s \n' % sys.version)
    text_file.write('Version of pytorch: %s \n' % torch.__version__)
    text_file.close()

    if options.split is None:
        iterator = list(np.arange(options.n_splits))
    else:
        iterator = options.split

    for split in iterator:

        # Get the data.
        training_tsv, valid_tsv = load_data(options.data_path, split, options.n_splits, selection=options.selection)

        data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations,
                                normalization=options.output_normalization, list_file_extension=options.file_extension,
                                blacklist_file=options.blacklist)
        data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations,
                                normalization=options.output_normalization,
                                age_mean=data_train.age_mean, age_std=data_train.age_std,
                                list_file_extension=options.file_extension,
                                blacklist_file=options.blacklist, n_site=data_train.n_site)

        options.n_covars = len(data_train[0]["covars"])
        # add model input size
        options.input_size = data_train.shape

        commandline_to_json(options)

        train_sampler = generate_sampler(data_train, options.sampler)
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  sampler=train_sampler,
                                  num_workers=options.num_workers
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers
                                  )

        # Initialize the model
        print('Initialization of the model')
        model = create_model(options)

        # Define criterion and optimizer
        criterion = eval('torch.nn.' + options.loss)()
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate, weight_decay=options.weight_decay)
        best_model_dir = os.path.join(options.output_dir, 'fold_' + str(split), 'best_model_dir')

        print('Beginning the training task')

        def prepare_batch(batch, device=None, non_blocking=False, dtype=None):
            x = batch['image'].to(device=device, dtype=dtype, non_blocking=non_blocking)
            y = batch['label'].to(device=device, dtype=dtype, non_blocking=non_blocking)
            batch['covars'] = batch['covars'].to(device=device, dtype=dtype, non_blocking=non_blocking)

            return x, y, {'covars': batch['covars']}

        validation_callback = ValidationCallback(valid_loader, TorchLoss(criterion))

        # instantiate trainer
        if "inception3D_main" in type(model).__name__.lower():  # TODO: create trainer from within the model.
            # multi-output model
            trainer = create_default_trainer(model, optimizer, criterion,
                                             prepare_batch_function=prepare_batch,
                                             loss_transform_function=lambda criterion, y_preds, y: criterion(y_preds[0], y) + 0.3*criterion(y_preds[1], y) + 0.3*criterion(y_preds[2], y),
                                             output_transform=lambda x, y, y_pred, loss: (x, y, y_pred[0], loss.item()),
                                             device='cuda' if options.gpu else 'cpu')
        else:
            trainer = create_default_trainer(model, optimizer, criterion,
                                             prepare_batch_function=prepare_batch,
                                             device='cuda' if options.gpu else 'cpu')

        csv_writer = CsvWriter(extra_header=[validation_callback.state_attribute_name],
                               extra_data_function=lambda state: [state.get(validation_callback.state_attribute_name)],
                               save_directory=os.path.join(options.output_dir, 'fold_' + str(split), 'log_dir'),
                               filename="training.tsv", delimiter='\t')

        # post iteration callback
        trainer.register_post_iteration_callback(validation_callback, frequency=options.evaluation_steps)
        trainer.register_post_iteration_callback(csv_writer)

        # post epoch callback
        trainer.register_post_epoch_callback(validation_callback)
        trainer.register_post_epoch_callback(csv_writer)
        trainer.register_post_epoch_callback(SaveCheckpointCallback(save_directory=os.path.join(options.output_dir, 'fold_' + str(split), 'checkpoint')))
        trainer.register_post_epoch_callback(SaveBestCheckpointCallback(validation_callback.state_attribute_name, save_directory=best_model_dir, saves_to_keep=1))

        trainer.add_progressbar_metric("validation loss %.4f", [validation_callback])

        trainer.train(train_dataloader=train_loader, max_epochs=options.epochs,
                      stop_condition=EarlyStopping(patience=options.patience,
                                                   metric=lambda state: getattr(state, validation_callback.state_attribute_name))
                      )

        # Perform final evaluation

        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        # load best model
        trainer.load(save_directory=best_model_dir, filename=default_best_filename)

        def csv_writer_extra_data_function(x, y, y_pred, loss, batch):
            assert len(y) == len(y_pred) == len(batch['subject_ID'])

            res = []
            for i in range(len(batch['subject_ID'])):
                res.append([batch['subject_ID'][i], y[i].item(), y_pred[i].item()])

            return res

        evaluation_path = os.path.join(options.output_dir, 'fold_' + str(split), 'performances_train')
        loss_train = trainer.evaluate(train_loader, TorchLoss(criterion),
                                      csv_writer=CsvWriter(save_directory=evaluation_path, filename="train_subject_level_result.tsv", extra_header=["participant_id", "true_age", "predicted_age"]),
                                      csv_writer_extra_data_function=csv_writer_extra_data_function)
        loss_valid = trainer.evaluate(valid_loader, TorchLoss(criterion),
                                      csv_writer=CsvWriter(save_directory=evaluation_path, filename="valid_subject_level_result.tsv", extra_header=["participant_id", "true_age", "predicted_age"]),
                                      csv_writer_extra_data_function=csv_writer_extra_data_function)

        with open(os.path.join(evaluation_path, 'evaluation_loss.txt'), 'w') as writer:
            writer.write('Mean loss on training set: %f \n' % (loss_train / len(data_train) * options.batch_size))
            writer.write('Mean loss on validation set: %f \n' % (loss_valid / len(data_valid) * options.batch_size))


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    commandline_to_json(options)
    main(options)
