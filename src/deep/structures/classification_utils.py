from __future__ import print_function

import os
import torch
import numpy as np
import shutil
import pandas as pd


def test(model, dataloader, use_cuda, criterion, full_return=False, log_path=None):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
    :return:
        balanced accuracy of the model (float)
        total loss on the dataloader
    """
    model.eval()

    columns = ["participant_id", "true_age", "predicted_age"]
    results_df = pd.DataFrame(columns=columns)

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs, labels = data['image'].cuda(), data['label'].cuda()
            data['covars'] = data['covars'].cuda()
        else:
            inputs, labels = data['image'], data['label']
            data['covars'] = data['covars'].cpu()
        age = data['age']
        outputs = model(inputs, covars=data['covars'])
        loss = criterion(outputs, labels.unsqueeze(1))
        total_loss += loss.item()
        predicted = outputs.data.squeeze(1)

        # Generate detailed DataFrame
        for idx, sub in enumerate(data['subject_ID']):
            prediction = predicted[idx]
            if 'v' in dataloader.dataset.normalization:
                prediction *= dataloader.dataset.age_std
            if 'm' in dataloader.dataset.normalization:
                prediction += dataloader.dataset.age_mean

            row = [sub, age[idx].item(), prediction.item()]
            row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
            results_df = pd.concat([results_df, row_df])

        del inputs, outputs, labels

    results_df.reset_index(inplace=True, drop=True)
    mae = np.mean(np.abs(results_df.predicted_age.astype(float) - results_df.true_age.astype(float)))

    model.train()

    if log_path is not None:
        if os.path.isfile(log_path) and not os.path.exists(os.path.dirname(log_path)):
            # if file is given
            os.makedirs(os.path.dirname(log_path))
        elif os.path.isdir(log_path) and not os.path.exists(log_path):
            # if directory is given
            os.makedirs(log_path)
            log_path = os.path.join(log_path, "result.tsv")

        results_df.to_csv(log_path, sep='\t', index=False)

    if full_return:
        return total_loss, mae, results_df

    return total_loss, mae


def unsupervised_test(model, dataloader, use_cuda):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :return:
        results_df (DataFrame) predicted ages of the set
    """
    model.eval()

    columns = ["participant_id", "predicted_age"]
    results_df = pd.DataFrame(columns=columns)

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
            data['covars'] = data['covars'].cuda()
        else:
            inputs = data['image']
            data['covars'] = data['covars'].cpu()
        outputs = model(inputs, covars=data['covars'])
        predicted = outputs.data.squeeze(1)

        # Generate detailed DataFrame
        for idx, sub in enumerate(data['subject_ID']):
            prediction = predicted[idx]
            if 'v' in dataloader.dataset.normalization:
                prediction *= dataloader.dataset.age_std
            if 'm' in dataloader.dataset.normalization:
                prediction += dataloader.dataset.age_mean

            row = [sub, prediction.item()]
            row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
            results_df = pd.concat([results_df, row_df])

        del inputs, outputs

    results_df.reset_index(inplace=True, drop=True)
    model.train()

    return results_df


def save_checkpoint(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_mae='best_mae', best_loss='best_loss'):

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if accuracy_is_best:
        best_accuracy_path = os.path.join(checkpoint_dir, best_mae)
        if not os.path.exists(best_accuracy_path):
            os.makedirs(best_accuracy_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(best_accuracy_path, 'model_best.pth.tar'))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))


def load_model(model, checkpoint_dir, filename='model_best.pth.tar'):
    from copy import deepcopy

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    best_model.load_state_dict(param_dict['model'])
    return best_model, param_dict['epoch']


def commandline_to_json(commandline, log_dir=None):
    """
    Write the python argparse object into a json file.

    :param commandline: the output of `parser.parse_known_args()`
    :param log_dir: (str) where the commandline is going to be saved.
    """
    import json
    from copy import deepcopy

    commandline_arg_dict = deepcopy(vars(commandline))
    # Array cannot be serialized
    if 'flattened_shape' in commandline_arg_dict.keys():
        commandline_arg_dict.pop('flattened_shape')

    # if train_from_stop_point, do not delete the folders
    if log_dir is None:
        log_dir = commandline_arg_dict['output_dir']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save to json file
    json = json.dumps(commandline_arg_dict)
    print("Path of json file:", os.path.join(log_dir, "commandline.json"))
    f = open(os.path.join(log_dir, "commandline.json"), "w")
    f.write(json)
    f.close()


def read_json(options, json_path=None):
    """
    Read a json file to update python argparse Namespace.

    :param options: (argparse.Namespace) options of the model
    :return: options (args.Namespace) options of the model updated
    """
    import json
    from os import path

    if json_path is None:
        json_path = path.join(options.model_path, 'commandline.json')

    with open(json_path, "r") as f:
        json_data = json.load(f)

    for key, item in json_data.items():
        # We do not change computational options
        if key not in ['gpu', 'num_workers', 'num_threads', 'output_dir']:
            setattr(options, key, item)

    if hasattr(options, 'convolutions'):
        # Flattened shape
        from structures.model import initial_shape

        n_conv = len(options.convolutions)
        flattened_shape = np.ceil(np.array(initial_shape) / 2 ** n_conv)
        setattr(options, "flattened_shape", flattened_shape)

    if not hasattr(options, 'loss'):
        options.loss = "L1Loss"
    if not hasattr(options, 'output_normalization'):
        options.output_normalization = ''
    if not hasattr(options, 'dropout'):
        options.dropout = 0.0
    if not hasattr(options, 'n_covars'):
        options.n_covars = 18
    if not hasattr(options, 'input_size'):
        options.input_size = np.array([1, 121, 145, 121])
    elif isinstance(options.input_size, list):
        options.input_size = np.array(options.input_size)
    if not hasattr(options, 'file_extension'):
        options.file_extension = None
    if not hasattr(options, 'selection'):
        options.selection = None
    if not isinstance(options.input_dir, list):
        options.input_dir = [options.input_dir]

    return options


def adjust_learning_rate(optimizer, lr):
    print(lr)
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)


############################
#          Debug           #
############################
def memReport():
    import gc

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
