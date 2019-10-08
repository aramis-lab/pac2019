import torch
import pandas as pd
import numpy as np
from os import path
from copy import deepcopy
from warnings import warn
from torch.utils.data import Dataset, sampler


class MRIDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, list_img_dir, data_file, blacklist_file=None, transform=None, age_mean=None, age_std=None,
                 normalization='', list_file_extension=None, n_site=None):
        """
        Args:
            list_img_dir (list): list of directories containing all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.list_img_dir = list_img_dir
        self.transform = transform
        self.sex_dict = {'m': 0, 'f': 1}
        self.labeled = True

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if blacklist_file is not None:
            # exclude black listed subjects
            with open(blacklist_file) as f:
                blacklist = f.read().splitlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            blacklist = [x.strip() for x in blacklist]

            for b in blacklist:
                index = self.df[self.df.subject_ID == b].index
                self.df.drop(index, inplace=True, errors='ignore')

        if ('gender' not in list(self.df.columns.values)) or \
           ('subject_ID' not in list(self.df.columns.values) or ('site' not in list(self.df.columns.values))):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['subject_ID', 'age', 'gender', 'site]")
        elif 'age' not in list(self.df.columns.values):
            self.labeled = False

        if age_mean is None:
            age_mean = np.mean(self.df.age.values)
        if age_std is None:
            age_std = np.std(self.df.age.values)
        self.age_mean = age_mean
        self.age_std = age_std
        self.normalization = normalization

        # If only one extension is give, apply it for all directories

        if list_file_extension is None:
            list_file_extension = list()
            for input_dir in list_img_dir:
                basename = path.basename(input_dir)
                list_file_extension.append("_" + basename + ".pt")
        if isinstance(list_file_extension, str):
            list_file_extension = [list_file_extension] * len(list_img_dir)
        else:
            if len(list_file_extension) != len(list_img_dir):
                raise ValueError("The list of the file extensions (len = %i) must have the same length than the list of"
                                 "image directories (len = %i)." % (len(list_file_extension), len(list_img_dir)))
            else:
                list_file_extension = list_file_extension

        self.list_file_extension = list_file_extension

        true_n_site = np.max(self.df.site) - np.min(self.df.site) + 1
        if n_site is None:
            self.n_site = true_n_site
        elif true_n_site > n_site:
            warn("The number of site asked (%i) is too small to handle the data. The number of sites will be %i."
                 % (n_site, true_n_site))
            self.n_site = true_n_site
        else:
            self.n_site = n_site

        self.shape = self[0]['image'].shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        img_name = data['subject_ID']
        sex = data['gender']
        site = data['site']

        if self.labeled:
            age = data['age']
            label = data['age']
            if 'm' in self.normalization:
                label -= self.age_mean
            if 'v' in self.normalization:
                label /= self.age_std

        image = torch.Tensor()
        for img_dir, file_extension in zip(self.list_img_dir, self.list_file_extension):
            image_path = path.join(img_dir, img_name + file_extension)
            sub_image = torch.load(image_path)
            image = torch.cat([image, sub_image])

        if self.transform:
            image = self.transform(image)

        discreete_site = torch.zeros(self.n_site)
        discreete_site[site] = 1
        sample = {'image': image, 'subject_ID': img_name, 'sex': sex, 'site': site,
                  # 'covars': torch.FloatTensor([self.sex_dict[sex], site])}
                  'covars': torch.cat([torch.FloatTensor([self.sex_dict[sex]]), discreete_site])}

        if self.labeled:
            sample['age'] = np.float32(age)
            sample['label'] = np.float32(label)

        return sample

    def age_from_label(self, label):
        return label * self.age_std + self.age_mean


class GaussianSmoothing3d(object):

    def __init__(self, sigma, filter_size=5, cuda=False):
        from scipy.ndimage.filters import gaussian_filter
        import torch.nn as nn
        import numpy as np

        if filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer")
        input_np = np.zeros((filter_size, filter_size, filter_size))
        kernel_np = gaussian_filter(input_np, sigma)
        self.conv = nn.Conv3d(1, 1, filter_size, padding=filter_size // 2)
        self.conv.weight.data = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).float()
        self.conv.weight.requires_grad = False
        self.conv.bias.data = torch.Tensor([0.0])
        self.conv.bias.requires_grad = False

        if cuda:
            self.conv.cuda()

    def __call__(self, image):
        unsqueezed_image = image
        while len(image.shape) < 5:
            unsqueezed_image = torch.unsqueeze(unsqueezed_image, 0)
        smoothed_image = self.conv(unsqueezed_image)

        return smoothed_image


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class GaussianNormalization(object):
    """Applies Gaussian normalization to a tensor"""

    def __call__(self, image):
        return (image - image.mean()) / image.std()


class Normalization(object):
    def __init__(self, normalization):
        if normalization is not None:
            self.transform = eval(normalization + 'Normalization')()
        else:
            self.transform = None

    def __call__(self, image):
        if self.transform is not None:
            return self.transform(image)

        return image


class CenterSize3d(object):

    def __init__(self, output_size):
        """
        :param output_size: (list) the desired size of the output
        """
        output_size = list(output_size)
        if len(output_size) == 4:
            self.output_size = output_size
        elif len(output_size) == 3:
            self.output_size = [1] + output_size
        else:
            raise ValueError("The argument must be of length 3 or 4.")

    def __call__(self, tensor):

        import torch.nn.functional as F
        from time import time

        t0 = time()

        size = tensor.shape
        updated_tensor = tensor
        for i in range(1, len(size)):
            input_size = size[i]
            output_size = int(self.output_size[i])
            if input_size > output_size:
                crop_index = int(abs((input_size - output_size) / 2))
                indices = torch.arange(0, output_size)
                indices += crop_index
                updated_tensor = torch.index_select(updated_tensor, i, indices)
                print("Cropping time", time() - t0)
            else:
                padding_indices = [0] * 6
                padding_indices[2 * i - 2] = int(abs((input_size - output_size) / 2))
                if (input_size - output_size) % 2 == 0:
                    padding_indices[2 * i - 1] = int(abs((input_size - output_size) / 2))
                else:
                    padding_indices[2 * i - 1] = int(abs((input_size - output_size) / 2)) + 1
                updated_tensor = F.pad(updated_tensor, padding_indices, 'constant', 0)
                print("Padding time", time() - t0)

        return updated_tensor.clone()


class RandomScale(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, initial_shape):
        self.center = CenterSize3d(initial_shape)

    def __call__(self, image):
        from scipy.ndimage import zoom
        import random

        coefficient = random.uniform(0.8, 1.2)
        # This part is far too slow, may be improved with torch.nn.functional.interpolate
        zoomed_tensor = torch.from_numpy(zoom(image, coefficient))

        return self.center(zoomed_tensor)


class RandomNoising(object):
    """Applies a random zoom to a tensor"""
    def __call__(self, image):
        import random

        sigma = random.uniform(0, 0.5)
        dist = torch.distributions.normal.Normal(0, sigma)
        return image + dist.sample(image.shape)


def load_data(train_val_path, split, n_splits=5, train_mode=True, selection=None):

    if train_mode:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split) + '_train.tsv')
        valid_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split) + '_validation.tsv')
    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split) + '.tsv')
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split) + '.tsv')

    print("Train", train_path)
    print("Valid", valid_path)

    train_df = pd.read_csv(train_path, sep='\t')
    valid_df = pd.read_csv(valid_path, sep='\t')

    if selection is not None:
        age_limit = int(selection[-2::])
        if "old" in selection:
            train_df = train_df[train_df.age > age_limit]
            valid_df = valid_df[valid_df.age > age_limit]
        elif "young" in selection:
            train_df = train_df[train_df.age < age_limit]
            valid_df = valid_df[valid_df.age < age_limit]
        else:
            raise ValueError("The selection value %s cannot be handled" % selection)

    return train_df, valid_df


def sort_predicted(df, model_path, split=0, set="train", keep_true=True, selection='loss'):
    output_df = deepcopy(df)
    result_path = path.join(model_path, 'fold_' + str(split), 'performances', 'best_' + selection,
                            set + '_subject_level_result.tsv')
    result_df = pd.read_csv(result_path, sep='\t')
    result_df.set_index(['participant_id', 'session_id'], inplace=True)

    for i in df.index.values:
        participant_id = df.loc[i, 'participant_id']
        session_id = df.loc[i, 'session_id']
        true_label = result_df.loc[(participant_id, session_id), 'true_label']
        predicted_label = result_df.loc[(participant_id, session_id), 'predicted_label']
        if (true_label == predicted_label) != keep_true:
            output_df.drop(i, inplace=True)

    output_df.reset_index(inplace=True, drop=True)
    return output_df


def generate_sampler(dataset, sampler_option='random', step=1):
    """
    Returns sampler according to the wanted options

    :param dataset: (MRIDataset) the dataset to sample from
    :param sampler_option: (str) choice of sampler
    :param step: (int) step to discretize ages and give a weight per class
    :return: (Sampler)
    """

    df = dataset.df
    min_age = np.min(df.age)
    max_age = np.max(df.age)

    if (max_age - min_age) % step == 0:
        max_age += step

    bins = np.arange(min_age, max_age, step)
    count = np.zeros(len(bins))
    for idx in df.index:
        age = df.loc[idx, "age"]
        key = np.argmax(np.logical_and(age - step < bins, age >= bins)).astype(int)
        count[key] += 1

    # weight_per_class = (1 / np.array(count)) if count.any() != 0 else 0.
    weight_per_class = np.zeros_like(count).astype(float)
    np.divide(1., count, out=weight_per_class, where=count != 0)
    weights = [0] * len(df)

    for idx, age in enumerate(df.age.values):
        key = np.argmax(np.logical_and(age - 5 <= bins, age > bins)).astype(int)
        weights[idx] = weight_per_class[key]

    weights = torch.FloatTensor(weights)

    if sampler_option == 'random':
        s = sampler.RandomSampler(dataset, replacement=False)
    elif sampler_option == 'weighted':
        s = sampler.WeightedRandomSampler(weights, len(weights))
    else:
        raise NotImplementedError("The option %s for sampler is not implemented" % sampler_option)

    return s
