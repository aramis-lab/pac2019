"""File with the preprocessing tools."""
import os
import numpy as np
import nibabel as nib

import pandas as pd
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel


# Change this path
path = ''  # folder containing the gray-matter maps

# Folders with the resulting data
output_data = 'Data/'
output_kernels = 'Kernels/'
output_target = 'Target/'

# List of all the NifTI files
nifti_images = [file for file in os.listdir(path) if file.endswith('.nii.gz')]

# Convert each NifTI into a numpy.ndarray
for file in nifti_images:
    img = nib.load(os.path.join(path, file))
    img_data = img.get_fdata()
    np.save(os.path.join(output_data, file.split('_')[0]), img_data)

# Get the subject IDs
subjects = []
listdir = os.listdir(output_data)
listdir = [x for x in listdir if not x.startswith('.')]
n_samples = len(listdir)

# Compute the kernels using batches to reduce the memory usage
batches = np.array_split(np.arange(len(listdir)), 20)

lin_kernel = np.empty((n_samples, n_samples))
euclidean_norm = np.empty((n_samples, n_samples))

for batch_i in tqdm(batches):
    data_i = []
    for i in batch_i:
        data_i.append(np.load(output_data + listdir[i]).ravel())
        subjects.append(listdir[i].split('.')[0])
    data_i = np.asarray(data_i)

    for batch_j in batches:
        data_j = []
        for j in batch_j:
            data_j.append(np.load(output_data + listdir[j]).ravel())
        data_j = np.asarray(data_j)

        # Compute the kernels
        euclidean_norm[batch_i[0]:batch_i[-1] + 1,
                       batch_j[0]:batch_j[-1] + 1] = (
            pairwise_distances(data_i, data_j, metric='euclidean') ** 2
        )

        lin_kernel[batch_i[0]:batch_i[-1] + 1, batch_j[0]:batch_j[-1] + 1] = (
            linear_kernel(data_i, data_j)
        )

# Save the kernels in CSV files
linear_kernel_df = pd.DataFrame(lin_kernel, index=subjects, columns=subjects)
linear_kernel_df.to_csv(output_kernels + 'linear_kernel.csv')

euclidean_norm_df = pd.DataFrame(euclidean_norm, index=subjects,
                                 columns=subjects)
euclidean_norm_df.to_csv(output_kernels + 'euclidean_norm.csv')

# Save the target variable in a CSV file
# Change this path
df_y = pd.read_csv("/Volumes/dtlake01.aramis/users/clinica/pac2019/dataset/"
                   "PAC2019_BrainAge_Training.csv")

y = []
for subject in subjects:
    y.append(df_y[df_y['subject_ID'] == subject]['age'].item())

df_y_new = pd.Series(y, index=subjects)
df_y_new.to_csv(output_target + 'age.csv')
