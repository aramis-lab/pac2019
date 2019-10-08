import argparse
import torch
import os
from os import path
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI.")
parser.add_argument("--output_dir", type=str, default=None)


ret = parser.parse_known_args()
options = ret[0]
if ret[1]:
    print("unknown arguments: %s" % parser.parse_known_args()[1])

if options.output_dir is None:
    output_dir = options.input_dir
else:
    if not path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    output_dir = options.output_dir

subjects = os.listdir(options.input_dir)
for subject in subjects:
    filename = subject.split('.')[0]
    pt_path = path.join(output_dir, filename + ".pt")
    nii_path = path.join(options.input_dir, subject)
    nii_data = nib.load(nii_path)
    image = nii_data.get_data()
    # print(image.dtype)
    # print(image.shape)
    image = image.astype(float)
    np.nan_to_num(image, copy=False)
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0)

    torch.save(tensor, pt_path)
