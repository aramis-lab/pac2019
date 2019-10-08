import os
import nibabel as nib
import numpy as np

base_dir = 'working/directory'

# Read from txt file the list of subjects with a wrong orientation
with open(os.path.join(base_dir, 'IDs_site14_header2Correct.txt')) as f:
    subject_list = f.read().splitlines()

# Permuation to apply to the sform
permutation = np.array([2, 1, 0, 3, 5, 4, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15])

for subject in subject_list:
    print(subject)

    # Load image with wrong orientation
    im_wrongorientation_filename = os.path.join(
        base_dir, 'raw', subject+'_raw.nii.gz')
    im_wrongorientation_nifti = nib.load(im_wrongorientation_filename)

    # Extract sform
    sform = im_wrongorientation_nifti.get_sform().flatten()

    # Apply transformation to the sform
    new_sform = sform[permutation].reshape(4, 4)

    # Save image with good orientation
    im_goodorientation_filename = os.path.join(
        base_dir, 'raw_site14', subject+'_raw.nii.gz')
    im_goodorientation_header = im_wrongorientation_nifti.header.copy()
    im_goodorientation_header.set_sform(new_sform)
    im_goodorientation_nifti = nib.Nifti1Image(
        im_wrongorientation_nifti.get_data(),
        new_sform,
        header=im_goodorientation_header)
    nib.save(im_goodorientation_nifti, im_goodorientation_filename)
