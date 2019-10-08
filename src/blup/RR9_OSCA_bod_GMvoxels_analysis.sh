#!/bin/bash
# CLUSTER OPTIONS
wd="working/directory"

# Convert in BOD format (10 batches of participants)
 ${wd}/bin/osca_05032019 --tefile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.${SLURM_ARRAY_TASK_ID}.txt --make-bod --no-fid --out ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.${SLURM_ARRAY_TASK_ID}  --thread-num 5 

# Merge the different batches (bind)
 ${wd}/bin/osca_05032019 --befile-flist ${wd}/dataset/gm_tsv_combined/MergeBodFiles_GMvoxels.txt --make-bod --no-fid --out ${wd}/dataset/gm_tsv_combined/GMvoxels.combined 

# Prune voxels with SD<0.03
 ${wd}/bin/osca_05032019 --befile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined --make-bod --sd-min 0.03 --make-bod --out ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 

# Get mean and variance of each voxel
 ${wd}/bin/osca_05032019 --befile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined --get-variance --get-mean --out ${wd}/dataset/gm_tsv_combined/GMvoxels.combined 

# Make brain-relatedness matrix
 ${wd}/bin/osca_05032019 --befile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 --make-orm --out ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 --task-num 1000 --task-id 1 --thread-num 10 

# Run REML analysis and BLUP prediction
 ${wd}/bin/osca_05032019 --reml --orm ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 --keep ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --pheno ${wd}/dataset/PAC2019_ageSD.txt --reml-pred-rand --reml-maxit 1000 --out ${wd}/results/linear/LMManalysis/BLUP_GMvoxels_SDpruned0.03/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5   
 ${wd}/bin/osca_05032019 --befile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 --keep ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --blup-probe ${wd}/results/linear/LMManalysis/BLUP_GMvoxels_SDpruned0.03/age_iter${SLURM_ARRAY_TASK_ID}.indi.blp --out ${wd}/results/linear/LMManalysis/BLUP_GMvoxels_SDpruned0.03/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5 
 ${wd}/bin/osca_05032019 --befile ${wd}/dataset/gm_tsv_combined/GMvoxels.combined.SDpruned0.03 --remove ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --score ${wd}/results/linear/LMManalysis/BLUP_GMvoxels_SDpruned0.03/age_iter${SLURM_ARRAY_TASK_ID}.probe.blp --out ${wd}/results/linear/LMManalysis/BLUP_GMvoxels_SDpruned0.03/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5 
