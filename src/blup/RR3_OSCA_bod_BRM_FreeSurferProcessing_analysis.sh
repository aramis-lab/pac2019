#!/bin/bash
# CLUSTER OPTIONS
wd="working/directory"

# Convert to BOD format
for hemi in lh rh 
do 
for moda in area thickness 
do 
 ${wd}/bin/osca_05032019 --tefile ${wd}/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.PAC2019.QC.txt --make-bod --no-fid --out ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --thread-num 5
 ${wd}/bin/osca_05032019 --befile ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --make-orm-bin --out ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --thread-num 5
done 
for moda in thick LogJacs 
do 
 ${wd}/bin/osca_05032019 --efile ${wd}/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.PAC2019.txt --make-bod --no-fid --out ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --thread-num 5 
 ${wd}/bin/osca_05032019 --befile ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --make-orm-bin --out ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 --thread-num 5
done 
done  

# Merge BOD files
 rm ${wd}/LMManalysis/BODFiles/fwhm0.fsaverage.list
 touch ${wd}/LMManalysis/BODFiles/fwhm0.fsaverage.list
for hemi in lh rh 
do 
for moda in area thickness thick LogJacs 
do 
 echo ${wd}/LMManalysis/BODFiles/${hemi}.${moda}.fwhm0.fsaverage.PAC2019 >> ${wd}/LMManalysis/BODFiles/fwhm0.fsaverage.list
done 
done 
 ${wd}/bin/osca_05032019 --befile-flist ${wd}/LMManalysis/BODFiles/fwhm0.fsaverage.list --make-bod --out ${wd}/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019 --thread-num 5 
 ${wd}/bin/osca_05032019 --multi-orm ${wd}/LMManalysis/BODFiles/fwhm0.fsaverage.list --make-orm-bin --out ${wd}/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019 --thread-num 5

# Standardise vertices
 ${wd}/bin/osca_05032019 --befile ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019 --get-mean --get-variance --out ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019
 ${wd}/bin/osca_05032019 --befile ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019 --std-probe --make-bod --out ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019.STD --thread-num 5 
 
# Run BLUP prediction and calculate scores
 ${wd}/bin/osca_05032019 --reml --orm ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019 --keep ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --pheno ${wd}/dataset/PAC2019_ageSD.txt --reml-pred-rand --reml-maxit 1000 --out ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5   
 ${wd}/bin/osca_05032019 --befile ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019.STD --keep ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --blup-probe ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter${SLURM_ARRAY_TASK_ID}.indi.blp --out ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5 
 ${wd}/bin/osca_05032019 --befile ${wd}/results/linear/LMManalysis/BODFiles/AllVertices.fwhm0.fsaverage.PAC2019.STD --remove ${wd}/dataset/cross_validation/train_splits-5/IDs_split_${SLURM_ARRAY_TASK_ID}.txt --score ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter${SLURM_ARRAY_TASK_ID}.probe.blp --out ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter${SLURM_ARRAY_TASK_ID} --thread-num 5 

# Calculate BLUP score on dataset for evaluation
 ${wd}/bin/osca_05032019 --befile ${wd}/results/linear/LMManalysis/BODFilesTest/AllVertices.fwhm0.fsaverage.PAC2019.STD --score ${wd}/results/linear/LMManalysis/BLUP_SD_noQC/age_iter0.probe.blp --out ${wd}/results/linear/LMManalysis_test/age_iter0 
 