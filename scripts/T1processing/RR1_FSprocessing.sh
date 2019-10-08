#!/bin/bash
# CLUSTER OPTIONS
wd="working/directory"
fsdir="freeSurfer/directory"

echo ${SLURM_ARRAY_TASK_ID}.$(sed -n "${SLURM_ARRAY_TASK_ID}{p;q}" ${wd}/dataset/PAC2019_BrainAge_Training.csv) 
dat=$(sed -n "${SLURM_ARRAY_TASK_ID}{p;q}" ${wd}/dataset/PAC2019_BrainAge_Training.csv) 
 echo ${dat}
ID=$(echo ${dat} | cut -f 1 -d ',' ) 
 echo ${ID}
 mkdir -p ${wd}/FS/ 
 mkdir -p ${wd}/FSOUTPUT/FSresults/${ID}/ 
 mkdir -p ${wd}/FSOUTPUT/ENIGMAshapeResults/${ID}/ 
module load FreeSurfer 
SUBJECTS_DIR=${wd}/FS 
 recon-all -subject ${ID} -i ${wd}/dataset/raw/${ID}_raw.nii.gz -all -qcache 
SUBJECTS_DIR=${fsdir}/subjects 
for hemi in lh rh 
do 
for moda in area thickness 
do 
for fwhm in 0 5 10 15 20 25 
do 
 ${fsdir}/bin/mris_convert -c ${wd}/FS/${ID}/surf/${hemi}.${moda}.fwhm${fwhm}.fsaverage.mgh ${fsdir}/subjects/fsaverage/surf/${hemi}.orig ${wd}/FSOUTPUT/FSresults/${ID}/${hemi}.${moda}.fwhm${fwhm}.fsaverage.asc 
done 
for fsav in fsaverage3 fsaverage4 fsaverage5 fsaverage6 
do 
 ${fsdir}/bin/mri_surf2surf --s fsaverage --hemi ${hemi} --sval ${wd}/FS/${ID}/surf/${hemi}.${moda}.fwhm0.fsaverage.mgh --trgsubject ${fsav} --tval ${wd}/FS/${ID}/surf/${hemi}.${moda}.fwhm0.${fsav}.mgh 
 ${fsdir}/bin/mris_convert -c ${wd}/FS/${ID}/surf/${hemi}.${moda}.fwhm0.${fsav}.mgh ${fsdir}/subjects/${fsav}/surf/${hemi}.orig ${wd}/FSOUTPUT/FSresults/${ID}/${hemi}.${moda}.fwhm0.${fsav}.asc 
done 
done 
done 
 perl ${wd}/bin/ENIGMA_shape/MedialDemonsShared/bin/Medial_Demons_shared.pl ${wd}/FS/${ID}/mri/aseg.mgz 10 11 12 13 17 18 26 49 50 51 52 53 54 58 ${wd}/FS/${ID}/ENIGMA_shape/ ${wd}/bin/ENIGMA_shape/MedialDemonsShared ${fsdir}/bin 
 rsync -r ${wd}/FS/${ID}/ENIGMA_shape/* ${wd}/FSOUTPUT/ENIGMAshapeResults/${ID}/ 
 rsync -r ${wd}/FS/${ID}/stats/* ${wd}/FSOUTPUT/FSresults/${ID}/ 
