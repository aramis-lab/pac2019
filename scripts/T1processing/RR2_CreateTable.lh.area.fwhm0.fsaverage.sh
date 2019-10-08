#!/bin/bash
# CLUSTER OPTIONS
wd="working/directory"
 echo "vertexnum" > ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.UKB.txt 
 awk '{print $1}' ${wd}/dataset/FSOUTPUT/FSresults/sub0/lh.area.fwhm0.fsaverage.asc >> ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.UKB.txt 
for ID in $(awk -F"," "NR>0 {print $1}" ${wd}/dataset/FSOUTPUT/PAC2019_IDs_batch${SLURM_ARRAY_TASK_ID}) 
do 
 echo ${ID} 
if [ -f ${wd}/dataset/FSOUTPUT/FSresults/${ID}/lh.area.fwhm0.fsaverage.asc ] 
then 
 echo ${ID} > ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.temp.lta 
 awk '{print $5}' ${wd}/dataset/FSOUTPUT/FSresults/${ID}/lh.area.fwhm0.fsaverage.asc >> ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.temp.lta 
 paste ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.UKB.txt ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.temp.lta > ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.temp2.lta 
 cp ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.temp2.lta ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.${SLURM_ARRAY_TASK_ID}.UKB.txt 
fi 
done 
