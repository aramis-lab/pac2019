#!/bin/bash
# CLUSTER OPTIONS
wd="working/directory"
echo fwhm0.fsaverage
awk 'BEGIN{OFS="\t"}$1="rht_"$1' ${wd}/dataset/FSOUTPUT/rh.thickness.fwhm0.fsaverage.0.UKB.txt > ${wd}/dataset/FSOUTPUT/CombinedData/rh.thickness.fwhm0.fsaverage.PAC2019.txt
awk 'BEGIN{OFS="\t"}$1="lht_"$1' ${wd}/dataset/FSOUTPUT/lh.thickness.fwhm0.fsaverage.0.UKB.txt > ${wd}/dataset/FSOUTPUT/CombinedData/lh.thickness.fwhm0.fsaverage.PAC2019.txt
awk 'BEGIN{OFS="\t"}$1="rha_"$1' ${wd}/dataset/FSOUTPUT/rh.area.fwhm0.fsaverage.0.UKB.txt > ${wd}/dataset/FSOUTPUT/CombinedData/rh.area.fwhm0.fsaverage.PAC2019.txt 
awk 'BEGIN{OFS="\t"}$1="lha_"$1' ${wd}/dataset/FSOUTPUT/lh.area.fwhm0.fsaverage.0.UKB.txt > ${wd}/dataset/FSOUTPUT/CombinedData/lh.area.fwhm0.fsaverage.PAC2019.txt 
for moda in area thickness
do
for hemi in lh rh
do
for batch in 1 2 3 4 5 6 7 8 9
do 
echo ${batch}.${hemi}.${moda} 
awk '{$1=""}1' ${wd}/dataset/FSOUTPUT/${hemi}.${moda}.fwhm0.fsaverage.${batch}.UKB.txt | awk '{$1=$1}1' > ${wd}/dataset/FSOUTPUT/${hemi}.${moda}.fwhm0.fsaverage.UKB.${batch}_noColNames.txt 
paste ${wd}/dataset/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.PAC2019.txt ${wd}/dataset/FSOUTPUT/${hemi}.${moda}.fwhm0.fsaverage.UKB.${batch}_noColNames.txt > ${wd}/dataset/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.UKB_2.txt 
cp ${wd}/dataset/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.UKB_2.txt ${wd}/dataset/FSOUTPUT/CombinedData/${hemi}.${moda}.fwhm0.fsaverage.PAC2019.txt 
done 
done 
done 

