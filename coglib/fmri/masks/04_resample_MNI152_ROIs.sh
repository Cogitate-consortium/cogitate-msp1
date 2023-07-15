#!/bin/bash
# Author: Yamil Vidal
# Email: hvidaldossantos@gmail.com
module purge
module load AFNI

# Define input files directory
input_dir=/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/masks/ICBM2009c_asym_nlin
master=/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fslFeat/group/ses-V1/phase3_V1_subset/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/mask.nii.gz

# Loop over input files
for input_file in ${input_dir}/ICBM2009c_asym_nlin_bh*.nii.gz; do
    base_name=$(basename ${input_file})
	echo Resampling ${base_name}
	# Define output file name
    output_file=${input_file/.nii.gz/_space-MNI152NLin2009cAsym.nii.gz}
    # resamples ROIs to the dimensions of the data
    #3dresample -input ${input_file} -prefix ${output_file} -orient LPI -dxyz 2.019 2.019 2
    3dresample -input ${input_file} -master ${master} -prefix ${output_file}
done

