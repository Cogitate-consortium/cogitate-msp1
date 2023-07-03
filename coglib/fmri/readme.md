# Introduction
This document provides an overview of the fMRI pipeline including code and how to run it.

*Authors:* David Richter, Yamil Vidal

## MRI data flow overview
0. Setup dicom to BIDS converter (once); BIDSMAPPER
1. Conversion of MRI DICOM data to BIDS; BIDSCOINER
2. Creation of events.tsv files; PYTHON CODE
3. BIDS validation; BIDSVALIDATOR
4. MRI data quality checks; MRIQC
5. Data rejection; PYTHON CODE
6. MRI preprocessing & visual data/preprocessing quality checks; FMRIPREP
7. Create (anatomical) ROI masks
8. Create regressor event txt files (3 column format)
9. Run 1st, 2nd and 3rd level GLMs
10. Create Decoding ROIs
11. Create seeds for GPPI analysis
12. Run Putative NCC analysis
13. Putative NCC analysis: Generate data for tables
14. Putative NCC analysis: Generate figures

## MRI data flow details
### Setup of data processing:
Run only once during setup (Requires sample dataset)

0. BIDS COIN SETUP
    
    module purge
    module load bidscoin/3.6.3
    bidsmapper /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/temp_raw_for_bidscoin /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids


### MRI data flow

1. DICOM TO BIDS
Converts DICOM to BIDS compliant niftis (switching to a compute node; adjust node as desired)
ssh -X cn09 
module purge
module load bidscoin/3.6.3
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/dicom_to_bids
python 01_convert_dicom_to_bids.py


### 2. Create events.tsv files & perform MRI log file checks (custom code) ###
# Create events.tsv files per run from experiment native log files
ssh -X cn09 
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/logfiles_and_checks
python 01_exp1_create_events_tsv_file.py
python 01_evcLoc_create_events_tsv_file.py
# exp.2 script to be added


### 3. BIDS Validator (https://neuroimaging-core-docs.readthedocs.io) ### 
# Validate BIDS compliance of dataset
ssh -X cn09 
module purge
module load nodejs/12.16.1-GCCcore-7.3.0 
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed
bids-validator bids


### 4a. MRI QC (https://mriqc.readthedocs.io; https://github.com/marcelzwiers/mriqc_sub) ###
# Run MRI QC for visual inspection of (f)MRI data quality. Perform visual inspection of each runs data (see ./bids/derivatives/mriqc)!
module purge
module load mriqc
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed
mriqc_sub.py /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids -t 48 -w /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/scratch/mriqc_workdir -o /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives


### 4b. MRI QC group level (run only when all data has been collected !)
module purge
module load mriqc
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed
mriqc_group.py bids


### 5. Data rejection using MRI QC IQMs (custom code) ###
# Extract IQMs of interest from MRI QC and reject runs/participants from further analysis (run only AFTER all data has been collected & processed with MRIQC !)
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/data_rejection
python 01_analyze_MRIQC_IQMs.py


### 6. fMRIprep. (https://fmriprep.org; https://github.com/marcelzwiers/fmriprep_sub) ###
# Preprocess (f)MRI data. Perform visual inspection of each runs data (see ./bids/derivatives/fmriprep)!
module purge
module load fmriprep/20.2.3
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed
fmriprep_sub.py /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids -w /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/scratch/fmriprep_workdir --time 80 --mem_mb 30000 -n 6 -a " --ignore sbref slicetiming --output-spaces T1w MNI152NLin2009cAsym"


### 7a. Create anatomical ROI masks (custom code; switching to a compute node, adjust node as desired) ###
ssh -X cn09
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load FreeSurfer
module load FSL
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/masks
python 01_create_ROI_masks.py

# Besides running the above code for the desired participants, it should be also run for the MNI152NLin2009cAsym standard brain (for group lvl analyses).
# For this we used the precomputed FreeSurfer output that can be found here: https://figshare.com/articles/dataset/FreeSurfer_reconstruction_of_the_MNI152_ICBM2009c_asymmetrical_non-linear_atlas/4223811

### 7b. Resample anatomical ROI masks to target space ###
ssh -X cn09
module purge
module load ANTs
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/masks
python 02_resample_ROI_masks_to_target_space.py

### 7c. Combine ROIs to create theory specific ROIs. Also creates FFA and LOC masks used for the creation of GPPI seeds ###
ssh -X cn09
module purge
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/masks
python 03_create_theory_ROI_masks.py

### 7d. Resample group lvl anatomical ROI masks to target space ###
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/masks
bash 04_resample_MNI152_ROIs.sh

### 7e. Combine ROIs to create theory specific ROIs (group lvl) ###
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/masks
python 05_create_theory_ROI_masks_MNI152.py

### 8. Create regressor event txt file (custom code) ###
# Create regressor event txt files; 3 column format; 1 per regressors (FSL FEAT compliant) from information in events.tsv files per run.
ssh -X cn09 
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/logfiles_and_checks
python 02_exp1_create_regressor_txt_files.py
python 02_evcLoc_create_regressor_txt_files.py
# exp.2 script to be added


### 9. Run 1st, 2nd and 3rd level GLMs ###
# Create confound regressor files to be used in 1st level GLMs. Then run first and second level GLMs using FSL FEAT; adjust analysis level in python script (Still requires some work). Perform visual inspection of each run's output data (see ./bids/derivatives/fslFeat)!
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load Spyder/4.1.5-foss-2019a-Python-3.7.2
module load FSL
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/glm
python 01_create_confound_regressor_ev_file.py
python 02_run_fsf_feat_analyses.py    # Do for each GLM lvl


### 10. Create Decoding ROIs ###
# Requires anatomical ROIs (step 7) and GLMs (step 9).
ssh -X cn09
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/decoding_rois
python 01_create_decoding_rois_all_runs.py
python 02_create_decoding_rois_leave_one_run_out.py


### 11. Create seeds for GPPI analysis ###
# Requires anatomical ROIs (step 7) and GLMs (step 9).
ssh -X cn09
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/seeds_for_gppi
python 01_create_gppi_seeds.py


### 12. Run Putative NCC analysis ###
# Run putative NCC analysis on FSL Feat outputs + perform bayesian tests.
ssh -X cn09
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load FSL
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/putative_ncc
python 01_PutativeNCC_analysis_on_FEAT_copes.py                  # Group lvl univariate
python 02_putative_ncc_create_C_not_A_or_B_maps.py               # Exclude voxels responsive to task goals and task relevance (Group lvl univariate)
python 03_putative_ncc_analysis_on_FEAT_copes_subject_level.py   # Subject lvl univariate
python 04_putative_ncc_subject_level_create_C_not_A_or_B_maps.py # Exclude voxels responsive to task goals and task relevance (Subject lvl univariate)
python 05_multivariate_putative_ncc_analysis.py                  # Group lvl multivariate
python 06_multivariate_putative_ncc_create_C_not_A_or_B_maps.py  # Exclude voxels responsive to task goals and task relevance (Group lvl multivariate)
python 07_putative_ncc_merge_phases.py                           # Combine optimization and replication phases, for plotting purposes

### 13. Putative NCC analysis: Generate data for tables ###
# Count detected voxels in each anatomical ROI and save data to csv files (used to produce tables).
# Requires anatomical masks (7) and the results of putative NCC analizes (12).
ssh -X cn09
module purge
module load Python/3.8.6-GCCcore-10.2.0
cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Yamil/fMRI/putative_ncc_tables
python 01_putative_ncc_group_level_tables.py
python 02_putative_ncc_subject_level_tables.py
python 03_multivariate_putative_ncc_group_level_tables.py

### 14. Putative NCC analysis: Generate figures ###
# Requires Slice Display (https://github.com/bramzandbelt/slice_display), and MATLAB
# Do not run on HPC. Run locally.
matlab Putative_NCC_01_univariate.m    # Univariate pNCC, main figure (5) and individual stimulus categories
matlab Putative_NCC_02_AB.m            # Areas responsive to task goals and task relevance
matlab Putative_NCC_03_multivariate.m  # Multivariate pNCC
matlab Putative_NCC_04_z_maps.m        # Individual z maps for each stimulus category and condition (relevant and irrelevant)




