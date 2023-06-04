% Converts 4D nifti files to 3D nifti files to be analyzed by SPM12 toolbox
% The 4D nifti files to be converted are the pre-processed nifti files produced by fmriprep

%Author: Aya Khalaf
%Date created: 04-18-2022

% SPM12 path
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm12/')
% BIDS path 
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids';
% fMRIprep pre-processed fMRI data path
fmriprep_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep';
% Output path for saving 3D nifti files
spm_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/spm';

% tsv file with information on phase 2 subjects 
tsv_data=tdfread(fullfile(bids_dir,'participants_fMRI_QC_included_phase2_sesV1.tsv'));
phase2_subjects = tsv_data.participant_id;
phase2_subjects = cellstr(phase2_subjects);
% Subject list
subject_list = cell2struct(phase2_subjects, 'name', length(phase2_subjects));

% Loop over subjects
for i = 1: length (subject_list)  
% Create a folder to save the 3D converted data    
output_dir = fullfile(spm_dir,subject_list(i).name, 'ses-V1','func', '3D nifti');
if(~exist(output_dir))
    mkdir(output_dir)
else
    continue
end
% Select the pre-processed 4D nifti files for all runs from the directory of each subject    
run_folders = dir(fullfile(fmriprep_dir,subject_list(i).name,'ses-V1','func','*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'));
num_runs = length(run_folders);
% Loop over runs
for r = 1:num_runs
mkdir(fullfile(output_dir, ['Run_' num2str(r)]))  
% Unzip 4D nifti files (convert file format from .nii.gz to .nii )
nifti_4D=gunzip(fullfile(fmriprep_dir,subject_list(i).name,'ses-V1','func',run_folders(r).name));
% Convert 4D nii files to 3D nii files
matlabbatch{1}.spm.util.split.vol = nifti_4D(1);
matlabbatch{1}.spm.util.split.outdir = {fullfile(output_dir, ['Run_' num2str(r)])};
spm('defaults', 'FMRI');
spm_jobman('run',matlabbatch)
end
cd(fullfile(fmriprep_dir,subject_list(i).name,'ses-V1','func'));
delete '*.nii'
end
