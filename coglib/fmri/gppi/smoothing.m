% Smooths fMRI BOLD data using a Gaussian smoothing kernel implemented in SPM 12 toolbox
% The input fMRI BOLD data to be smoothed are the 3D nifti files produced by nifti3D_conversion.m

%Author: Aya Khalaf
%Email: aya.khalaf@yale.edu
%Date created: 04-20-2022

% SPM12 path
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm12/')
% 3D nifti files path
spm_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/spm';

% List of subjects with 3D nifti files
subject_list = dir(fullfile(spm_dir, 'sub*'));
subject_list= subject_list(cell2mat({subject_list(:).isdir}));

% Loop over subjects
for i = 1: length (subject_list)
% Create a folder to save the smoothed data
output_dir = fullfile(spm_dir,subject_list(i).name, 'ses-V1','func', '3D nifti smoothed');
if(~exist(output_dir))
    mkdir(output_dir)
else
    continue
end
run_folders = dir(fullfile(spm_dir,subject_list(i).name,'ses-V1','func','3D nifti', 'Run*'));
num_runs = length(run_folders);

% Loop over runs
for r = 1:num_runs
% Create a folder for each run
mkdir(fullfile(output_dir, ['Run_' num2str(r)]))    
% Copy the non-smoothed data to the run folder
copyfile(fullfile(spm_dir,subject_list(i).name,'ses-V1','func','3D nifti', ['Run_' num2str(r)]),fullfile(output_dir, ['Run_' num2str(r)]) )

% Smooth data saved in the run folder 
run_dir=dir(fullfile(output_dir,run_folders(r).name,'*.nii'));
nii_files = fullfile({fullfile(output_dir,run_folders(r).name)}',{run_dir.name}'); 
matlabbatch{1}.spm.spatial.smooth.data = nii_files;
% Gaussian smoothing kernel with FWHM of 5 mm
matlabbatch{1}.spm.spatial.smooth.fwhm = [5 5 5];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = 's';
spm('defaults', 'FMRI');
spm_jobman('run',matlabbatch)
cd(fullfile(output_dir, ['Run_' num2str(r)]))
% Delete the non-smoothed files from the run folder and keep only the smoothed ones
delete 'sub*'
end
end