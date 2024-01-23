% Runs first-level GLM analysis using SPM12 toolbox

%Author: Aya Khalaf
%Email: aya.khalaf@yale.edu
%Date created: 04-21-2022


% SPM12 path
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm12/')
% BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids';
% fMRIprep pre-processed fMRI data path
fmriprep_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep';
% 3D nifti files path
spm_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/spm';
% Confound regressors path
confound_regressors_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/regressoreventfiles';
% Output path
output_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/glm';
% tsv file with information on phase 3 subjects 
tsv_data=tdfread(fullfile(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv'));
phase3_subjects = tsv_data.participant_id;
phase3_subjects = cellstr(phase3_subjects);
% Subject list
subject_list = cell2struct(phase3_subjects, 'name', length(phase3_subjects));
spm fmri
%Parameters
% Scan repetition time
TR = 1.5;
% Number of volumes that will not be included in the analysis 
no_dummy_vols = 3;
excluded_subs = [];
% Loop over subjects
for i = 1: length (subject_list)
% Run folders    
run_folders = dir(fullfile(fmriprep_dir,subject_list(i).name,'ses-V1','func','*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'));
% Event files 
tsv_files = dir(fullfile(bids_dir,subject_list(i).name,'ses-V1','func', '*tsv')); 
% Confound regressors files
confound_regressors = dir(fullfile(confound_regressors_dir, subject_list(i).name,'ses-V1', 'confound_event_files', '*txt')); 
num_runs = length(run_folders);
% Skip subjects if they have less than 8 runs or if their confound regressors data or event files do not exist   
if(isempty(tsv_files) || isempty(confound_regressors) || (length(run_folders)<8))
    excluded_subs = [excluded_subs, subject_list(i).name];
    continue
end
% Create an output dir for each subject to save glm output data
subject_output_dir = fullfile(output_dir, subject_list(i).name,'ses-V1');
if ~exist(subject_output_dir)
mkdir (subject_output_dir)
else
continue
end
matlabbatch{1}.spm.stats.fmri_spec.dir = {subject_output_dir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

% Loop over runs
for r = 1:num_runs 
% List smoothed nifti files within each run    
nifti_list = dir(fullfile(spm_dir,subject_list(i).name,'ses-V1','func', '3D nifti smoothed', ['Run_' num2str(r)]));
nifti_list = nifti_list(~cell2mat({nifti_list(:).isdir}));
nifti_list = fullfile({fullfile(spm_dir,subject_list(i).name,'ses-V1','func','3D nifti smoothed', ['Run_' num2str(r)])}',{nifti_list.name}');
% Remove dummy volues
nifti_list = nifti_list(no_dummy_vols+1:end);
% Read event data
tsv_data=tdfread(fullfile(bids_dir,subject_list(i).name,'ses-V1','func',tsv_files(r).name));
conditions = tsv_data.trial_type;
conditions_cell = cellstr(conditions);
relevance = tsv_data.task_relevance;
relevance_cell = cellstr(relevance);
durations = tsv_data.duration;
%Shift the onset times to account for the removed volumes
onsets = tsv_data.onset - (TR * no_dummy_vols);

% Construct the GLM
matlabbatch{1}.spm.stats.fmri_spec.sess(r).scans = nifti_list;                                               
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).name = 'targetScreen';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).onset = onsets(strcmp('targetScreen',conditions_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).duration = durations(strcmp('targetScreen',conditions_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).name = 'face_relevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).onset = onsets(strcmp('face',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).duration = durations(strcmp('face',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).name = 'face_irrelevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).onset = onsets(strcmp('face',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).duration = durations(strcmp('face',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).name = 'object_relevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).onset = onsets(strcmp('object',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).duration = durations(strcmp('object',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).name = 'object_irrelevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).onset = onsets(strcmp('object',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).duration = durations(strcmp('object',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).name = 'letter_relevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).onset = onsets(strcmp('letter',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).duration = durations(strcmp('letter',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(6).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).name = 'letter_irrelevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).onset = onsets(strcmp('letter',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).duration = durations(strcmp('letter',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(7).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).name = 'falseFont_relevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).onset = onsets(strcmp('falseFont',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).duration = durations(strcmp('falseFont',conditions_cell) & strcmp('relevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(8).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).name = 'falseFont_irrelevant';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).onset = onsets(strcmp('falseFont',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).duration = durations(strcmp('falseFont',conditions_cell) & strcmp('irrelevant',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(9).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).name = 'target';
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).onset = onsets(strcmp('target',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).duration = durations(strcmp('target',relevance_cell));
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(10).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess(r).regress = struct('name', {}, 'val', {});
  
matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg ={fullfile(confound_regressors_dir, subject_list(i).name,'ses-V1', 'confound_event_files', confound_regressors(r).name)};
matlabbatch{1}.spm.stats.fmri_spec.sess(r).hpf = 128;

end
%%
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Run the GLM model
spm_jobman('run',matlabbatch)

end
