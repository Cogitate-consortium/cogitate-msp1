function gppi_analysis_combined(seed, n_voxels)
% Performs gppi analysis on the relevant and irrelevant conditions combined using gppi toolbox implemented in SPM08

%Author: Aya Khalaf
%Email: aya.khalaf@yale.edu
%Date created: 04-22-2022

% SPM8 path
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm8/')
% gppi path
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm8/toolbox/PPPI/')
addpath('/hpc/workspace/TOOLBOXES/MATLAB/spm8/toolbox/PPPI/PPPIv13/')

% BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids';
% 3D nifti files path
spm_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/spm';
% Seeds path
seeds_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi_seeds/new';
% Output path
output_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/glm_combined_phase2';

% tsv file with information on phase 3 subjects 
tsv_data=tdfread(fullfile(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv'));
phase3_subjects = cellstr(phase3_subjects);
% Subject list
subject_list = cell2struct(phase3_subjects, 'name', length(phase3_subjects));
% Specify seed folder and nifti file name 
if strcmp(seed,'FFA') 
seed_filename ='face';    
P.Region=strcat('FFA_gPPI_',num2str(n_voxels));
roi_folder = strcat('PPI_FFA_gPPI_',num2str(n_voxels));
elseif strcmp(seed,'LOC')
seed_filename ='object';     
P.Region=strcat('LOC_gPPI_',num2str(n_voxels));
roi_folder = strcat('PPI_LOC_gPPI_',num2str(n_voxels));    
end
P.GroupDir = fullfile('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/contrasts_combined', roi_folder);

% Loop over Subjects
for i =1: length (subject_list)
% Skip subject if they do not have a GLM dir   
if(~exist(fullfile(output_dir, subject_list(i).name,'ses-V1'), 'dir'))
continue
end
% Skip subject if they do not have a seeds dir  
if(~exist(fullfile(seeds_dir, subject_list(i).name), 'dir'))
continue
end 

subject_output_dir = fullfile(output_dir, subject_list(i).name,'ses-V1');
% Unzip seed file (convert nii.gz file to nii file)
seed_struct = dir(fullfile(seeds_dir, subject_list(i).name,['*_' seed_filename '*' strcat('_',num2str(n_voxels)) '*.nii.gz']));
seed = gunzip(fullfile(seeds_dir,subject_list(i).name,seed_struct.name));
P.VOI = seed{1};
P.subject=subject_list(i).name;
P.directory=subject_output_dir;

% Specify gppi parameters and contrasts
P.Estimate=1;
P.contrast=0;
P.extract='eig';
P.Tasks={'1'  'face'  'object'  'letter' 'falseFont'  'target', 'targetScreen'};

P.Weights=[];
P.analysis='psy';
P.method='cond';
P.CompContrasts=1;
P.Weighted=0;

P.equalroi=0;
P.FLmask=1;

P.Contrasts(1).left={'face'};
P.Contrasts(1). right={'none'};
P.Contrasts(1).STAT='T';
P.Contrasts(1).Weighted=0;
P.Contrasts(1).MinEvents=5;
P.Contrasts(1).name='face';

P.Contrasts(2).left={'object'};
P.Contrasts(2).right={'none'};
P.Contrasts(2).STAT='T';
P.Contrasts(2).Weighted=0;
P.Contrasts(2).MinEvents=5;
P.Contrasts(2).name='object';

P.Contrasts(3).left={'letter'};
P.Contrasts(3).right={'none'};
P.Contrasts(3).STAT='T';
P.Contrasts(3).Weighted=0;
P.Contrasts(3).MinEvents=5;
P.Contrasts(3).name='letter';

P.Contrasts(4).left={'falseFont'};
P.Contrasts(4).right={'none'};
P.Contrasts(4).STAT='T';
P.Contrasts(4).Weighted=0;
P.Contrasts(4).MinEvents=5;
P.Contrasts(4).name='falseFont';

P.Contrasts(5).left={'face'};
P.Contrasts(5). right={'object'};
P.Contrasts(5).STAT='T';
P.Contrasts(5).Weighted=0;
P.Contrasts(5).MinEvents=5;
P.Contrasts(5).name='face-object';

P.Contrasts(6).left={'letter'};
P.Contrasts(6).right={'falseFont'};
P.Contrasts(6).STAT='T';
P.Contrasts(6).Weighted=0;
P.Contrasts(6).MinEvents=5;
P.Contrasts(6).name='letter-falseFont';



% Run gppi
PPPI(P)
end
end
