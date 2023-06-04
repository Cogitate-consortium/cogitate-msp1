function gppi_analysis(seed, n_voxels)
% Performs gppi analysis using gppi toolbox implemented in SPM08

%Author: Aya Khalaf
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
seeds_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi_seeds';
% Output path
output_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/glm';

% tsv file with information on phase 3 subjects 
tsv_data=tdfread(fullfile(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv'));
phase3_subjects = tsv_data.participant_id;
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
P.GroupDir = fullfile('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/contrasts_phase2', roi_folder);
list=[];
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
if(exist(fullfile(output_dir, subject_list(i).name,'ses-V1',roi_folder,'SPM.mat'), 'file'))
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
P.Tasks={'1'  'face_relevant'  'object_relevant'  'letter_relevant' 'falseFont_relevant' 'face_irrelevant'  'object_irrelevant'  'letter_irrelevant' 'falseFont_irrelevant', 'target', 'targetScreen'};

P.Weights=[];
P.analysis='psy';
P.method='cond';
P.CompContrasts=1;
P.Weighted=0;

P.equalroi=0;
P.FLmask=1;

P.Contrasts(1).left={'face_relevant'};
P.Contrasts(1). right={'none'};
P.Contrasts(1).STAT='T';
P.Contrasts(1).Weighted=0;
P.Contrasts(1).MinEvents=5;
P.Contrasts(1).name='face_rel';

P.Contrasts(2).left={'object_relevant'};
P.Contrasts(2).right={'none'};
P.Contrasts(2).STAT='T';
P.Contrasts(2).Weighted=0;
P.Contrasts(2).MinEvents=5;
P.Contrasts(2).name='object_rel';

P.Contrasts(3).left={'letter_relevant'};
P.Contrasts(3).right={'none'};
P.Contrasts(3).STAT='T';
P.Contrasts(3).Weighted=0;
P.Contrasts(3).MinEvents=5;
P.Contrasts(3).name='letter_rel';

P.Contrasts(4).left={'falseFont_relevant'};
P.Contrasts(4).right={'none'};
P.Contrasts(4).STAT='T';
P.Contrasts(4).Weighted=0;
P.Contrasts(4).MinEvents=5;
P.Contrasts(4).name='falseFont_rel';

P.Contrasts(5).left={'face_relevant'};
P.Contrasts(5). right={'object_relevant'};
P.Contrasts(5).STAT='T';
P.Contrasts(5).Weighted=0;
P.Contrasts(5).MinEvents=5;
P.Contrasts(5).name='face_rel-object_rel';

P.Contrasts(6).left={'letter_relevant'};
P.Contrasts(6).right={'falseFont_relevant'};
P.Contrasts(6).STAT='T';
P.Contrasts(6).Weighted=0;
P.Contrasts(6).MinEvents=5;
P.Contrasts(6).name='letter_rel-falseFont_rel';

P.Contrasts(7).left={'face_irrelevant'};
P.Contrasts(7). right={'none'};
P.Contrasts(7).STAT='T';
P.Contrasts(7).Weighted=0;
P.Contrasts(7).MinEvents=5;
P.Contrasts(7).name='face_irr';

P.Contrasts(8).left={'object_irrelevant'};
P.Contrasts(8).right={'none'};
P.Contrasts(8).STAT='T';
P.Contrasts(8).Weighted=0;
P.Contrasts(8).MinEvents=5;
P.Contrasts(8).name='object_irr';

P.Contrasts(9).left={'letter_irrelevant'};
P.Contrasts(9).right={'none'};
P.Contrasts(9).STAT='T';
P.Contrasts(9).Weighted=0;
P.Contrasts(9).MinEvents=5;
P.Contrasts(9).name='letter_irr';

P.Contrasts(10).left={'falseFont_irrelevant'};
P.Contrasts(10).right={'none'};
P.Contrasts(10).STAT='T';
P.Contrasts(10).Weighted=0;
P.Contrasts(10).MinEvents=5;
P.Contrasts(10).name='falseFont_irr';

P.Contrasts(11).left={'face_irrelevant'};
P.Contrasts(11). right={'object_irrelevant'};
P.Contrasts(11).STAT='T';
P.Contrasts(11).Weighted=0;
P.Contrasts(11).MinEvents=5;
P.Contrasts(11).name='face_irr-object_irr';

P.Contrasts(12).left={'letter_irrelevant'};
P.Contrasts(12).right={'falseFont_irrelevant'};
P.Contrasts(12).STAT='T';
P.Contrasts(12).Weighted=0;
P.Contrasts(12).MinEvents=5;
P.Contrasts(12).name='letter_irr-falseFont_irr';

% Run gppi
PPPI(P)
end
end
