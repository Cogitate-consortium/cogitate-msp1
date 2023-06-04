% Runs generalized psychophysiological interactions (gppi)analysis on the relevant and
% irrelevant conditions combined using gppi toolbox implemented in SPM08
% The code runs gppi analysis using FFA and LOC seeds 

%Author: Aya Khalaf
%Date created: 04-22-2022

% Select seed
seed = 'FFA'; % either FFA or LOC
% Select seed size in voxels
n_voxels = 300;
gppi_analysis_combined(seed, n_voxels)

% Navigate to gppi codes path 
cd('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Aya/fMRI/gppi')
% Select seed
seed = 'LOC'; % either FFA or LOC
% Select seed size in voxels
n_voxels = 300;
gppi_analysis_combined(seed, n_voxels)