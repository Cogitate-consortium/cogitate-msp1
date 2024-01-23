% Runs generalized psychophysiological interactions (gppi)analysis using gppi toolbox implemented in SPM08
% The code runs gppi analysis using FFA and LOC seeds 
% The size of the seeds included in the analysis has to be specified for the code to run 

%Author: Aya Khalaf
%Email: aya.khalaf@yale.edu
%Date created: 04-22-2022

% Select seed
seed = 'FFA'; % either FFA or LOC
% Select seed size in voxels
n_voxels = 300;
gppi_analysis(seed, n_voxels)

% Navigate to gppi codes path 
cd('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Aya/fMRI/gppi')
% Select seed
seed = 'LOC'; % either FFA or LOC
% Select seed size in voxels
n_voxels = 300;
gppi_analysis(seed, n_voxels)