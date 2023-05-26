% Script looping over all directories:
addpath ..
activation_fig_root = "/hpc/users/alexander.lepauvre/plotting_test/activation_analysis";
dir_content = dir(activation_fig_root);
isub = [dir_content(:).isdir]; %# returns logical vector
nameFolds = {dir_content(isub).name}';
for subfold=1:length(nameFolds)
    if ~strcmp(nameFolds(subfold), '.') && ~strcmp(nameFolds(subfold), '..')
        root = fullfile(activation_fig_root, nameFolds(subfold));
        try
            handlePlotBrain(fullfile(root, "coords.csv"), fullfile(root, "coords_colors.csv"), ...
                fullfile(root, "coords_rois.csv"), fullfile(root, "rois_dict.csv"), root)
        catch
            continue
        end
    end
end
