%% Plotting parameters:
subject_dir = '/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs';
transparency = 0.3;
force_to_side = 'L';
roi_map = struct();
elec_size = 3;
perspective = 5;
activations = [];
sulc = [];
surface = 'pial';
parc = 'aparc.a2009s';
viewspoints = [-90, 0; 90, 0; -45, 0; 45, 0; -90, -45; -100, 10];
views = ["lateral", "medial", "back_lateral", "back_medial", "ventral", "top"];

%% Script looping over files:
addpath ..
activation_fig_root = "/hpc/users/alexander.lepauvre/plotting_test/pncc";
time_windows = ["200_300", "300_400", "400_500"];
for i=1:length(time_windows)
    time_win = time_windows(i);
    coords_file = fullfile(activation_fig_root, "coords_" + time_win + ".csv");
    colors_file = fullfile(activation_fig_root, "coords_colors_" + time_win + ".csv");
    % Load the data:
    % Load the coordinates:
    coords_tbl = readtable(coords_file);
    % Load the colorss:
    colors = readtable(colors_file);
    % Extract the coords:
    coords = table2array(coords_tbl(:, ["x", "y", "z"]));
    colors = table2array(colors(:, ["r", "g", "b"]));
    elec_size = table2array(coords_tbl(:, ["size"]));
    % prepare the file name:
    save_file_png = fullfile(activation_fig_root, "putative_ncc" + time_win + "_%s.png");
    
    % Plot the brain in the specified view:
    for view=1:length(viewspoints)
        fig = figure('visible', 'off');
        plotBrain("hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
            "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
            "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
            "elec_type", 'sphere', "add_perspective", 10)
        % Save the figure:
        exportgraphics(fig, sprintf(save_file_png, views(view)),'Resolution',300)
    end
end
