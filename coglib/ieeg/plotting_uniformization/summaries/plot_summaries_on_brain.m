%% Set parameters:
addpath ..
subject_dir = 'C:\Users\alexander.lepauvre\Documents\CoG_Der\BIDS\derivatives\fs';
transparency = 0;
force_to_side = 'L';
perspective = 20;
activations = [];
sulc = [];
surface = 'pial';
parc = 'aparc.a2009s';
viewspoints = [-90, 0; 90, 0; -45, 0; 45, 0; -90, -100; -100, 10; 30, -22];
views = ["lateral", "medial", "back_lateral", "back_medial", "ventral", "top", "back"];

% Load the different ROIs:
theory_roi_table = 'C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\theory_rois_dict.csv';
% % Load the ROI dict:
theory_roi_tbl = readtable(theory_roi_table);
theory_roi_map = struct('anatomy', [], 'color', []);
for i=1:height(theory_roi_tbl)
    theory_roi_map(i).anatomy = theory_roi_tbl.roi(i);
    theory_roi_map(i).color = [theory_roi_tbl.r(i), theory_roi_tbl.g(i), theory_roi_tbl.b(i)];
end

iit_roi_table = 'C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\iit_rois_dict.csv';
% % Load the ROI dict:
iit_roi_tbl = readtable(iit_roi_table);
iit_roi_map = struct('anatomy', [], 'color', []);
for i=1:height(iit_roi_tbl)
    iit_roi_map(i).anatomy = iit_roi_tbl.roi(i);
    iit_roi_map(i).color = [iit_roi_tbl.r(i), iit_roi_tbl.g(i), iit_roi_tbl.b(i)];
end

anat_rois_table = 'C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\anatomical_rois_dict.csv';
% % Load the ROI dict:
anat_roi_tbl = readtable(anat_rois_table);
anat_roi_map = struct('anatomy', [], 'color', []);
for i=1:height(anat_roi_tbl)
    anat_roi_map(i).anatomy = anat_roi_tbl.roi(i);
    anat_roi_map(i).color = [anat_roi_tbl.r(i) * 0.75, anat_roi_tbl.g(i) * 0.75, anat_roi_tbl.b(i) * 0.75];
end

%% Responsiveness:
coord_file = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Onset_Responsiveness\brain_plots\responsiveness_coords.csv";
colors_ti = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Onset_Responsiveness\brain_plots\responsiveness_coords_colors_ti.csv";
colors_tr = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Onset_Responsiveness\brain_plots\responsiveness_coords_colors_tr.csv";
coords_tbl = readtable(coord_file);
colors_ti = readtable(colors_ti);
colors_tr = readtable(colors_tr);
% Sort the rows to make sure we don't have any misalignment:
coords_tbl = sortrows(coords_tbl, "channel");
colors_ti = sortrows(colors_ti, "channel");
colors_tr = sortrows(colors_tr, "channel");

% Plot the task relevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_tr(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\Onset_Responsiveness\\brain_plots\\responsiveness_brain_plot_%s_tr.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", 'aparc', "roi_map", anat_roi_map, "elec_type", 'scatter', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

% Plot the task irrelevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_ti(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\Onset_Responsiveness\\brain_plots\\responsiveness_brain_plot_%s_ti.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", 'aparc', "roi_map", anat_roi_map, "elec_type", 'scatter', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end


%% Category selectivity:
coord_file = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Category_Selectivity\brain_plots\category_selectivity_coords_ti.csv";
colors_ti = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Category_Selectivity\brain_plots\category_selectivity_coords_colors_ti.csv";
colors_tr = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\Category_Selectivity\brain_plots\category_selectivity_coords_colors_tr.csv";
coords_tbl = readtable(coord_file);
colors_ti = readtable(colors_ti);
colors_tr = readtable(colors_tr);
% Sort the rows to make sure we don't have any misalignment:
coords_tbl = sortrows(coords_tbl, "channel");
colors_ti = sortrows(colors_ti, "channel");
colors_tr = sortrows(colors_tr, "channel");

% Plot the task relevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_tr(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\Category_Selectivity\\brain_plots\\selectivity_brain_plot_%s_ti.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", theory_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

% Plot the task irrelevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_ti(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\Category_Selectivity\\brain_plots\\seletivity_brain_plot_%s_tr.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", theory_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end


%% Duration decoding:
coord_file = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_decoding\duration_decoding_coords.csv";
colors_ti = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_decoding\duration_decoding_coords_colors_ti.csv";
colors_tr = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_decoding\duration_decoding_coords_colors_tr.csv";
coords_tbl = readtable(coord_file);
colors_ti = readtable(colors_ti);
colors_tr = readtable(colors_tr);
% Sort the rows to make sure we don't have any misalignment:
coords_tbl = sortrows(coords_tbl, "channel");
colors_ti = sortrows(colors_ti, "channel");
colors_tr = sortrows(colors_tr, "channel");
% Plot the task relevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_tr(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\duration_decoding\\brain_plots_decoding\\duration_decoding_brain_plot_%s_ti.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", theory_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

% Plot the task irrelevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_ti(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\duration_decoding\\brain_plots_decoding\\duration_decoding__brain_plot_%s_tr.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", theory_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

%% Duration tracking:
coord_file = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_tracking\duration_tracking_coords.csv";
colors_ti = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_tracking\duration_tracking_coords_colors_ti.csv";
colors_tr = "C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\duration_decoding\brain_plots_tracking\duration_tracking_coords_colors_tr.csv";
coords_tbl = readtable(coord_file);
colors_ti = readtable(colors_ti);
colors_tr = readtable(colors_tr);
% Sort the rows to make sure we don't have any misalignment:
coords_tbl = sortrows(coords_tbl, "channel");
colors_ti = sortrows(colors_ti, "channel");
colors_tr = sortrows(colors_tr, "channel");

% Plot the task relevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_tr(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\duration_decoding\\brain_plots_tracking\\duration_decoding_brain_plot_%s_ti.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", iit_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

% Plot the task irrelevant:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors_ti(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\duration_decoding\\brain_plots_tracking\\duration_decoding__brain_plot_%s_tr.png";
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
        "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
        "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
        "parc", parc, "roi_map", iit_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
    saveas(fig, sprintf(save_file_png, views(view)))
    close
end

%% Activation:
coord_files = "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\activation_analysis_supplementary\\brain_plots\\activation_%s_%s_coords.csv";
colors_files = "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\activation_analysis_supplementary\\brain_plots\\activation_%s_%s_colors.csv";
tasks = ["ti", "tr"];
signals = ["HGP" "alpha" "ERP"];
for task_i = 1:length(tasks)
    task = tasks(task_i);
    for signal_i = 1:length(signals)
        signal = signals(signal_i);
        % Get the data:
        coord_file = sprintf(coord_files, signal, task);
        color_file = sprintf(colors_files, signal, task);
        coords_tbl = readtable(coord_file);
        colors_tbl = readtable(color_file);
        % Sort the rows to make sure we don't have any misalignment:
        coords_tbl = sortrows(coords_tbl, "channel");
        colors_tbl = sortrows(colors_tbl, "channel");
        coords = table2array(coords_tbl(:, ["x", "y", "z"]));
        colors = table2array(colors_tbl(:, ["r", "g", "b"]));
        elec_size = table2array(coords_tbl(:, ["radius"]));
        % Set save file:
        save_file_png =  "C:\\Users\\alexander.lepauvre\\Documents\\PhD\\COGITATE\\activation_analysis_supplementary\\brain_plots\\activation_brain_plot_%s_%s_%s.png";
        for view=1:length(viewspoints)
            fig = figure('visible', 'off');
            plotBrain("subjects_dir", subject_dir,"hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
                "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
                "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
                "parc", parc, "roi_map", theory_roi_map, "elec_type", 'sphere', "add_perspective", perspective, "remove_distant_elecs", true)
            saveas(fig, sprintf(save_file_png, task, signal, views(view)))
            close
        end
    end
end
