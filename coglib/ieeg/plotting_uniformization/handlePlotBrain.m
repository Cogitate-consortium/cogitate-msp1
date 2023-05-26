function [] = handlePlotBrain(coords_file, coords_colors, coords_rois, roi_dict, save_dir, varargin)
%% Handle optional parameters:
p = inputParser;
addOptional(p, 'file_suffix', '', @ischar);
addOptional(p, 'edge_color', '', @isstring);
parse(p, varargin{:});
for v = fieldnames(p.Results)',
    eval([ v{:} '= p.Results.( v{:} );']);
end

%% Set the default parameters:
subject_dir = '/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs';
transparency = 0;
force_to_side = 'L';
roi_map = struct();
elec_size = 3;
perspective = 10;
activations = [];
sulc = [];
surface = 'pial';
parc = 'aparc.a2009s';
viewspoints = [-90, 0; 90, 0; -45, 0; 45, 0; -90, -100; -100, 10; 30, -22];
views = ["lateral", "medial", "back_lateral", "back_medial", "ventral", "top", "back"];


%% Handle the different files:
% Load the coordinates:
coords_tbl = readtable(coords_file);
% Load the colors:
colors = readtable(coords_colors);
rois = readtable(coords_rois);
% Format the coordinates and the colors:
coords_tbl = sortrows(coords_tbl, "channel");
colors = sortrows(colors, "channel");
rois = sortrows(rois, "channel");
% Extract the coords:
coords = table2array(coords_tbl(:, ["x", "y", "z"]));
colors = table2array(colors(:, ["r", "g", "b"]));
elec_size = table2array(coords_tbl(:, ["radius"]));
% Get the coords rois:
rois = table2cell(rois(:, ["roi"]))';
% Load the ROI dict:
roi_tbl = readtable(roi_dict);
roi_map = struct('anatomy', [], 'color', []);
for i=1:height(roi_tbl)
    roi_map(i).anatomy = roi_tbl.roi(i);
    roi_map(i).color = [roi_tbl.r(i), roi_tbl.g(i), roi_tbl.b(i)];
end
if isempty(edge_color)
    edge_color = [];
else
    edge_color = readtable(edge_color);
    edge_color = sortrows(edge_color, "channel");
    edge_color = table2array(edge_color(:, ["r", "g", "b"]));
end

%% Call plot brain
save_file_png = fullfile(save_dir, "brain_plot_%s_%s.png");
save_file_svg = fullfile(save_dir, "brain_plot_%s_%s.svg");
for view=1:length(viewspoints)
    fig = figure('visible', 'off');
    if isempty(edge_color)
        plotBrain("hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
            "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
            "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
            "parc", parc, "roi_map", roi_map, "elec_type", 'sphere', "force_to_nearest_roi", rois, "add_perspective", 10)
    else
        plotBrain("hemi", 'lh', "coords", coords*1e3, "surface", surface, ...
            "transparency", transparency, "elec_color", colors, "force_to_nearest_vertex", true, ...
            "force_to_side", force_to_side, "elec_size", elec_size, "viewpoint", viewspoints(view, :), ...
            "parc", parc, "roi_map", roi_map, "elec_type", 'scatter', "force_to_nearest_roi", rois, "add_perspective", 10, "EdgeColor", edge_color)
    end
    exportgraphics(fig, sprintf(save_file_png, views(view), file_suffix),'Resolution',300)
    saveas(fig, sprintf(save_file_svg, views(view), file_suffix))
    close
end
end