function [h, fvcdat, faces, vertices] = plotBrain(varargin)

% 
%
% optional arguments (name/value pairing)
%  subject:         string, subject id [default: 'fsaverage']
%  subjects_dir:    freesurfer subjects directory [default is value of getenv('SUBJECTS_DIR') environment variable]
%  brain:           override subject and use structure containing brain.vert brain.faces
%  hemi:            string or cell array of hemispheres (e.g., hemi={'lh', 'rh'} )[default: 'lh']
%  viewpoint:       camera position (default: [-180 0])
%  coords:          electrode coordinates
%  transparency:    brain transparency [0 opaque - 1 transparency] (default: 0.9)
%  brain_color:     base brain color (default: [0.58 0.58 0.58]);
%  sulc_color:      base sulcus color (default: [0.42 0.42 0.42]);
%  parc:            parcellation atlas
%  elec_color:      color of electrodes (default: [0.2 0.2 0.2])
%  elec_size:       size of electrodes (default: 40)
%  force_to_nearest_vertex:    force electrodes to nearest vertex of brain surface
%  force_to_side:   force electrodes to a hemisphere, e.g. 'L' or 'R' (default: none)
%  activations:     vector of electrode activations, size of coords x 1. The values are linearly mapped to the colors in the current colormapcolor code electrodes based on value)
%  activation_distance: numeric, distance that each activation spreads (mm) [default: 10] 
%  activation_scaling: string, how to scale activations over the distance (e.g., 'linear', 'gaussian', 'exponential') [default: 'linear'] 
%  activation_method: string or function definition for how to combine values across vertices [defailt: @mean]
%  activation_show_electrodes: show/hide electrodes from activations [default: false]
%  color_map:       specify a colormap (default: jet)
%  color_bar:       show colorbar (default: false)
%  scanner2tkr:     scanner2tkr transform
%  add_perspective: pull electrodes towards the camera (in mm) [default: 0]
%  update_lightsource:  update lighting on rotation

% parse arguments
p = inputParser;
addOptional(p, 'brain', [], @(x) (ischar(x) || iscell(x) || isstruct(x)));
addOptional(p, 'subjects_dir', [], @(x) (ischar(x) || iscell(x) || isstruct(x)));
addOptional(p, 'subject', 'fsaverage', @(x) (ischar(x) || iscell(x)) );
addOptional(p, 'surface', 'pial', @(x) (ischar(x) || iscell(x) || isstruct(x)));
addOptional(p, 'hemi', 'lh', @(x) (ischar(x) || iscell(x)));
addOptional(p, 'sulc', 'curv', @(x) (isempty(x) || ischar(x) || iscell(x) || isstruct(x)));
addOptional(p, 'viewpoint', [-90 0], @isnumeric);
addOptional(p, 'coords', [], @isnumeric);
addOptional(p, 'transparency', 0, @isnumeric);
addOptional(p, 'brain_color', [0.58 0.58 0.58], @isnumeric);
addOptional(p, 'sulc_color', [0.42 0.42 0.42], @isnumeric);
addOptional(p, 'parc', [], @(x) (ischar(x)));
addOptional(p, 'elec_color', [0.2 0.2 0.2], @isnumeric);
addOptional(p, 'elec_size', 3, @isnumeric);
addOptional(p, 'elec_type', 'scatter',  @(x) ( isstring(x) || ischar(x) ) );
addOptional(p, 'force_to_nearest_vertex', false);
addOptional(p, 'force_to_side', '', @ischar);
addOptional(p, 'force_to_nearest_roi', []);
addOptional(p, 'roi_map', [], @isstruct);
addOptional(p, 'activations', [], @isnumeric);
addOptional(p, 'activation_distance', 10, @isnumeric);
addOptional(p, 'activation_scaling', 'linear', @isnumeric);
addOptional(p, 'activation_method', 'nanmean');
addOptional(p, 'activation_show_electrodes', false, @islogical);
addOptional(p, 'roi_color', [], @isstruct); % roi_color.anatomy = string, roi_color.color = [r b g];
addOptional(p, 'color_map', 'RdYlBu');
addOptional(p, 'color_bar', false);
addOptional(p, 'colorbar_title', [], @ischar);
addOptional(p, 'colorbar_title_fontsize', 16, @isnumeric);
addOptional(p, 'scanner2tkr', [0 0 0], @isnumeric);
addOptional(p, 'remove_distant_elecs', false);
addOptional(p, 'add_perspective', 0, @isnumeric);
addOptional(p, 'update_lighting', false, @islogical );
addOptional(p, 'labels', []);


parse(p, varargin{:});
for v = fieldnames(p.Results)',
    eval([ v{:} '= p.Results.( v{:} );']);
end

% some initial checks
if ~isempty(roi_map), assert( ~isempty(parc), 'parcellation required if using roi_map.'); end
if isempty(subjects_dir) & isempty(brain), try, subjects_dir = getenv('SUBJECTS_DIR'); catch, error('Could not locate freesurfer SUBJECTS_DIR'); end, end
if ~isempty(activations), assert(size(activations,1) == size(coords,1), 'activations vector must be the same size as coords.'); end
if strcmp( eval('surface'), 'inflated'), force_to_nearest_vertex = true; end


if ~exist('brain','var')|isempty(brain),
    % get freesurfer directory
    if isempty(subjects_dir),
        % try to locate it
        if exist('/Applications/freesurfer/', 'dir'),
            folders = dir('/Applications/freesurfer/*');
            for j=1:length(folders),
                if folders(j).isdir & exist(fullfile(folders(j).folder, folders(j).name, 'subjects'), 'dir'),
                    subjects_dir = fullfile(folders(j).folder, folders(j).name, 'subjects');
                end
            end
        end
    end
    if ~exist(subjects_dir, 'dir') || isempty(subjects_dir),
        error('Could not locate freesurfer directory')
    end

    % load in the appropriate surfaces
    if ischar(hemi), hemi = {hemi}; end
    faces = [];
    vertices = [];
    vertex_inds = [0 0];
    for j=1:length(hemi),
        tmp = fs_read_surf( fullfile(subjects_dir, subject, 'surf', [hemi{j} '.' eval('surface')]) );
        faces = cat(1, faces, tmp.faces+length(vertices));
        vertex_inds(j+1, :) = [vertex_inds(j,2)+1 vertex_inds(j,2)+size(tmp.vertices,1)];
        vertices = cat(1, vertices, tmp.vertices + scanner2tkr);
    end
    vertex_inds(1, :) = [];

else
    % load local brain files
    % get faces/vertices
    if isstruct(brain),
        if isfield(brain, 'tri'),
            faces = brain.tri;
            vertices = brain.vert;
        elseif isfield(brain, 'faces'),
            faces = brain.faces;
            vertices = brain.coords;
        end
    elseif iscell(brain),
        faces = [];
        vertices = [];
        for j=1:length(brain),
            tmp = fs_read_surf(brain{j});
            faces = cat(1, faces, tmp.faces+length(vertices));
            vertices = cat(1, vertices, tmp.vertices + scanner2tkr);
        end
        %     subj_ = ft_read_headshape(brain{j});
        %     faces = subj_.tri;
        %     vertices = subj_.pos;
    else
        tmp = fs_read_surf(brain);
        faces = tmp.faces;
        vertices = tmp.vertices + scanner2tkr;
    end
end

% initialize brain color
fvcdat = repmat(brain_color(:)', [size(vertices, 1) 1]);

% add initial coloring based on curvature file (if present)
if ~isempty(sulc),

    curv = [];
    for j=1:length(hemi),
        curv_file = fullfile(subjects_dir, subject, 'surf', [hemi{j} '.curv']);
        [a, b] = read_curv( curv_file );
        curv = [curv; a(:)];
    end
    % apply darker shading to sulci
    fvcdat( (curv > 0), :) = repmat(sulc_color, sum((curv>0)), 1);
end

boundaries = {};
boundaries_color = {};
if ~isempty(parc)

    % load in parc for each hemi
    for j=1:length(hemi),


        [~,albl,actbl]=fs_read_annotation(  fullfile(subjects_dir, subject, 'label', [hemi{j} '.' parc '.annot']) );

        % special case to color your own rois
        if ~isempty(roi_map),
%             actbl.table(:,1:3)=repmat(255*[1 1 1]*.7, length(actbl.struct_names),1); %make all rois grey at first
            for r=1:length(roi_map),
                if ~isnan(roi_map(r).color),
                    idx_ = find(strcmp(actbl.struct_names, roi_map(r).anatomy));
%                     actbl.table(idx_,1:3) = roi_map(r).color;
                    ind = ismember(albl,actbl.table(idx_,5));
                    fvcdat( ind, :) = repmat(roi_map(r).color(:)', sum(ind), 1);
                    if isfield(roi_map, 'edge_color')&~isempty(roi_map(r).edge_color),
%                         % get the boundary of the roi
%                         % TODO: make this work when both hemis are plotted
%                         % (needs a special vertices, faces handling)
%                         ind = find(ind);
%                         k = boundary( vertices(ind, :) );
%                         fvcdat( ind(k(:,1)), :) = repmat( roi_map(r).edge_color, size(k,1), 1);
                        vertex_id = zeros( size(vertices,1), 1);
                        vertex_id( (albl == actbl.table(strcmp(actbl.struct_names, roi_map(r).anatomy), end)) ) = 1;
                        b = findROIboundaries(vertices, faces, vertex_id, 'edge_faces');
                        boundaries = cat(2, boundaries, b);
                        boundaries_color = cat(2, boundaries_color, repmat( {roi_map(r).edge_color}, 1, length(b )) );

                    end
                end
            end
%             [~,locTable]=ismember(albl,actbl.table(:,5));
%             locTable(locTable==0)=1; % for the case where the label for the vertex says 0
%             fvcdat=cat(1, fvcdat, actbl.table(locTable,1:3)./255); %scale RGB values to 0-1
        else

%             actbl.table(43,1:3)=255*[1 1 1]*.7; %make medial wall the same shade of grey as functional plots
            actbl.table(1,1:3)=255*[1 1 1]*.7; %make medial wall the same shade of grey as functional plots
            [~,locTable]=ismember(albl,actbl.table(:,5));
            locTable(locTable==0)=1; % for the case where the label for the vertex says 0


            fvcdat(vertex_inds(j, 1):vertex_inds(j, 2), :)=actbl.table(locTable,1:3)./255; %scale RGB values to 0-1
        end
    end


end

% add electrode activations
if ~isempty(activations),
    % make sure we have pial surface for localizing initial electrode position
    if strcmp(eval('surface'), 'pial'),
        vertices_pial = vertices;
    else
        vertices_pial = [];
        for j=1:length(hemi),
            tmp = fs_read_surf( fullfile(subjects_dir, subject, 'surf', [hemi{j} '.pial']) );
            vertices_pial = cat(1, vertices_pial, tmp.vertices + scanner2tkr);
        end
    end


    % initialize a keymap (for version <2022b)
    V = containers.Map( 'KeyType', 'double', 'ValueType', 'any');
    % store min.max vals
    mn = []; mx = [];
    fvcdat_acts = nan( size(fvcdat,1), size(coords,1) );
    for j=1:size(coords,1),
        % find nearest vertex
        val = activations(j);
        [~, idx_vertex] = pdist2(vertices_pial, coords(j, :), 'euclidean', 'Smallest', 1);
        idx_nbrs  = find( pdist2(vertices_pial, vertices_pial(idx_vertex, :)) < activation_distance );

        dist = pdist2(vertices_pial, vertices_pial(idx_nbrs, :), 'euclidean', 'Smallest', 1);

        switch activation_scaling,
                case 'linear',
                    vertex_val = val.*(1 - dist ./ activation_distance);
                case 'gaussian',
                    vertex_val = val.*(1 - dist ./ activation_distance);
                case 'exponential',
                    vertex_val = val.*(1 - dist ./ activation_distance);
        end

        fvcdat_acts(idx_nbrs, j) = vertex_val;

%         for idx_ = idx_nbrs(:)',
%             dist  = pdist2( vertices_pial(idx_vertex, :), vertices_pial(idx_, :) );
%             switch activation_scaling,
%                 case 'linear',
%                     vertex_val = val*(1 - dist / activation_distance);
%                 case 'gaussian',
%                     vertex_val = val*(1 - dist / activation_distance);
%                 case 'exponential',
%                     vertex_val = val*(1 - dist / activation_distance);
%             end
%
%             if ~isempty(V.keys) & sum(ismember( cell2mat( V.keys ), idx_ )),
%                 V( idx_ ) = [V( idx_ ) vertex_val];
%             else
%                 V( idx_ ) = vertex_val;
%             end
%
%             if isempty(mn),
%                 mn = vertex_val;
%                 mx = vertex_val;
%             else
%                 mn = min(mn, vertex_val);
%                 mx = max(mx, vertex_val);
%             end
%         end
    end
    fvcdat_acts = feval( activation_method, fvcdat_acts')';
    mn = min(fvcdat_acts); mx = max(fvcdat_acts);

    % create normalized map values
    if strcmp(color_map(end-1:end), '_r'),
        color_map = ['*' color_map(1:end-2)];
    end
    cm = brewermap(128, color_map);
    cvals = linspace( mn, mx, size(cm,1) );

%     % add to fvcdat
%     keys = V.keys;
%     for j=1:length(keys),
%         val = feval( activation_method, V( keys{j}) );
%         [~,val] = min( abs(cvals - val) );
%         fvcdat(keys{j}, :) = cm( val, :);
%     end


    idx_ = find( ~isnan(fvcdat_acts) );
    for ii = 1:length(idx_),
        val = fvcdat_acts(ii);
        [~,val] = min( abs(cvals - val) );
        fvcdat( idx_(ii), :) = cm( val, :);
    end

end

h = trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), ...
    'FaceVertexCData', fvcdat, ...
    'FaceColor', 'flat', ...
    'FaceVertexAlphaData', .2 * ones(size(vertices, 1), 1), ...
    'Tag', 'brain', 'FaceAlpha',1-transparency);
     hold on;


% plot any roi boundaries
for i = 1:length(boundaries)
    plot3(boundaries{i}(:,1), boundaries{i}(:,2), boundaries{i}(:,3), 'Color', boundaries_color{i}, 'LineWidth',2,'Clipping','off');
end

% Set viewing params
shading interp;
lighting flat;
material dull;
axis off;
axis equal;
camproj('orthographic');

view(viewpoint);
% delete(findobj(gca, 'type', 'light'));
LightSource = camlight('headlight', 'infinite');
% camlight(viewpoint(1), viewpoint(2));

% LightSource = lightangle(viewpoint(1), -viewpoint(2));
% h.FaceLighting = 'gouraud';
% h.AmbientStrength = 0.35;
% h.DiffuseStrength = 0.9;
% h.SpecularStrength = 0;
% h.SpecularExponent = 25;
% h.BackFaceLighting = 'unlit';

if ~isempty(coords),

    if force_to_side,
        switch force_to_side,
            case 'L'
                coords(:, 1) = -abs(coords(:, 1));
            case 'R'
                coords(:, 1) = abs(coords(:, 1));
            otherwise,
                error('wrong argument force_to_side')
        end
    end
    if remove_distant_elecs,
        vertices = get(h, 'Vertices');
        [D wh_closest] = pdist2(vertices, coords, 'euclidean', 'Smallest', 1);
        idx_ = find( (D' > 20) & (abs(coords(:, 1))>30) );
        coords(idx_, :) = [];
        elec_color(idx_, :) = [];
        if ~isempty(force_to_nearest_roi),
            force_to_nearest_roi(idx_) = [];
        end
    end
    if force_to_nearest_vertex,
        % get vertex normals
        vertices = get(h, 'Vertices');
        if ~strcmp(eval('surface'), 'pial'),
            vertices_pial = [];
            for j=1:length(hemi),
                tmp = fs_read_surf( fullfile(subjects_dir, subject, 'surf', [hemi{j} '.pial']) );
                vertices_pial = cat(1, vertices_pial, tmp.vertices + scanner2tkr);
            end
        else
            vertices_pial = vertices;
        end
        [D wh_closest] = pdist2(vertices_pial, coords, 'euclidean', 'Smallest', 1);
        coords = vertices(wh_closest,:);
%         % pull electrodes towards camera position so they are not obscured by the mesh
%         cam_pos = get(gca, 'CameraPosition');
%         source_coordinates = [0 0 0];
%         direction_vector = cam_pos - source_coordinates;
%         direction_vector = direction_vector / norm(direction_vector) * pull_length;
%         coords(:, 1) = coords(:, 1)+direction_vector(1);
%         coords(:, 2) = coords(:, 2)+direction_vector(2);
%         coords(:, 3) = coords(:, 3)+direction_vector(3);
    end

    if ~isempty(force_to_nearest_roi),
        vertices = get(h, 'Vertices');
        % load in parcelation
        if ~exist("actbl", 'var')|isempty(actbl),
            % todo load in anot file for each hemi
            [~,albl,actbl]=fs_read_annotation(  fullfile(subjects_dir, subject, 'label', [hemi{1} '.' parc '.annot']) );
        end
        vertices_pial = vertices;

        for r=1:length(force_to_nearest_roi),
            verts = find( albl == actbl.table( find(strcmpi(actbl.struct_names, force_to_nearest_roi{r} )), end) );
            roi_vertices = vertices_pial(verts, :);
            [D wh_closest] = pdist2(roi_vertices, coords(r, :), 'euclidean', 'Smallest', 1);
            coords(r, :) = roi_vertices(wh_closest, :);
        end


%         tmp = find( sum(ismember(fvcdat, cat(1, roi_map.color)),2)/3 );
%         roi_vertices = vertices_pial(tmp, :);
%         [D wh_closest] = pdist2(roi_vertices, coords, 'euclidean', 'Smallest', 1);
%         idx = find( D > 5 );
%         coords( idx, :) = roi_vertices(wh_closest(idx), :);

%         % pull electrodes towards camera position so they are not obscured by the mesh
%         cam_pos = get(gca, 'CameraPosition');
%         source_coordinates = [0 0 0];
%         direction_vector = cam_pos - source_coordinates;
%         direction_vector = direction_vector / norm(direction_vector) * pull_length;
%         coords(:, 1) = coords(:, 1)+direction_vector(1);
%         coords(:, 2) = coords(:, 2)+direction_vector(2);
%         coords(:, 3) = coords(:, 3)+direction_vector(3);
    end
%

    % add some depth perspective
    if add_perspective,
        cam_pos = get(gca, 'CameraPosition');
        source_coordinates = [0 0 0];
        direction_vector = cam_pos - coords;
%         direction_vector = cam_pos - source_coordinates;
        direction_vector = direction_vector ./ vecnorm(direction_vector, 2, 2) .* randi([0 add_perspective*100], size(coords,1), 1)./100;  %linspace(0, pull_length, size(coords,1))';
        coords = coords + direction_vector;
    end

%     normals = compute_elec_normals(coords, viewpoint, true);
%     theta = 0:0.2:2*pi;
%         costheta = cos(theta);
%         sintheta = sin(theta);

    if isempty(activations) || (~isempty(activations) && activation_show_electrodes),
        % set electrode colors
        if size(elec_color,1) ~= size(coords,1),
            C = repmat(elec_color, size(coords,1), 1);
        else
            C = elec_color;
        end

        switch elec_type,
            case 'scatter',
                scatter3( coords(:, 1), coords(:, 2), coords(:, 3), elec_size*20, C, 'filled', 'markeredgecolor', [0 0 0]);
            case {'sphere', 'spheres'}
                [X,Y,Z] = sphere;
                radius = elec_size;
                if numel(radius) == 1,
                    radius = repmat(radius, size(coords,1), 1);
                end
%                 xx = []; yy = []; zz = [];
                for j=1:size(coords,1),
                    X_ = X * radius(j);
                    Y_ = Y * radius(j);
                    Z_ = Z * radius(j);
                    sp = surf(X_+coords(j,1),Y_+coords(j,2),Z_+coords(j,3));
                    set(sp,'FaceColor',C(j,:), 'FaceAlpha',0.9, 'FaceLighting', 'gouraud', 'EdgeColor', ['none']);
                    material(sp, [0.3 0.3 1 20 0.1]);
%                     material('dull');

%                       plot3(coords(j,1),coords(j,2),coords(j,3), 'ok', 'MarkerFaceColor', C(j,:));
%                     xx = [xx; X+coords(j,1)];
%                     yy = [yy; Y+coords(j,2)];
%                     zz = [zz; Z+coords(j,3)];
                end

%                 [f,v,c] = surf2patch(xx, yy, zz);
%                 trisurf(f, v(:, 1), v(:, 2), v(:, 3), 'EdgeAlpha', 0);
        end

        if ~isempty(labels),
            for j=1:size(coords,1),
                text(coords(j, 1)-5, coords(j, 2)-5, coords(j, 3), labels{j});
            end
        end
    end
end



if color_bar,
    cb = colorbar();
    colormap(cm);
    cb.Ticks = [0, 1];
    cb.Limits = [0 1];
    cb.TickLabels = cellstr(num2str(round([mn;mx], 2)));
    if ~isempty(colorbar_title),
        cb.Label.String = colorbar_title;
        cb.Label.FontSize = colorbar_title_fontsize;
    end
end

if update_lighting,
    h = rotate3d;                 % Create rotate3d-handle
    set(h,'ActionPostCallback',{@RotationCallback, LightSource});
end










% ---------   Utilities  ---------  %

% update lightsource on rotation
function RotationCallback(src,evt,LightSource)
        camlight(LightSource, 'headlight');

function [surf] = fs_read_surf(fname)
% fs_read_surf - read a freesurfer surface file
%
% [surf] = fs_read_surf(fname)
%
% Reads the vertex coordinates (mm) and face lists from a surface file
%
% surf is a structure containing:
%   nverts: number of vertices
%   nfaces: number of faces (triangles)
%   faces:  vertex numbers for each face (3 corners)
%   vertices: x,y,z coordinates for each vertex
%
% this is a modification of Darren Weber's freesurfer_read_surf
%   which was a modified version of freesurfer's read_surf
%
% see also: fs_read_trisurf, fs_find_neighbors, fs_calc_triarea
%
% created:        03/02/06 Don Hagler
% last modified:  03/31/10 Don Hagler
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%QUAD_FILE_MAGIC_NUMBER =  (-1 & 0x00ffffff) ;
%NEW_QUAD_FILE_MAGIC_NUMBER =  (-3 & 0x00ffffff) ;

TRIANGLE_FILE_MAGIC_NUMBER  =  16777214 ;
QUAD_FILE_MAGIC_NUMBER      =  16777215 ;

% open it as a big-endian file
fid = fopen(fname, 'rb', 'b') ;
if (fid < 0),
  error(sprintf('could not open surface file %s.',fname));
end

magic = fs_fread3(fid) ;

if (magic == QUAD_FILE_MAGIC_NUMBER),
  surf.nverts = fs_fread3(fid) ;
  surf.nfaces = fs_fread3(fid) ;
%  fprintf('%s: reading %d quad file vertices...',mfilename,surf.nverts); tic;
  surf.vertices = fread(fid, surf.nverts*3, 'int16') ./ 100 ;
%  t=toc; fprintf('done (%0.2f sec)\n',t);
%  fprintf('%s: reading %d quad file faces (please wait)...\n',...
%    mfilename,surf.nfaces); tic;
  surf.faces = zeros(surf.nfaces,4);
  for iface = 1:surf.nfaces,
    for n=1:4,
      surf.faces(iface,n) = fs_fread3(fid) ;
    end
%    if(~rem(iface, 10000)), fprintf(' %7.0f',iface); end
%    if(~rem(iface,100000)), fprintf('\n'); end
  end
%  t=toc; fprintf('\ndone (%0.2f sec)\n',t);
elseif (magic == TRIANGLE_FILE_MAGIC_NUMBER),
%  fprintf('%s: reading triangle file...',mfilename); tic;

  tline = fgets(fid); % read creation date text line
  tline = fgets(fid); % read info text line

  surf.nverts = fread(fid, 1, 'int32') ; % number of vertices
  surf.nfaces = fread(fid, 1, 'int32') ; % number of faces

  % vertices are read in column format and reshaped below
  surf.vertices = fread(fid, surf.nverts*3, 'float32');

  % faces are read in column format and reshaped
  surf.faces = fread(fid, surf.nfaces*3, 'int32') ;
  surf.faces = reshape(surf.faces, 3, surf.nfaces)' ;
%  t=toc; fprintf('done (%0.2f sec)\n',t);
else
  error(sprintf('unknown magic number in surface file %s.',fname));
end

surf.vertices = reshape(surf.vertices, 3, surf.nverts)' ;
fclose(fid) ;

%fprintf('...adding 1 to face indices for matlab compatibility.\n\n');
surf.faces = surf.faces + 1;

return


function [curv, fnum] = read_curv(fname)
%
% [curv, fnum] = read_curv(fname)
% reads a binary curvature file into a vector
%


%
% read_curv.m
%
% Original Author: Bruce Fischl
%
% Copyright ?? 2021 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%


%fid = fopen(fname, 'r') ;
%nvertices = fscanf(fid, '%d', 1);
%all = fscanf(fid, '%d %f %f %f %f\n', [5, nvertices]) ;
%curv = all(5, :)' ;

% open it as a big-endian file
fid = fopen(fname, 'rb', 'b') ;
if (fid < 0)
	 str = sprintf('could not open curvature file %s', fname) ;
	 error(str) ;
end
vnum = fread3(fid) ;
NEW_VERSION_MAGIC_NUMBER = 16777215;
if (vnum == NEW_VERSION_MAGIC_NUMBER)
	 vnum = fread(fid, 1, 'int32') ;
	 fnum = fread(fid, 1, 'int32') ;
	 vals_per_vertex = fread(fid, 1, 'int32') ;
   curv = fread(fid, vnum, 'float') ;

  fclose(fid) ;
else

	fnum = fread3(fid) ;
  curv = fread(fid, vnum, 'int16') ./ 100 ;
  fclose(fid) ;
end
%nvertices = fscanf(fid, '%d', 1);
%all = fscanf(fid, '%d %f %f %f %f\n', [5, nvertices]) ;
%curv = all(5, :)' ;



function [retval] = fread3(fid)
% [retval] = fd3(fid)
% read a 3 byte integer out of a file


%
% fread3.m
%
% Original Author: Bruce Fischl
%
% Copyright ?? 2021 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%

b1 = fread(fid, 1, 'uchar') ;
b2 = fread(fid, 1, 'uchar') ;
b3 = fread(fid, 1, 'uchar') ;
retval = bitshift(b1, 16) + bitshift(b2,8) + b3 ;





function [vertices, label, colortable] = fs_read_annotation(filename, varargin)
%
% NAME
%
%       function [vertices, label, colortable] = ...
%                                       read_annotation(filename [, verbosity])
%
% ARGUMENTS
% INPUT
%       filename        string          name of annotation file to read
%
% OPTIONAL
%       verbosity       int             if true (>0), disp running output
%                                       + if false (==0), be quiet and do not
%                                       + display any running output
%
% OUTPUT
%       vertices        vector          vector with values running from 0 to
%                                       + size(vertices)-1
%       label           vector          lookup of annotation values for 
%                                       + corresponding vertex index.
%       colortable      struct          structure of annotation data
%                                       + see below
%       
% DESCRIPTION
%
%       This function essentially reads in a FreeSurfer annotation file
%       <filename> and returns structures and vectors that together 
%       assign each index in the surface vector to one of several 
%       structure names.
%       
% COLORTABLE STRUCTURE
% 
%       Consists of the following fields:
%       o numEntries:   number of entries
%       o orig_tab:     filename of original colortable file
%       o struct_names: cell array of structure names
%       o table:        n x 5 matrix
%                       Columns 1,2,3 are RGB values for struct color
%                       Column 4 is a flag (usually 0)
%                       Column 5 is the structure ID, calculated from
%                       R + G*2^8 + B*2^16 + flag*2^24
%                       
% LABEL VECTOR
% 
%       Each component of the <label> vector has a structureID value. To
%       match the structureID value with a structure name, lookup the row
%       index of the structureID in the 5th column of the colortable.table
%       matrix. Use this index as an offset into the struct_names field
%       to match the structureID with a string name.      
%
% PRECONDITIONS
%
%       o <filename> must be a valid FreeSurfer annotation file.
%       
% POSTCONDITIONS
%
%       o <colortable> will be an empty struct if not embedded in a
%         FreeSurfer annotation file. 
%       

%
% read_annotation.m
% Original Author: Bruce Fischl
% CVS Revision Info:
%    $Author: nicks $
%    $Date: 2011/03/02 00:04:12 $
%    $Revision: 1.7 $
%
% Copyright ?? 2011 The General Hospital Corporation (Boston, MA) "MGH"
%
% Terms and conditions for use, reproduction, distribution and contribution
% are found in the 'FreeSurfer Software License Agreement' contained
% in the file 'LICENSE' found in the FreeSurfer distribution, and here:
%
% https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
%
% Reporting: freesurfer@nmr.mgh.harvard.edu
%
%

fp = fopen(filename, 'r', 'b');

verbosity = 1;

if ~isempty(varargin)
    verbosity       = varargin{1};  
end;

if(fp < 0)
   if verbosity, disp('Annotation file cannot be opened'); end;
   return;
end

A = fread(fp, 1, 'int');

tmp = fread(fp, 2*A, 'int');
vertices = tmp(1:2:end);
label = tmp(2:2:end);

bool = fread(fp, 1, 'int');
if(isempty(bool)) %means no colortable
   if verbosity, disp('No Colortable found.'); end;
   colortable = struct([]);
   fclose(fp);
   return; 
end

if(bool)
    
    %Read colortable
    numEntries = fread(fp, 1, 'int');

    if(numEntries > 0)
        
        if verbosity, disp('Reading from Original Version'); end;
        colortable.numEntries = numEntries;
        len = fread(fp, 1, 'int');
        colortable.orig_tab = fread(fp, len, '*char')';
        colortable.orig_tab = colortable.orig_tab(1:end-1);

        colortable.struct_names = cell(numEntries,1);
        colortable.table = zeros(numEntries,5);
        for i = 1:numEntries
            len = fread(fp, 1, 'int');
            colortable.struct_names{i} = fread(fp, len, '*char')';
            colortable.struct_names{i} = colortable.struct_names{i}(1:end-1);
            colortable.table(i,1) = fread(fp, 1, 'int');
            colortable.table(i,2) = fread(fp, 1, 'int');
            colortable.table(i,3) = fread(fp, 1, 'int');
            colortable.table(i,4) = fread(fp, 1, 'int');
            colortable.table(i,5) = colortable.table(i,1) + colortable.table(i,2)*2^8 + colortable.table(i,3)*2^16 + colortable.table(i,4)*2^24;
        end
        if verbosity
            disp(['colortable with ' num2str(colortable.numEntries) ' entries read (originally ' colortable.orig_tab ')']);
        end
    else
        version = -numEntries;
        if verbosity
          if(version~=2)    
            disp(['Error! Does not handle version ' num2str(version)]);
          else
            disp(['Reading from version ' num2str(version)]);
          end
        end
        numEntries = fread(fp, 1, 'int');
        colortable.numEntries = numEntries;
        len = fread(fp, 1, 'int');
        colortable.orig_tab = fread(fp, len, '*char')';
        colortable.orig_tab = colortable.orig_tab(1:end-1);
        
        colortable.struct_names = cell(numEntries,1);
        colortable.table = zeros(numEntries,5);
        
        numEntriesToRead = fread(fp, 1, 'int');
        for i = 1:numEntriesToRead
            structure = fread(fp, 1, 'int')+1;
            if (structure < 0)
              if verbosity, disp(['Error! Read entry, index ' num2str(structure)]); end;
            end
            if(~isempty(colortable.struct_names{structure}))
              if verbosity, disp(['Error! Duplicate Structure ' num2str(structure)]); end;
            end
            len = fread(fp, 1, 'int');
            colortable.struct_names{structure} = fread(fp, len, '*char')';
            colortable.struct_names{structure} = colortable.struct_names{structure}(1:end-1);
            colortable.table(structure,1) = fread(fp, 1, 'int');
            colortable.table(structure,2) = fread(fp, 1, 'int');
            colortable.table(structure,3) = fread(fp, 1, 'int');
            colortable.table(structure,4) = fread(fp, 1, 'int');
            colortable.table(structure,5) = colortable.table(structure,1) + colortable.table(structure,2)*2^8 + colortable.table(structure,3)*2^16 + colortable.table(structure,4)*2^24;       
        end
        if verbosity 
          disp(['colortable with ' num2str(colortable.numEntries) ' entries read (originally ' colortable.orig_tab ')']);
        end
    end    
else
    if verbosity
        disp('Error! Should not be expecting bool = 0');    
    end;
end

fclose(fp);


