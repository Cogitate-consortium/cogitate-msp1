

gnw_els = readtable('gnw_channels.csv');
iit_els = readtable('iit_channels.csv');


% rois to map
gnw = {"ctx_lh_G_and_S_cingul-Ant",
"ctx_rh_G_and_S_cingul-Ant",
"ctx_lh_G_and_S_cingul-Mid-Ant",
"ctx_rh_G_and_S_cingul-Mid-Ant",
"ctx_lh_G_and_S_cingul-Mid-Post",
"ctx_rh_G_and_S_cingul-Mid-Post",
"ctx_lh_G_front_inf-Opercular",
"ctx_rh_G_front_inf-Opercular",
"ctx_lh_G_front_inf-Orbital",
"ctx_rh_G_front_inf-Orbital",
"ctx_lh_G_front_inf-Triangul",
"ctx_rh_G_front_inf-Triangul",
"ctx_lh_G_front_middle",
"ctx_rh_G_front_middle",
"ctx_lh_Lat_Fis-ant-Horizont",
"ctx_rh_Lat_Fis-ant-Horizont",
"ctx_lh_Lat_Fis-ant-Vertical",
"ctx_rh_Lat_Fis-ant-Vertical",
"ctx_lh_S_front_inf",
"ctx_rh_S_front_inf",
"ctx_lh_S_front_middle",
"ctx_rh_S_front_middle",
"ctx_lh_S_front_sup",
"ctx_rh_S_front_sup"};
gnw = cellfun( @(x) x{:}, gnw, 'UniformOutput', false)
gnw = regexprep(gnw, {'ctx_lh_', 'ctx_rh_'}, '');
gnw = unique(gnw);


iit = {"ctx_lh_G_temporal_inf",
            "ctx_rh_G_temporal_inf",
            "ctx_lh_Pole_temporal",
            "ctx_rh_Pole_temporal",
            "ctx_lh_G_cuneus",
            "ctx_rh_G_cuneus",
            "ctx_lh_G_occipital_sup",
            "ctx_rh_G_occipital_sup",
            "ctx_lh_G_oc-temp_med-Lingual",
            "ctx_rh_G_oc-temp_med-Lingual",
            "ctx_lh_Pole_occipital",
            "ctx_rh_Pole_occipital",
            "ctx_lh_S_calcarine",
            "ctx_rh_S_calcarine",
            "ctx_lh_G_and_S_occipital_inf",
            "ctx_rh_G_and_S_occipital_inf",
            "ctx_lh_G_occipital_middle",
            "ctx_rh_G_occipital_middle",
            "ctx_lh_G_oc-temp_lat-fusifor",
            "ctx_rh_G_oc-temp_lat-fusifor",
            "ctx_lh_G_oc-temp_med-Parahip",
            "ctx_rh_G_oc-temp_med-Parahip",
            "ctx_lh_S_intrapariet_and_P_trans",
            "ctx_rh_S_intrapariet_and_P_trans",
            "ctx_lh_S_oc_middle_and_Lunatus",
            "ctx_rh_S_oc_middle_and_Lunatus",
            "ctx_lh_S_oc_sup_and_transversal",
            "ctx_rh_S_oc_sup_and_transversal",
            "ctx_lh_S_temporal_sup",
            "ctx_rh_S_temporal_sup"};
iit = cellfun( @(x) x{:}, iit, 'UniformOutput', false)
iit = regexprep(iit, {'ctx_lh_', 'ctx_rh_'}, '');
iit = unique(iit);

roi_map = struct('anatomy', gnw, 'color', [0.00784313725490196,0.6196078431372549,0.45098039215686275]);
roi_iit = struct('anatomy', iit, 'color', [0.00392156862745098,0.45098039215686275,0.6980392156862745]);
roi_map = [roi_map().' roi_iit(:).'];


colors = [0.6706    0.8667    0.6431; 0.1686    0.5137    0.7294];
els = [gnw_els.x gnw_els.y gnw_els.z; iit_els.x iit_els.y iit_els.z;].*1e3;
el_colors = [repmat(colors(1, :), length(gnw_els.x), 1); repmat(colors(2, :), length(iit_els.x), 1);];



% for each electrode get it's roi that maps to the gnw/iit list 
rois = cat(1, gnw_els.roi, iit_els.roi);
nearest_roi = {};
for r=1:length(rois),
     idx_ = find( contains({roi_map.anatomy}, regexprep( split( rois{r}, ',' ), {'[', ']', '''', ' ', 'ctx_lh_', 'ctx_rh_'}, {''}) ), 1);
     nearest_roi{r} = roi_map(idx_).anatomy;
end




%% basic pial plot 
figure;
plotBrain(hemi={'lh'}, surface='pial', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3);


%% force to nearest roi 
figure;
plotBrain(hemi={'lh'}, surface='pial', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3, force_to_nearest_roi=nearest_roi);

%% add some persepctive by "pulling" electrodes towards the camera (add_persepctive=10 , adds 10mm of random pull towards the camera)
figure;
plotBrain(hemi={'lh'}, surface='pial', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3, force_to_nearest_roi=nearest_roi, add_perspective=10);


%% works for inflated surfaces too!
%% basic pial plot 
figure;
plotBrain(hemi={'lh'}, surface='inflated', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3);


%% force to nearest roi 
figure;
plotBrain(hemi={'lh'}, surface='inflated', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3, force_to_nearest_roi=nearest_roi);

%% add some persepctive by "pulling" electrodes towards the camera (add_persepctive=10 , adds 10mm of random pull towards the camera)
figure;
plotBrain(hemi={'lh'}, surface='inflated', sulc='curv', force_to_side='L', elec_color=el_colors, coords=els, force_to_nearest_vertex=false, remove_distant_elecs=false, roi_map=roi_map, parc='aparc.a2009s', elec_type='sphere', elec_size=3, force_to_nearest_roi=nearest_roi, add_perspective=10);

