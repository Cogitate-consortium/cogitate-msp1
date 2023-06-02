clear
close all

% Select phase
%listType = 'phase2_V1';
listType = 'phase3_V1';

dataPath         = fullfile(pwd,'Data','putative_ncc',listType);
outPath          = fullfile(pwd,'Data','plots',listType);
sliceDisplayPath = fullfile(pwd,'slice_plotting','SliceDisplay');

mkdir(outPath);
addpath(genpath(sliceDisplayPath));


%% Subject, session and gFeat (contrast) folder & labels

fileName = 'pNCC_conjunction';

% standard space template brain (MNI T1 2mm)
mni     = fullfile(pwd,'Data','MNI152NLin2009cAsym_res-01_T1w.nii.gz');
mnimask = fullfile(pwd,'Data','MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz');

conList = {'C_combined_sum_Activation' 'C_combined_sum_Deactivation';
    'C_Face_Activation' 'C_Face_Deactivation';
    'C_Object_Activation' 'C_Object_Deactivation';
    'C_Letter_Activation' 'C_Letter_Deactivation';
    'C_FalseFont_Activation' 'C_FalseFont_Deactivation'};...
    
CustomColor
colorList = repmat({'Act' 'Deact'},size(conList,1),1);

opacity = .6;

for idx = 1:size(conList,1)
    [con1, con2] = conList{idx,:};
    [colors1, colors2] = colorList{idx,:};
    
    % binary mask of the effect
    bin_1 = fullfile(dataPath,[con1 '_conjunction_not_A_or_B.nii.gz']);
    bin_2 = fullfile(dataPath,[con2 '_conjunction_not_A_or_B.nii.gz']);
    
    
    %% Initialize empty layers and settings variables
    layers                              = sd_config_layers('init',{'truecolor','blob','blob'});
    settings                            = sd_config_settings('init');
    
    %% Define layers
    % Layer 1: Anatomical map
    layers(1).color.file                = fullfile(mni);    %SPM default T1
    layers(1).color.map                 = gray(256);
    layers(1).mask.file                 = fullfile(mnimask);
    
    % Layer 3:
    layers(3).color.file                = fullfile(bin_1);    % PEs of contrast 3d.nii
    layers(3).color.map                 = eval(colors1); %CyBuGyRdYl;CyBuBkRdYl
    layers(3).color.label               = con1;
    layers(3).color.range               = [0,5];
    layers(3).color.opacity             = opacity;    
    layers(3).mask.file                 = fullfile(bin_1);
    layers(3).opacity.label             = 'bin';

    % Layer 2:
    layers(2).color.file                = fullfile(bin_2);    % PEs of contrast 3d.nii
    layers(2).color.map                 = eval(colors2); %CyBuGyRdYl;CyBuBkRdYl
    layers(2).color.label               = con2;
    layers(2).color.range               = [0,5];
    layers(2).color.opacity             = opacity;
    
    layers(2).mask.file                 = fullfile(bin_2);
    layers(2).opacity.label             = 'bin';
    
    layers(3).color.label = '';
    layers(2).color.label = '';
    
    %% Specify display settings
    settings.slice.orientation          = 'axial';
    settings.slice.disp_slices          = -15:15:60;   % axial_narrow
    settings.fig_specs.n.slice_column   = 3;
    settings.slice.show_labels          = 0;
    settings.slice.show_orientation     = 0;
    
    %% Display
    settings.fig_specs.n.slice = numel(settings.slice.disp_slices);
    disp(['no slices: ' num2str(settings.fig_specs.n.slice)])
    
    % Display the layers
    [settings,p] = sd_display(layers,settings);
    
    %% Save figure
    SaveFigures
end