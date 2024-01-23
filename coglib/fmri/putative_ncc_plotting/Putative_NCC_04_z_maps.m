%{
Plots individual z maps used in the putative NCC analysis (step 12). Requires Slice Display (https://github.com/bramzandbelt/slice_display)

Author: Yamil Vidal
Email: hvidaldossantos@gmail.com
 
%}

clear
close all

%listType = 'Phase2';
listType = 'Phase3';

dataPath         = fullfile(pwd,'Data','zmaps',listType);
outPath          = fullfile(pwd,'Data','plots',listType);
sliceDisplayPath = fullfile(pwd,'slice_plotting','SliceDisplay');

mkdir(outPath);
addpath(genpath(sliceDisplayPath));

%% Subject, session and gFeat (contrast) folder & labels

fileName = 'z_maps';

% standard space template brain (MNI T1 2mm)
mni     = fullfile(pwd,'Data','MNI152NLin2009cAsym_res-01_T1w.nii.gz');
mnimask = fullfile(pwd,'Data','MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz');
    
conList = {'RelFace_act' 'RelFace_deact';... 
           'RelObject_act' 'RelObject_deact';...
           'RelLetter_act' 'RelLetter_deact';...
           'RelFalseFont_act' 'RelFalseFont_deact';...
           'IrrelFace_act' 'IrrelFace_deact';... 
           'IrrelObject_act' 'IrrelObject_deact';...
           'IrrelLetter_act' 'IrrelLetter_deact';...
           'IrrelFalseFont_act' 'IrrelFalseFont_deact'};

CustomColor_zMaps

colorList = repmat({'A' 'D'},size(conList,1),1);

opacity = .6;
color_range_act   = [3.1,8.5];
color_range_deact = [3.1,6.5];

for idx = 1:size(conList,1)
    [con1, con2] = conList{idx,:};
    [colors1, colors2] = colorList{idx,:};
        
    bin_1 = fullfile(dataPath,[con1, '.nii.gz']);
    bin_2 = fullfile(dataPath,[con2 '.nii.gz']);
    
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
    layers(3).color.range               = color_range_act;
    layers(3).color.opacity             = opacity;
    
    layers(3).mask.file                 = fullfile(bin_1);
    layers(3).opacity.file              = fullfile(bin_1);   % t stat of corresponding contrast
    layers(3).opacity.label             = 'bin';
    
    % Layer 2:
    layers(2).color.file                = fullfile(bin_2);    % PEs of contrast 3d.nii
    layers(2).color.map                 = eval(colors2); %CyBuGyRdYl;CyBuBkRdYl
    layers(2).color.label               = con2;
    layers(2).color.range               = color_range_deact;
    layers(2).color.opacity             = opacity;
    
    layers(2).mask.file                 = fullfile(bin_2);
    layers(2).opacity.file              = fullfile(bin_2);   % t stat of corresponding contrast
    layers(2).opacity.label             = 'bin';
    
    layers(3).color.label = 'Activation';
    layers(2).color.label = 'Deactivation';

    %% Specify display settings
    settings.slice.orientation          = 'axial';
    settings.slice.disp_slices          = -15:15:60;   % axial_narrow
    settings.fig_specs.n.slice_column   = 6;

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