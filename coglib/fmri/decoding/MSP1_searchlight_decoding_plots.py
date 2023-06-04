"""
Plots searchlight accuracy maps on a brain surface

Author: Aya Khalaf
Date created: 03-04-2023
"""

from plotters import plot_brain
import matplotlib
import matplotlib.pyplot as plt
import os
import config

# Select whether to apply plotting to category or orientation decoding problems - options 'category' and 'orientation'
decoding_problem = 'category'
decoding_problem = 'orientation'
# Select within condition decoding or generalization across conditions - options 'within_condition' and 'generalization'
# For orientation decoding, approach should be set to 'within_condition'
approach = 'generalization'
approach = 'within_condition'
# if approach = 'within_condition', the line below selects  relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# if approach = 'generalization', the line below selects generalization direction  - options 'relevant-irrelevant' and 'irrelevant-relevant'
# For orientation decoding, no need to change 'condition' as it will be specified at line 40.
condition = 'irrelevant-relevant'
#condition ='relevant'

classifier = 'SVM'

# Stimulus categories  - options include 'face', 'object', 'letter' , and 'falseFont'
# For orientation decoding, do not change 'stimulus_categories' here and change line 42 instead.
stimulus_categories = ['face', 'object']
#stimulus_categories = ['letter', 'falseFont']
#stimulus_categories = ['face', 'baseline']
#stimulus_categories = ['object', 'baseline']
#stimulus_categories = ['letter', 'baseline']
#stimulus_categories = ['falseFont', 'baseline']
#stimulus_categories = ['falseFonts', 'baseline']
# Radius of searchlight sphere
searchlight_radius = 4
# Chance level (50% for category decoding (binary) and 33.33% for orientation decoding (3-class) )
chance_level = 0.5

data_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/searchlight_decoding',approach, condition, classifier, str(searchlight_radius) + 'mm', 'category/' + stimulus_categories[0] + '_' + stimulus_categories[1])

if decoding_problem == 'orientation':
    condition = 'relevant+irrelevant'
    chance_level = 0.33
    stimulus_category = 'face'
    #stimulus_category = 'object'
    #stimulus_category = 'letter'
    #stimulus_category = 'falseFont'
    data_dir = os.path.join( '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/searchlight_decoding',approach, condition, classifier, str(searchlight_radius) + 'mm', 'Orientation/' + stimulus_category)


if decoding_problem == 'category':
    if ((condition == 'relevant') | (condition == 'irrelevant-relevant')):
        cmaps = 'Oranges'
        cmap_start = 0.1
        cmap_end = 1
    elif ((condition == 'irrelevant') | (condition == 'relevant-irrelevant')):
        cmaps = 'Purples'
        cmap_start = 0.5
        cmap_end = 1
    #plot_brain( surface='inflated', cmap=cmaps, cmap_start=0.5, cmap_end=1, overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')], overlay_threshold=0, vmin=chance_level, vmax=0.75, outline_overlay=True, save_file=os.path.join(data_dir, condition + '_' + 'lh_outline.png'))
    ## for the right side
    #plot_brain( surface='inflated', cmap=cmaps, cmap_start=0.5, cmap_end=1, hemi='rh', overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')], overlay_threshold=0, vmin=chance_level, vmax=0.75, outline_overlay=True, views=['medial', 'lateral'], save_file=os.path.join(data_dir, condition + '_' + 'rh_outline.png'))
    plot_brain( surface='inflated', cmap=cmaps,  overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')], overlay_threshold=0, vmin=chance_level, vmax=0.75, cmap_start=cmap_start, cmap_end=cmap_end, outline_overlay=True, views=['lateral',(90, -30, 0)], save_file=os.path.join(data_dir, condition + '_' + 'lh_outline_30deg.png'))
    ## for the right side
    plot_brain( surface='inflated', cmap=cmaps,  hemi='rh', overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')], overlay_threshold=0, vmin=chance_level, vmax=0.75, cmap_start=cmap_start, cmap_end=cmap_end, outline_overlay=True, views=[(-90, 30, 0), 'lateral'], save_file=os.path.join(data_dir, condition + '_' + 'rh_outline_30deg.png'))

elif decoding_problem == 'orientation':
    param = config.param
    colors = [(1, 1, 1), param['colors']['face']]
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', colors, N=128)

    """
    plot_brain(surface='inflated', cmap=cmaps, cmap_start=0.5, cmap_end=1,
               overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')],
               overlay_threshold=0, vmin=chance_level, vmax=0.45, outline_overlay=True, views=['lateral', (180, 20, -10)],
               save_file=os.path.join(data_dir, 'orientation_lh_inf_outline_posterior.png'))
    # for the right side
    plot_brain(surface='inflated', cmap=cmaps,  cmap_start=0.5, cmap_end=1, hemi='rh',
               overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')],
               overlay_threshold=0, vmin=chance_level, vmax=0.45, outline_overlay=True, views=[(180, 20, -10), 'lateral'],
               save_file=os.path.join(data_dir, 'orientation_rh_inf_outline_posterior.png'))
"""
plot_brain(surface='inflated', cmap=cmaps, cmap_start=0.5, cmap_end=1,
           overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')],
           overlay_threshold=0, vmin=chance_level, vmax=0.45, outline_overlay=True, views=['lateral', 'medial'],
           save_file=os.path.join(data_dir, 'orientation_lh_inf_outline.png'))
# for the right side
plot_brain(surface='inflated', cmap=cmaps, cmap_start=0.5, cmap_end=1, hemi='rh',
           overlays=[os.path.join(data_dir, 'searchlight_group_accuracy_map_nonparametric.nii')],
           overlay_threshold=0, vmin=chance_level, vmax=0.45, outline_overlay=True, views=['medial', 'lateral'],
           save_file=os.path.join(data_dir, 'orientation_rh_inf_outline.png'))

