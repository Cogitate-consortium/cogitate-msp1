
# Runs cluster-based permutation testing on the searchlight accuracy maps against the chance level
"""
Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 10-10-2022
Date modified: 03-10-2022
"""

import os
import pandas as pd
import nilearn

bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# Select whether to apply group analysis on category or orientation decoding problems - options 'category' and 'orientation'
decoding_problem = 'category'
#decoding_problem = 'orientation'
# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
# For orientation decoding, approach should be set to 'within_condition'
approach = 'generalization'
#approach ='within_condition'
# if approach = 'within_condition', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# if approach = 'generalization', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant-irrelevant' and 'irrelevant-relevant'
# For orientation decoding, no need to change 'condition' as it will be specified at line 40.
condition = 'relevant-irrelevant'
#condition ='relevant'
# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'

# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
# For orientation decoding, do not change 'stimulus_categories' here and change line 48 instead.
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

searchlight_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/searchlight_decoding',approach, condition, classifier, str(searchlight_radius) + 'mm', 'category/' + stimulus_categories[0] + '_' + stimulus_categories[1])

if decoding_problem == 'orientation':
    condition = 'relevant+irrelevant'
    chance_level = 0.33
    stimulus_category = 'face'
    #stimulus_category = 'object'
    #stimulus_category = 'letter'
    #stimulus_category = 'falseFont'
    searchlight_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/searchlight_decoding',approach, condition, classifier, str(searchlight_radius) + 'mm', 'Orientation/' + stimulus_category)

def second_level_analysis(second_level_input,chance_level):

    # Performs second level analysis
    searchlight_images = list()
    # Threshold accuracy maps before running second level GLM
    for index in second_level_input:
        searchlight_images.append(nilearn.image.math_img('a-' + str(chance_level), a=index))
    # Get mean accuracy map across subjects
    accuracy_map = nilearn.image.mean_img(second_level_input)
    # Create Design matrix
    design_matrix = pd.DataFrame([1] * len(searchlight_images), columns=['intercept'])
    # Run second level analysis
    from nilearn.glm.second_level import non_parametric_inference

    out_dict = non_parametric_inference(
        searchlight_images,
        design_matrix=design_matrix,
        n_perm=5000,
        two_sided_test=False,
        n_jobs=6,
        threshold=0.001,
    )
    logp_map = out_dict["logp_max_size"]
    return logp_map, accuracy_map

def second_level_display(logp_map, accuracy_map, chance_level, searchlight_dir):

    # Displays group-level accuracy map on axial brain slices
    from nilearn import plotting
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    # Define figure parameters
    fig = plt.figure(figsize=(5, 7))
    fig.set_facecolor((0, 0, 0))
    plt.subplots_adjust(hspace=0.001)
    plt.subplots_adjust(wspace=0.001)
    c_map = mpl.cm.hot
    slices = range(-56, 96, 8)

    # MNI template to display group-level accuracy
    file_name = 'mni_icbm152_t1_tal_nlin_asym_09c.nii'
    # Cluster correction threshold
    threshold= -np.log10(0.05)
    # Apply threshold and get the corrected group-level accuracy map
    accuracy_mask = nilearn.image.math_img(f'img > {threshold}', img=logp_map)
    accuracy_map_thresholded = nilearn.image.math_img('a*b', a= accuracy_mask, b=accuracy_map)
    # Save group-level accuracy map
    accuracy_map_thresholded.to_filename(os.path.join(searchlight_dir, 'searchlight_group_accuracy_map_nonparametric.nii'))

    # Plot group accuracy map on axial brain slices
    slice_index = 0
    for index in range(18):
        ax = plt.subplot(5, 4, index + 1)
        plotting.plot_img(accuracy_map_thresholded, cut_coords=range(slices[slice_index], slices[slice_index + 1], 8), bg_img=file_name,
                          axes=ax, annotate=False, threshold=chance_level, display_mode="z",
                          cmap=c_map, colorbar=False, vmin=chance_level, vmax=0.8)

        slice_index = slice_index + 1

    ax = plt.subplot(5, 4, 20)
    fig.subplots_adjust(bottom=0.12, top=0.7, left=0.43, right=0.63)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=c_map),
                      cax=ax, orientation='vertical')

    cb.ax.set_yticklabels([chance_level, 0.65, 0.8], fontsize=10, weight='bold')
    cb.set_label('Accuracy', rotation=90, color='white', fontsize=12, weight='bold')
    cb.ax.tick_params(colors='white')
    cb.ax.tick_params(size=0)
    # Save group accuracy map as a figure
    plt.savefig(os.path.join(searchlight_dir, 'searchlight_group_accuracy_map_nonparametric.png'))
    plotting.show()


# Get phase3 subjects
tsv_file = os.path.join(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv')
tsv_data= pd.read_csv(tsv_file, sep='\t')
subjects_phase2 = tsv_data.participant_id
subject_list=subjects_phase2.tolist()
searchlight_filename = list()
# Loop over subjects
for sub in subject_list:
    for root, sess_dirs, session_files in os.walk(os.path.join(searchlight_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            if sess_directory.find('ses-V1') != -1:
                # Get searchlight accuracy maps
                for filecnt, filename in enumerate(os.listdir(os.path.join(searchlight_dir, sub, sess_directory))):
                    if filename.find('searchlight_group_accuracy_map_nonparametric.nii') != -1:
                        searchlight_filename.append(os.path.join(searchlight_dir, sub, sess_directory, filename))

# Perform second level analysis
stats_map, accuracy_map = second_level_analysis(searchlight_filename, chance_level)
# Display group level accuracy map on axial brain slices
second_level_display(stats_map, accuracy_map, chance_level, searchlight_dir)












