# Generates corrected and uncorrected group_level gppi stats maps
# Corrected group-level maps are obtained by running cluster-based permutation testing on the gppi contrast maps
"""
Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 03-14-2023
Date modified:
"""

import os
import pandas as pd
import nilearn

# Select whether to apply group analysis on contrasts from the 'separate conditions' or  'combined conditions' analysis
# Separate conditions
contrasts = ['face_rel-object_rel',  'face_irr-object_irr']
# Combined conditions
#contrasts = ['face-object']
# Select seed size in voxels
n_voxels = 300
# Select seed (FFA or LOC)
# LOC
roi_folder = 'PPI_LOC_gPPI_' + str(n_voxels)
# FFA
#roi_folder = 'PPI_FFA_gPPI_' + str(n_voxels)

def second_level_analysis(second_level_input):

    # Performs second level analysis
    # Create Design matrix
    design_matrix = pd.DataFrame([1] * len(second_level_input), columns=['intercept'])
    # Run second level analysis
    from nilearn.glm.second_level import non_parametric_inference

    out_dict = non_parametric_inference(
        second_level_input,
        design_matrix=design_matrix,
        n_perm=5000,
        two_sided_test=True,
        n_jobs=6,
        threshold=0.001,
    )
    logp_map = out_dict["logp_max_size"]
    t_map = out_dict["t"]
    return t_map, logp_map

def second_level_display(t_map, logp_map, group_dir, n_subjects):

    # Displays group-level gppi stats map on axial brain slices
    from nilearn import plotting
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    # Define figure parameters
    fig = plt.figure(figsize=(5, 7))
    fig.set_facecolor((0, 0, 0))
    plt.subplots_adjust(hspace=0.001)
    plt.subplots_adjust(wspace=0.001)
    c_map = mpl.cm.jet
    slices = range(-56, 96, 8)

    # MNI template to display gppi stats map
    file_name = 'mni_icbm152_t1_tal_nlin_asym_09c.nii'
    # Cluster correction threshold
    threshold = -np.log10(0.05)
    # Apply threshold
    abs_logp_map = nilearn.image.math_img('np.abs(img)', img=logp_map)
    logp_mask = nilearn.image.math_img(f'img > {threshold}', img=abs_logp_map)
    corrected_stats_map = nilearn.image.math_img('a*b', a=logp_mask, b=t_map)
    # Save corrected group-level stats map
    corrected_stats_map.to_filename(os.path.join(group_dir, 'gppi_group_stats_map.nii'))

    # Set threshold for uncorrected group-level stats map
    unc_pval = 0.01
    dof = n_subjects - 2  # DOF: group and intercept
    threshold = stats.t.isf(unc_pval / 2, df=dof)  # two-sided
    # Apply threshold to the group-level stats map
    abs_t_map = nilearn.image.math_img('np.abs(img)', img=t_map)
    t_mask = nilearn.image.math_img(f'img > {threshold}', img=abs_t_map)
    uncorrected_stats_map = nilearn.image.math_img('a*b', a=t_mask, b=t_map)
    # Save uncorrected group-level stats map
    uncorrected_stats_map.to_filename(os.path.join(group_dir, 'gppi_group_stats_map_uncorr_0.01.nii'))

    # Plot corrected stats map on sagittal brain slices
    slice_index = 0
    for index in range(18):
        ax = plt.subplot(5, 4, index + 1)
        plotting.plot_stat_map(corrected_stats_map, bg_img= file_name, cut_coords=range(slices[slice_index], slices[slice_index + 1], 8),
                            axes=ax, annotate=False, display_mode="z",
                            cmap=c_map, colorbar=False, vmax=6)

        slice_index = slice_index + 1

    ax = plt.subplot(5, 4, 20)
    fig.subplots_adjust(bottom=0.12, top=0.7, left=0.43, right=0.63)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=c_map),
                      cax=ax, orientation='vertical')

    cb.ax.set_yticklabels([-6, 0, 6], fontsize=10, weight='bold')
    cb.set_label('T-value', rotation=90, color='white', fontsize=12, weight='bold')
    cb.ax.tick_params(colors='white')
    cb.ax.tick_params(size=0)
    # Save gppi stats map as a figure
    plt.savefig(os.path.join(group_dir, 'gppi_group_map_nonparametric_uncorr_0.01.png'))
    plotting.show()


import nilearn.image
gppi_filename = list()
# Loop over subjects
for contrast in contrasts:
    # Contrast files path
    contrasts_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/contrasts', roi_folder)
    # Output path
    group_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/second level_nonparametric', roi_folder, contrast)
    # Create an output folder if it does not already exist
    if not os.path.isdir(group_dir):
        os.makedirs(group_dir, exist_ok=True)
    for filecnt, filename in enumerate(os.listdir(contrasts_dir)):
        if ((filename.find('con_PPI_' + contrast + '_sub') != -1) & (filename.find('.img') != -1)):
            gppi_filename.append(nilearn.image.load_img(os.path.join(contrasts_dir, filename)))
    # Perform second level analysis
    t_map, logp_map = second_level_analysis(gppi_filename)
    # Display corrected group level gppi stats map on axial brain slices
    second_level_display(t_map, logp_map, group_dir, len(gppi_filename))















