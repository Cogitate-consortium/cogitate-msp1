"""
Plots ROI accuracy results on a brain surface

Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 03-04-2023
"""

import numpy as np
import matplotlib.pyplot as plt
from plotters import plot_time_series, plot_matrix, plot_rasters, plot_brain
import config
import os
import pandas as pd
# get the parameters dictionary
param = config.param

# =================================================================================
# Select whether to apply plotting to category or orientation decoding problems - options 'category' and 'orientation'
decoding_problem = 'category'
#decoding_problem = 'orientation'
#Select within condition decoding or generalization across conditions - options 'within_condition' and 'generalization'
# For orientation decoding, approach should be set to 'within_condition'
approach = 'generalization'
approach = 'within_condition'
# if approach = 'within_condition', the line below selects  relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# if approach = 'generalization', the line below selects generalization direction  - options 'relevant-irrelevant' and 'irrelevant-relevant'
condition = 'relevant'
condition = 'relevant-irrelevant'
classifier = 'SVM'
# Stimulus categories - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['face', 'object']
stimulus_categories = ['letter', 'falseFont']
#stimulus_categories = ['letter', 'falsefont']
# Number of voxels per ROI (ROI size)
n_voxels = 300
# Chance level (50% for category decoding (binary) and 33.33% for orientation decoding (3-class) )
chance_level = 0.5
csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding/',approach, classifier, decoding_problem, stimulus_categories[0] + '_' + stimulus_categories[1] + '_' + str(n_voxels))
if decoding_problem == 'orientation':
    condition = 'relevant+irrelevant'
    chance_level = 0.33
    stimulus_category = 'face'
    #stimulus_category = 'object'
    #stimulus_category = 'letter'
    #stimulus_category = 'falseFont'
    csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding/', approach, classifier, decoding_problem, stimulus_category + '_' + str(n_voxels))

if decoding_problem == 'category':
            vmin=0.5
            vmax=0.85
            if ((condition=='relevant') | (condition=='irrelevant-relevant')):
                cmaps = 'Oranges'
                cmap_start = 0.1
                cmap_end = 1
            elif ((condition == 'irrelevant') | (condition == 'relevant-irrelevant')):
                cmaps = 'Purples'
                cmap_start = 0.5
                cmap_end = 1
elif decoding_problem == 'orientation':
    cmaps = 'Purples'
    vmin=0.33
    vmax=0.40
    cmap_start = 0.5
    cmap_end = 1

# Plotting brain surface:
acc_values = {'G_front_middle': 0.9, 'Pole_temporal': 0.7, 'G_occipital_middle': 0.6}
#IIT_roi_list = ['GNW','IIT','IIT_excluded', 'IIT_extended', 'G_and_S_frontomargin', 'G_and_S_transv_frontopol', 'G_front_sup', 'G_rectus', 'G_subcallosal', 'S_orbital_lateral', 'S_orbital_med-olfact', 'S_orbital-H_Shaped', 'S_suborbital']
GNW_roi_list = ['G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant', 'G_and_S_cingul-Mid-Post', 'G_front_inf-Opercular', 'G_front_inf-Orbital', 'G_front_inf-Triangul', 'G_front_middle', 'Lat_Fis-ant-Horizont', 'Lat_Fis-ant-Vertical', 'S_front_inf', 'S_front_middle', 'S_front_sup']
# IIT Basic ROI list
IIT_roi_list_1 = ['G_temporal_inf', 'Pole_temporal', 'G_cuneus', 'G_occipital_sup', 'G_oc-temp_med-Lingual', 'Pole_occipital', 'S_calcarine', 'G_and_S_occipital_inf', 'G_occipital_middle', 'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Parahip', 'S_intrapariet_and_P_trans', 'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_temporal_sup']
roi_list = GNW_roi_list + IIT_roi_list_1

csv_file = os.path.join(csv_dir, 'accuracy_stats_' + condition + '.csv')
save_file= os.path.join(csv_dir, condition + '.eps')
data_df = pd.read_csv(csv_file)
rois = (data_df['ROI']).tolist()
accuracies= (data_df['Average Accuracy']).array
Significance = (data_df['Significance']).array
#Significance[10]=0
#Significance[36]=0
#G_orbital
#Significance[18]=0
#Significance[16]=0
rois_dict = {}
k=0
for roi in rois:
    if roi in roi_list:
        if Significance[k]:
            rois_dict[roi] =accuracies[k]
    k=k+1


plot_brain( roi_map=rois_dict, subject='fsaverage', surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s',
                         views=['lateral',(90, -30, 0)],cmap_start=cmap_start, cmap_end=cmap_end,
                         cmap=cmaps, colorbar=True, colorbar_title='ACC', vmin=vmin, vmax=vmax, outline_overlay=True, overlay_method='overlay',
                         brain_cmap='Greys', brain_alpha=1, save_file=os.path.join(csv_dir, condition + '_lh_outline_30deg.png'))

#roi_map_edge_color = [0, 0, 0] to add borders around rois



