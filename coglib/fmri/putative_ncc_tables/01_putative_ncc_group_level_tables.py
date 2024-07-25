#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:02:20 2022

@author: yamil.vidal
"""
import sys, os
import numpy as np
import pandas as pd

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 
data_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type
output_dir = projectRoot + '/bids/derivatives/putative_ncc_tables/' + subject_list_type

mask_dir = bids_dir + '/derivatives/masks/ICBM2009c_asym_nlin'
conjunct_pattern = data_dir + '/%(conj)s_conjunction_not_A_or_B.nii.gz'
output_fname_pattern_cvs = output_dir + '/%(conj)s_%(type)s_MNI.csv'
output_fname_pattern_pickle = output_dir + '/%(conj)s_%(type)s_MNI.pkl'

group_mask = bids_dir + '/derivatives/fslFeat/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import load_mri

# %% ROIs
roi_list = ['G_and_S_cingul-Mid-Post',
            'Lat_Fis-ant-Horizont',
            'Lat_Fis-ant-Vertical',
            'G_and_S_cingul-Ant',
            'G_and_S_cingul-Mid-Ant',
            'G_front_inf-Opercular',
            'G_front_inf-Orbital',
            'G_front_inf-Triangul',
            'G_front_middle',
            'S_front_middle',
            'S_front_sup',
            'G_and_S_frontomargin',
            'G_and_S_transv_frontopol',
            'G_front_sup',
            'G_rectus',
            'G_subcallosal',
            'S_orbital_lateral',
            'S_orbital_med-olfact',
            'S_orbital-H_Shaped',
            'S_suborbital',
            'G_and_S_occipital_inf',
            'G_oc-temp_lat-fusifor',
            'G_occipital_middle',
            'S_oc_middle_and_Lunatus',
            'G_cuneus',
            'G_occipital_sup',
            'G_oc-temp_med-Lingual',
            'G_oc-temp_med-Parahip',
            'G_temporal_inf',
            'Pole_occipital',
            'Pole_temporal',
            'S_calcarine',
            'S_intrapariet_and_P_trans',
            'S_oc_sup_and_transversal',
            'S_temporal_sup',
            'S_front_inf',
            'G_orbital',
            'G_pariet_inf-Angular',
            'G_pariet_inf-Supramar',
            'G_precentral',
            'G_temp_sup-Lateral',
            'G_temp_sup-Plan_tempo',
            'G_temporal_middle',
            'S_interm_prim-Jensen',
            'S_occipital_ant',
            'S_oc-temp_lat',
            'S_precentral-inf-part',
            'S_temporal_inf',
            'GNW',
            'IIT',
            'IIT_extended',
            'IIT_excluded']


# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = True

# %% Conjunctions

conjunctions = ['C_Face_Activation',
                'C_Face_Deactivation',
                'C_Object_Activation',
                'C_Object_Deactivation',
                'C_Letter_Activation',
                'C_Letter_Deactivation',
                'C_FalseFont_Activation',
                'C_FalseFont_Deactivation',
                'C_combined_or_Activation',
                'C_combined_or_Deactivation',
                'B',
                'A']

#conjunctions = ['C_Face_Activation']

# %%

# load all anatomical ROIs
def get_mask_list(sub_mask_dir):
    """
    Get list of paths to all masks for current subject. Prints how many masks
    are found or prints warning (not an error) if none have been found.
    sub_mask_dir: path to mask dir
    Returns: list of paths to all masks
    """
    from glob import glob
    if process_only_bilateral_masks:
        mask_list = glob(sub_mask_dir + '/*_bh_*' + 'MNI152NLin2009cAsym.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list

def load_a_rois():
    """
    Load all anatomica ROIs of a subject into a dictionary
    
    sub_id: Subject ID
    Returns: Dictionary containing all the anatomical ROIs of the subject
    """
    
    mask_paths = get_mask_list(mask_dir)
    mask_paths.sort()
    
    sub_mask_list = [l[115:] for l in mask_paths]
    sub_mask_list = [l[:-33] for l in sub_mask_list]
    
    # empty dictionary that will contain all masks of a subject
    a_rois = {}
    
    for n in range(0,len(mask_paths)):
        mask = mask_paths[n]
        m = load_mri(mask, group_mask)
        a_rois[sub_mask_list[n]] = m
        
    return a_rois, sub_mask_list


def count_voxels_in_roi(a_roi,conjunction):
    
    # Set to zero the voxels that don't belong to the ROI
    
    c = conjunction.copy()
    c[a_roi == 0] = 0
    
    n_voxels = int(np.sum(c > 0))
    
    return n_voxels

# %% run

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
            
for conjunct_name in conjunctions:
    
    voxel_count = dict(zip(roi_list,np.zeros((len(roi_list),2), dtype=float)))
    
    if conjunct_name[0] != 'C':
        conjunct_pattern = data_dir + '/%(conj)s_conjunction.nii.gz'

    # Load the relevant functional maps
    conjunction = np.squeeze(load_mri(conjunct_pattern%{'conj':conjunct_name},group_mask))
    
    a_rois, sub_roi_list = load_a_rois()
        
    for roi_name in roi_list:
        
        print(conjunct_name + ' ROI: ' + roi_name)
        #roi_name = roi_list[0]
        a_roi = np.squeeze(a_rois[roi_name])

        n_voxels = count_voxels_in_roi(a_roi,conjunction)
        #print(str(n_voxels))
        #print(str(n_voxels/sum(a_roi)))
        voxel_count[roi_name][0] = n_voxels
        voxel_count[roi_name][1] = n_voxels/sum(a_roi)*100
    
    voxel_df = pd.DataFrame.from_dict(voxel_count, orient='index')
    voxel_df.to_csv(output_fname_pattern_cvs%{'conj':conjunct_name, 'type':'voxels'}, header=False)
    voxel_df.to_pickle(output_fname_pattern_pickle%{'conj':conjunct_name, 'type':'voxels'})
    