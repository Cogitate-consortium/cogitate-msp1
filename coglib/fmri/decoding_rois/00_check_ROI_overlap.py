#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:02:20 2022

@author: yamil.vidal
"""
import os, sys
import numpy as np
import pandas as pd

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

#subject_list_type = 'phase2_V1'
subject_list_type = 'phase3_V1_subset'
#subject_list_type = 'debug'

bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 
old_dir = projectRoot + '/bids/derivatives/decoding_rois/OLD'
new_dir = projectRoot + '/bids/derivatives/decoding_rois/'

mask_dir_pattern = old_dir + '/%(sub_id)s'

output_fname_pattern_cvs = new_dir + '/%(conj)s_%(type)s.csv'
output_fname_pattern_pickle = new_dir + '/%(conj)s_%(type)s.pkl'

group_mask = bids_dir + '/derivatives/fslFeat/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import get_subject_list, load_mri


space = 'MNI152NLin2009cAsym'

# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = True


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
        mask_list = glob(sub_mask_dir + '/*irrel_bh_face*300*' + space + '.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list

def load_a_rois(sub_id):
    """
    Load all anatomica ROIs of a subject into a dictionary
    
    sub_id: Subject ID
    Returns: Dictionary containing all the anatomical ROIs of the subject
    """
    sub_mask_dir = mask_dir_pattern%{'sub_id':sub_id}
    mask_paths = get_mask_list(sub_mask_dir)
    mask_paths.sort()
    
    mask_paths_new = [l[:79] + l[83:] for l in mask_paths]
    
    sub_mask_list = [l[95:] for l in mask_paths]
    sub_mask_list = [l[:-33] for l in sub_mask_list]
    
    # empty dictionary that will contain all masks of a subject
    a_rois = {}
    
    for n in range(0,len(mask_paths)):
        mask = mask_paths[n]
        old = load_mri(mask, group_mask)
        
        mask = mask_paths_new[n]
        new = load_mri(mask, group_mask)
        
        a_rois[sub_mask_list[n]] = sum(np.logical_and(old,new))/sum(old)
        
    return a_rois, sub_mask_list


# %% run
subjects = get_subject_list(bids_dir, subject_list_type)

        
for sub in range(len(subjects)):
    
    sub_id = subjects[sub]
    a_rois, sub_roi_list = load_a_rois(sub_id)
