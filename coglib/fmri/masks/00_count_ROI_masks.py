#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combines several masks into theory specific ROIs. PFC for GNW and Posterior for IIT
Created on Fri Jul  8 17:19:11 2022

@author: Yamil Vidal
"""

import sys
import numpy as np

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 
data_dir = projectRoot + '/bids/derivatives/fslFeat'

mask_dir_pattern = bids_dir + '/derivatives/masks/%(sub_id)s'

f_name_pattern = '/%(sub)s_bh_%(roi)s_space-%(space)s.nii.gz'

group_mask = data_dir + '/group/ses-V1/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'
#group_mask = bids_dir + '/MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'

# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import get_subject_list, load_mri, save_mri

subject_list_type = 'phase2_V1'
#subject_list_type = 'debug'

space = 'MNI152NLin2009cAsym'

# %%

# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = False

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
        mask_list = glob(sub_mask_dir + '/*_bh_*' + space + '.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list


# %%
subjects = get_subject_list(bids_dir, subject_list_type)

for sub_id in subjects:

    sub_mask_dir = mask_dir_pattern%{'sub_id':sub_id}    
    mask_paths = get_mask_list(sub_mask_dir)
