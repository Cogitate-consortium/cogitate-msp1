#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combines several masks into theory specific ROIs. PFC for GNW and Posterior for IIT
Created on Fri Jul  8 17:19:11 2022

@author: Yamil Vidal
"""

import sys, os
import numpy as np
from datetime import datetime

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

bids_dir = projectRoot + '/bids'
code_dir = projectRoot + '/bids/code'
data_dir = projectRoot + '/bids/derivatives/fslFeat'

type_of_roi = 'gppi_seeds'
mask_dir_pattern = bids_dir + '/derivatives/'+ type_of_roi +'/%(sub_id)s'
old_mask_dir_pattern = bids_dir + '/derivatives/'+ type_of_roi +'/new/%(sub_id)s'

brain_mask_pattern = data_dir + '/%(sub_id)s/ses-V1/%(sub_id)s_ses-V1_task-Dur_analysis-2ndGLM_space-MNI152NLin2009cAsym.gfeat/mask.nii.gz'

# Get the current date and time as a string
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
log_file = bids_dir + '/derivatives/' + type_of_roi + '/roi_size_log_' + dt_string + '.txt'


# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import get_subject_list, load_mri

subject_list_type = 'phase2_V1'
#subject_list_type = 'debug'

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
        mask_list = glob(sub_mask_dir + '/*_bh_*LOC*' + space + '.nii.gz')
        mask_list = [s for s in mask_list if "leave_run" not in s]
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list

def compare_rois(roi_name):

    path1 = mask_dir_pattern%{'sub_id':sub_id} + '/' + roi_name
    path2 = old_mask_dir_pattern%{'sub_id':sub_id} + '/' + roi_name

    roi1 = load_mri(path1, brain_mask)
    try:
        roi2 = load_mri(path2, brain_mask)

        a = np.logical_and(roi1,roi2)
        o = np.logical_or(roi1,roi2)

        overlap = sum(a)/sum(o)

        if not(all(roi1 == roi2)):
            message = 'New: ' + str(int(np.sum(roi1))) + '\n' + \
                      'Old: ' + str(int(np.sum(roi2))) + '\n' + \
                      'Overlap: ' + str(overlap) + '\n' + path1 + '\n\n'
            print('ROIs dont fully overlap! Check log file!!\n')

            try:
                with open(log_file, 'a') as file:
                    file.write(message)
            except FileNotFoundError:
                with open(log_file, 'w') as file:
                    file.write(message)
    except FileNotFoundError:
        message = 'Missing: ' + path2 + '\n\n'
        print('Old ROI is missing!!\n')

        try:
            with open(log_file, 'a') as file:
                file.write(message)
        except FileNotFoundError:
            with open(log_file, 'w') as file:
                file.write(message)

    return overlap


# %%
subjects = get_subject_list(bids_dir, subject_list_type)

remove_subjects = ['sub-CD122','sub-CD196']
for r in remove_subjects:
    subjects = subjects[subjects != r]

print('Removed subjects:',remove_subjects)
print('Total subjects:',len(subjects))

Overlaps = np.array([])

for sub_id in subjects:
    # Subject's brain mask
    brain_mask = brain_mask_pattern%{'sub_id':sub_id}

    # Get a list of ROIs to compare
    sub_mask_dir = mask_dir_pattern%{'sub_id':sub_id}
    mask_paths = get_mask_list(sub_mask_dir)
    roi_names = [os.path.basename(mask_paths[x]) for x in range(0,len(mask_paths))]

    for roi_name in roi_names:
        print('Comparing ROI:\n', roi_name,'\n')
        overlap = compare_rois(roi_name)
        Overlaps = np.append(Overlaps,overlap)