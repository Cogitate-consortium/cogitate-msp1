#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample ROI masks to target space using ANTs apply transform for each mask 
and participant. Useful if voxel dimensions of target space (e.g. EPI data) 
differ from original mask space dimension (e.g. FS masks).
First gets subject list from bids_dir, then finds mask dir and target space 
examples niftis. Then resamples all masks in mask dir to target space using 
an identity matrix or if supplied a different ANTs compatible transformation 
matrix.

Make sure ANTs is loaded before executing the python script here. 
I.e., in the shell run, before executing the python script:
module load ANTs

Created on Tue Sep 22 13:01:00 2021

@author: David Richter
@tag: prereg_v4.2
"""

import sys, os


#%% Paths and Parameters

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'

#subject_list_type = 'phase2_V1'
subject_list_type = 'debug'

# path where the masks are located 
mask_dir_pattern = bids_dir + '/derivatives/masks/%(sub_id)s'

# Target space to resample to (must correspond to an existing bids space- 
# key-value pair; e.g. T1w); is also appended as label to output mask
# target_space = 'T1w'
target_space = 'MNI152NLin2009cAsym'

# Example target space nifti file corresponding to the target space
target_space_example = bids_dir + '/derivatives/fmriprep/%(sub_id)s/ses-V1/func/%(sub_id)s_ses-V1_task-Dur_run-1_space-%(target_space)s_boldref.nii.gz'
# Example target space nifti file used in case the first example does not exist (e.g. for subjects who do not have (valid) exp1 data)
backup_target_space_example = bids_dir + '/derivatives/fmriprep/%(sub_id)s/ses-V2/func/%(sub_id)s_ses-V2_task-VG_run-1_space-%(target_space)s_boldref.nii.gz'

# Input transformation matrix. If the target space does not correspond to the 
# already registered mask space (probably T1w is using FS masks) then a custom
# transformation matrix MUST be supplied here to ensure correct registration of
# mask and target space - e.g. for MNI space this could be used:
custom_transformation_matrix_pattern = bids_dir + '/derivatives/fmriprep/%(sub_id)s/ses-V1/anat/%(sub_id)s_ses-V1_acq-anat_run-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
# If none is supplied or set to False the identity matrix is used, assuming
# output space and mask are already registered 
# custom_transformation_matrix_pattern = False


# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = True


# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/code'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import get_subject_list, run_subprocess


#%% resample functions
def get_mask_list(sub_mask_dir):
    """
    Get list of paths to all masks for current subject. Prints how many masks
    are found or prints warning (not an error) if none have been found.
    sub_mask_dir: path to mask dir
    Returns: list of paths to all masks
    """
    from glob import glob
    if process_only_bilateral_masks:
        mask_list = glob(sub_mask_dir + '/*_bh_*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list

def get_target_space_example(sub_id,target_space):
    """
    Get example nifti file of target space. First checks intended target space
    as defined in target_space_example; if that one does not exist tries to use
    the file in backup_target_space_example.
    sub_id: subject id
    target_space: target space (e.g. T1w)
    Returns: paths to target space examplar for current subject
    """
    original_sub_target_space = target_space_example%{'sub_id':sub_id,'target_space':target_space}
    backup_sub_target_space = backup_target_space_example%{'sub_id':sub_id,'target_space':target_space}
    if not os.path.isfile(original_sub_target_space) and not os.path.isfile(backup_sub_target_space):
        print('! No target space example found for: ' + sub_id + ' | Skipping subject !!!')
        sub_target_space = []
    elif os.path.isfile(original_sub_target_space):
        print('. Using original target space example for: ' + sub_id)
        sub_target_space = original_sub_target_space
    elif not os.path.isfile(original_sub_target_space) and os.path.isfile(backup_sub_target_space):
        print('. Using backup target space example for: ' + sub_id)
        sub_target_space = backup_sub_target_space
    return sub_target_space

def resample_to_target(sub_id):
    """
    Resample masks in mask_list to target space without any warping 
    (i.e. identity matrix) using ANTs apply transform. 
    Useful if target space (e.g. EPI data) have a different voxel dimension 
    than input masks (e.g. from FS).
    sub_id: subject ID.
    mask_list: list of mask labels
    mask_output_dir: Path to mask output folder.
    target_space_example: Path to example nifti data in target space.
    Returns: nothing, but creates masks in target space in mask_output_dir
    
    """
    # get mask dir and mask list
    sub_mask_dir = mask_dir_pattern%{'sub_id':sub_id}
    mask_list = get_mask_list(sub_mask_dir)
    # get current target space
    sub_target_space = get_target_space_example(sub_id,target_space)
    # create & run command for each mask in mask list
    if sub_target_space:
        for input_mask in mask_list:
            print('. . Resampling mask: ' + input_mask[input_mask.find('masks')::])
            # get input mask file name
            input_fname = input_mask[0:input_mask.find('.nii.gz')]
            output_fname = input_fname + '_space-' + target_space
            # get transformation matrix (if supplied)
            if custom_transformation_matrix_pattern:
                transformation_mat = custom_transformation_matrix_pattern%{'sub_id':sub_id}
            else:
                transformation_mat = 'identity'
            # make ants apply transform command with identity mat and target space reference
            ants_cmd_pattern = 'antsApplyTransforms --input %(input_fname)s.nii.gz --interpolation NearestNeighbor --output %(output_fname)s.nii.gz --reference-image %(sub_target_space)s --transform %(transformation_mat)s'
            full_cmd = ants_cmd_pattern%{'input_fname':input_fname, 'output_fname':output_fname, 'sub_target_space':sub_target_space, 'transformation_mat':transformation_mat}
            # execute command 
            run_subprocess(full_cmd)


# %% run
if __name__ == '__main__':
    """
    Resample ROI masks in mask dir to target space for each subject
    Gets subject list from bids_dir, gets mask to resample, then resamples
    using ANTs apply transform.
    """
    
    # get subject list
    subjects = get_subject_list(bids_dir, subject_list_type)
    
    # run mask creation for each subject
    for sub_id in subjects:
        print('Preparing to resample ROI mask for: ' + sub_id)
        resample_to_target(sub_id)
