#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge MNI masks across subjects into one 4D nifti file containing masks for 
all subjects in bids_dir. Then averages across participant masks, 
thresholds, binarizes and writes one output group mask per ROI to 
group_mask_dir containing the final group MNI mask.

Make sure FSL is loaded before executing the python script here; i.e., in
the shell run:

module load FSL
python 03_merge_MNI_masks.py

Created on Tue Sep 28 11:30:00 2021

@author: David Richter

"""

import os, sys


#%% Paths and Parameters

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'

# list of MNI space masks, dervived from subject anatomy to merge into one 
# group level mask (unless otherwise specified all masks are assumed to contain 
# voxels across both hemispheres (i.e. have a bh_ label in front of the mask name))
mask_list = ['V1_exvivo', 'fusiform', 'lateraloccipital',
                     'frontalpole', 'medialorbitofrontal', 
                     'lateralorbitofrontal', 'rostralmiddlefrontal', 
                     'superiorfrontal','rostralanteriorcingulate', 
                     'caudalanteriorcingulate', 'parsorbitalis', 
                     'parstriangularis', 'parsopercularis']

# specific MNI space label
MNI_space = 'MNI152NLin2009cAsym'

# threshold for averaged MNI masks. Since masks are derived from individual T1 
# scans based on cortex parcellation masks in MNI space for each participant 
# can differ; a group mask will be created by averaging masks and thresholding 
# at the threshold set here; e.g. threshold = 0.5 would mean that 50% of 
# participants' masks must contain the voxel for making it into the group mask
threshold = 0.5


# %% Additional paths/parameters. These should be identical across BIDS datasets

# path where custom masks are stored
mask_dir_pattern = bids_dir + '/derivatives/masks/%(sub)s'

# group level custom mask dir
group_mask_dir = mask_dir_pattern%{'sub':'group'}

# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/code'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import get_subject_list, run_subprocess


# %% functions to merge and average masks
def merge_MNI_mask_to_4d_file(mask, mask_dir_pattern):
    """
    Merge MNI masks from separate nifti files (participants) into one 4D mask 
    file. Using all subjects found in bids_dir.
    mask: mask label
    mask_dir_pattern: Pattern to mask dir
    Returns: nothing, but creates merged mask (group4D) in group_mask_dir with 
    MNI_space label
    """
    # get subject list
    subjects = get_subject_list(bids_dir)
    # create fsl merge command pattern
    fslmerge_cmd_pattern = 'fslmerge -t %(group_mask_dir)s/group4D_%(mask)s_space-%(MNI_space)s.nii.gz '
    full_cmd = fslmerge_cmd_pattern%{'group_mask_dir':group_mask_dir,'mask':mask,'MNI_space':MNI_space}
    # append individual subject masks to merge cmd
    for sub in subjects:
        sub_id = 'sub-' + sub
        # get list of masks in MNI space for subject
        mask_dir = mask_dir_pattern%{'sub':sub_id}
        current_mask_path = get_MNI_mask_path(mask_dir, mask, sub_id)
        # append path to merge cmd
        full_cmd = full_cmd + current_mask_path + ' '
    # execute command to merge all subject masks to one nifti
    run_subprocess(full_cmd)

def average_and_threshold_MNI_masks(mask):
    """
    Averages, thresholds and binarizes MNI masks from 4D mask to obtain one 
    group mask. E.g. if threshold is set to 0.5 the resulting MNI group mask
    only contains voxels that at least 50% of participants have in their MNI
    mask.
    mask: mask label
    Returns: nothing, but creates averaged (group) mask in group_mask_dir
    """
    # get input & output mask labels
    input_mask = '%(group_mask_dir)s/group4D_%(mask)s_space-%(MNI_space)s.nii.gz'
    output_mask = '%(group_mask_dir)s/group_%(mask)s_space-%(MNI_space)s.nii.gz'
    # make fslmaths cmd pattern
    fslmaths_cmd_pattern = 'fslmaths %(input_mask)s -Tmean -thr %(threshold)s -bin %(output_mask)s'%{'input_mask':input_mask, 'threshold':threshold, 'output_mask':output_mask}
    # fill placeholders
    full_cmd = fslmaths_cmd_pattern%{'group_mask_dir':group_mask_dir,'mask':mask,'MNI_space':MNI_space}
    # execute command to average all subject masks to form one group mask
    run_subprocess(full_cmd)

def get_MNI_mask_path(mask_dir, mask, sub_id):
    """
    Get path to current MNI mask for subject.
    mask_dir: path to this subjects masks.
    mask: mask label
    sub_id: subject ID
    Returns: path to MNI mask or warning if not exactly one mask is found
    """
    from glob import glob
    MNI_path = mask_dir + '/*' + mask + '*space-MNI*.nii.gz'
    MNI_mask_path = glob(MNI_path)
    if len(MNI_mask_path) == 0:
        print('! WARNING: no MNI mask ' + mask + ' | found for subject: ' + str(sub_id) + '. Returning empty string !!!' )
        MNI_mask_path = ''
    elif len(MNI_mask_path) == 1:
        print('. Getting MNI mask ' + mask + ' | for subject: ' + str(sub_id))
        MNI_mask_path = MNI_mask_path[0]
    elif len(MNI_mask_path) > 1:
        print('! WARNING: multiple/ambigious MNI masks ' + mask + ' | found for subject: ' + str(sub_id) + '. Returning empty string !!!' )
        MNI_mask_path = ''
    return MNI_mask_path
    

# %% run
if __name__ == '__main__':
    """
    Merge MNI masks in mask_list into one 4D nifti file containing masks for 
    all subjects in bids_dir. Then averages across participant masks, 
    thresholds, binarizes and writes output group mask to group_mask_dir 
    containing the final group MNI mask.
    """
    
    # make group mask dir
    if not os.path.isdir(group_mask_dir):
        os.makedirs(group_mask_dir)
    
    # loop over masks
    for mask in mask_list:
        print('Merging MNI space masks for mask: ' + mask)
        
        # merge individual subject masks into one 4D nifti
        merge_MNI_mask_to_4d_file(mask, mask_dir_pattern)
            
        # average, threshold and binarize 4D mask file to get average group mask
        average_and_threshold_MNI_masks(mask)
        