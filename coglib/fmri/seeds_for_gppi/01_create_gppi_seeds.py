#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:02:20 2022

@author: Yamil Vidal

"""
import os, sys
import numpy as np
from datetime import datetime

projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

bids_dir = projectRoot + '/bids'
code_dir = projectRoot + '/bids/code'
data_dir = projectRoot + '/bids/derivatives/fslFeat'

mask_dir_pattern = bids_dir + '/derivatives/masks/%(sub_id)s'
cope_pattern = data_dir + '/%(sub_id)s/ses-V1/%(sub_id)s_ses-V1_task-Dur_analysis-2ndGLM_space-MNI152NLin2009cAsym.gfeat/cope%(cope_num)s.feat/stats/zstat1.nii.gz'
brain_mask_pattern = data_dir + '/%(sub_id)s/ses-V1/%(sub_id)s_ses-V1_task-Dur_analysis-2ndGLM_space-MNI152NLin2009cAsym.gfeat/mask.nii.gz'

output_pattern = bids_dir + '/derivatives/gppi_seeds/new/%(sub)s'
f_name_pattern = '/%(sub)s_%(rel)s_bh_%(cond)s_%(roi)s_n_voxels_%(n_voxels)s_space-%(space)s.nii.gz'

# Get the current date and time as a string
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
log_file = bids_dir + '/derivatives/gppi_seeds/roi_size_log_' + dt_string + '.txt'

subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import get_subject_list, load_mri, save_mri

space = 'MNI152NLin2009cAsym'

#roi_sizes = [300, 50, 100, 200]
roi_sizes = [300]

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

def load_a_rois(sub_id):
    """
    Load all anatomica ROIs of a subject into a dictionary

    sub_id: Subject ID
    Returns: Dictionary containing all the anatomical ROIs of the subject
    """
    sub_mask_dir = mask_dir_pattern%{'sub_id':sub_id}
    mask_paths = get_mask_list(sub_mask_dir)
    mask_paths.sort()

    sub_mask_list = [l[95:] for l in mask_paths]
    sub_mask_list = [l[:-33] for l in sub_mask_list]

    # empty dictionary that will contain all masks of a subject
    a_rois = {}

    for n in range(0,len(mask_paths)):
        mask = mask_paths[n]
        m = load_mri(mask, brain_mask)
        a_rois[sub_mask_list[n]] = m

    return a_rois, sub_mask_list


def make_seeds(sub_id,a_roi,cope,n_voxels):

    # Load the relevant functional maps
    f_roi = load_mri(cope_pattern%{'sub_id':sub_id,'cope_num':cope},brain_mask)

    # Load StimAll_vs_baseline
    vis_driven = load_mri(cope_pattern%{'sub_id':sub_id,'cope_num':'16'},brain_mask)

    # Set to zero the voxels that don't belong to the ROI
    f_roi[a_roi == 0] = 0
    f_roi[vis_driven < 1] = 0

    # Pick only non-zero voxels
    idxnz = np.nonzero(-f_roi)[0]
    idx_pos = idxnz[np.argsort(-f_roi[idxnz])[:n_voxels]]
    idx_neg = idxnz[np.argsort(f_roi[idxnz])[:n_voxels]]

    ffa_seed = np.zeros(f_roi.shape)
    loc_seed = np.zeros(f_roi.shape)
    ffa_seed[idx_pos] = 1
    loc_seed[idx_neg] = 1

    return ffa_seed, loc_seed, f_roi

def check_for_zeros(d_roi,f_roi,f_name):
    if any(f_roi[d_roi.astype(bool)]==0):

        message = 'Found ' + str(sum((f_roi[d_roi.astype(bool)]==0))) + ' zeros in ' + f_name + '\n'
        print('Seed contains zeros!! Check log file\n')

        try:
            with open(log_file, 'a') as file:
                file.write(message)
        except FileNotFoundError:
            with open(log_file, 'w') as file:
                file.write(message)

def check_roi_size(d_roi,f_name,n_voxels):
    if np.sum(d_roi) != n_voxels:

        message = str(int(np.sum(d_roi))) + ' instead of ' + str(n_voxels) + ' ' + f_name + '\n'
        print('Unexpected number of voxels in decoding ROI!! Check log file\n')

        try:
            with open(log_file, 'a') as file:
                file.write(message)
        except FileNotFoundError:
            with open(log_file, 'w') as file:
                file.write(message)

def save_seed(sub_id,d_roi,f_name):

    output_dir = output_pattern%{'sub':sub_id}
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    full_f_name = output_dir + f_name
    save_mri(d_roi, brain_mask,full_f_name)

# %% run
subjects = get_subject_list(bids_dir, subject_list_type)

remove_subjects = ['sub-CD122','sub-CD196']
for r in remove_subjects:
    subjects = subjects[subjects != r]

print('Removed subjects:',remove_subjects)
print('Total subjects:',len(subjects))


cope = '23' # RelIrrelFacesvsObject
rel = 'rel_irrel'

for n_voxels in roi_sizes:

    for sub_id in subjects:
        brain_mask = brain_mask_pattern%{'sub_id':sub_id}
        a_rois, sub_roi_list = load_a_rois(sub_id)

        roi_name = 'FFA'
        print(sub_id + ' ' + 'ROI: ' + roi_name + ' n_voxels: ' + str(n_voxels))
        a_roi = np.squeeze(a_rois[roi_name])

        ffa_seed, loc_seed, f_roi = make_seeds(sub_id,a_roi,cope,n_voxels)
        f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'cond':'face','roi':roi_name,'n_voxels':n_voxels,'space':space}
        check_for_zeros(ffa_seed,f_roi,f_name)
        check_roi_size(ffa_seed,f_name,n_voxels)
        save_seed(sub_id,ffa_seed,f_name)

        roi_name = 'LOC'
        print(sub_id + ' ' + 'ROI: ' + roi_name + ' n_voxels: ' + str(n_voxels))
        a_roi = np.squeeze(a_rois[roi_name])

        ffa_seed, loc_seed, f_roi = make_seeds(sub_id,a_roi,cope,n_voxels)
        f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'cond':'object','roi':roi_name,'n_voxels':n_voxels,'space':space}
        check_for_zeros(loc_seed,f_roi,f_name)
        check_roi_size(loc_seed,f_name,n_voxels)
        save_seed(sub_id,loc_seed,f_name)
