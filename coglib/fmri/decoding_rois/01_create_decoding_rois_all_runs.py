#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates regions of interes (ROIs) for several decoding analyses.
Decoding ROIs are produced for each anatomical region of interest (step 7).
Requires several contrasts of parameter estimates produced in step 9.

@author: Yamil Vidal
Email: hvidaldossantos@gmail.com
Created on Tue Jun 21 11:02:20 2022
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

output_pattern = bids_dir + '/derivatives/decoding_rois/%(sub)s'
f_name_pattern = '/%(sub)s_%(rel)s_bh_%(deco)s_n_voxels_%(n_voxels)s_%(roi)s_space-%(space)s.nii.gz'

# Get the current date and time as a string
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
log_file = bids_dir + '/derivatives/decoding_rois/deco_roi_log_' + dt_string + '.txt'

subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import get_subject_list, load_mri, save_mri

space = 'MNI152NLin2009cAsym'

#roi_sizes = [300, 50, 100, 200]
roi_sizes = [300]

roi_list = ['G_and_S_cingul-Ant',
            'G_and_S_cingul-Mid-Ant',
            'G_and_S_cingul-Mid-Post',
            'G_and_S_occipital_inf',
            'G_cuneus',
            'G_front_inf-Opercular',
            'G_front_inf-Orbital',
            'G_front_inf-Triangul',
            'G_front_middle',
            'G_oc-temp_lat-fusifor',
            'G_oc-temp_med-Lingual',
            'G_oc-temp_med-Parahip',
            'G_occipital_middle',
            'G_occipital_sup',
            'G_orbital',
            'G_pariet_inf-Angular',
            'G_pariet_inf-Supramar',
            'G_precentral',
            'G_temp_sup-Lateral',
            'G_temp_sup-Plan_tempo',
            'G_temporal_inf',
            'G_temporal_middle',
            'Lat_Fis-ant-Horizont',
            'Lat_Fis-ant-Vertical',
            'Pole_occipital',
            'Pole_temporal',
            'S_calcarine',
            'S_front_inf',
            'S_front_middle',
            'S_front_sup',
            'S_interm_prim-Jensen',
            'S_intrapariet_and_P_trans',
            'S_oc-temp_lat',
            'S_oc_middle_and_Lunatus',
            'S_oc_sup_and_transversal',
            'S_occipital_ant',
            'S_precentral-inf-part',
            'S_temporal_inf',
            'S_temporal_sup',
            'G_and_S_frontomargin',
            'G_and_S_transv_frontopol',
            'G_front_sup',
            'G_rectus',
            'G_subcallosal',
            'S_orbital_lateral',
            'S_orbital_med-olfact',
            'S_orbital-H_Shaped',
            'S_suborbital',
            'GNW',
            'GNW_S_front_inf',
            'IIT',
            'IIT_extended',
            'IIT_excluded']

roi_list = ['GNW',
            'GNW_S_front_inf']

# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = True

# %%
# load all anatomical ROIs
def get_mask_list(sub_mask_dir, process_only_bilateral_masks=True):
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


def make_d_roi(sub_id,a_roi,cope,n_voxels):
    
    # Load the relevant functional maps
    f_roi = load_mri(cope_pattern%{'sub_id':sub_id,'cope_num':cope},brain_mask)
    
    # Set to zero the voxels that don't belong to the ROI
    f_roi[a_roi == 0] = 0
    
    # Pick only non-zero voxels
    idxnz = np.nonzero(-f_roi)[0]
    idx_pos = idxnz[np.argsort(-f_roi[idxnz])[:n_voxels//2]]
    idx_neg = idxnz[np.argsort(f_roi[idxnz])[:n_voxels//2]]
    
    d_roi = np.zeros(f_roi.shape)
    d_roi[idx_pos] = 1
    d_roi[idx_neg] = 1
    
    return d_roi, f_roi


def make_d_roi_baseline(sub_id,a_roi,cope,n_voxels):

    # Load the relevant functional maps
    f_roi = load_mri(cope_pattern%{'sub_id':sub_id,'cope_num':cope},brain_mask)
    
    # Set to zero the voxels that don't belong to the ROI
    f_roi[a_roi == 0] = 0
    
    # Pick only non-zero voxels
    idxnz = np.nonzero(-f_roi)[0]
    idx = idxnz[np.argsort(-f_roi[idxnz])[:n_voxels]]

    d_roi = np.zeros(f_roi.shape)
    d_roi[idx] = 1

    return d_roi, f_roi


def check_for_zeros(d_roi,f_roi,f_name):
    if any(f_roi[d_roi.astype(bool)]==0):
        
        message = 'Found ' + str(sum((f_roi[d_roi.astype(bool)]==0))) + ' zeros in ' + f_name + '\n'
        print('ROI contains zeros!! Check log file\n')
        
        try:
            with open(log_file, 'a') as file:
                file.write(message)
        except FileNotFoundError:
            with open(log_file, 'w') as file:
                file.write(message)
                
def check_roi_size(sub_id,d_roi,f_name,n_voxels):
    if np.sum(d_roi) != n_voxels:
        
        message = str(int(np.sum(d_roi))) + ' instead of ' + str(n_voxels) + ' ' + f_name + '\n'
        print('Unexpected number of voxels in decoding ROI! Check log file')
        
        try:        
            with open(log_file, 'a') as file:
                file.write(message)
        except FileNotFoundError:
            with open(log_file, 'w') as file:
                file.write(message)

def save_d_roi(sub_id,d_roi,f_name):
    output_dir = output_pattern%{'sub':sub_id}
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    full_f_name = output_dir + f_name
    save_mri(d_roi, brain_mask,full_f_name)

# %% run
subjects = get_subject_list(bids_dir, subject_list_type)

for n_voxels in roi_sizes:
    #for sub_id in ['sub-SC136']:
    for sub_id in subjects:
        
        brain_mask = brain_mask_pattern%{'sub_id':sub_id}

        a_rois, sub_roi_list = load_a_rois(sub_id)

        for roi_name in roi_list:
        #for roi_name in ['G_front_inf-Orbital']:
        
            print(sub_id + ' Size: ' + str(n_voxels)+ ' ' + roi_name)
            #roi_name = roi_list[0]
            a_roi = np.squeeze(a_rois[roi_name])
            
            rel = 'rel'
            deco = 'face_vs_object'
            cope = '19'
            d_roi, f_roi = make_d_roi(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'rel'
            deco = 'letter_vs_falsefont'
            cope = '20'
            d_roi, f_roi = make_d_roi(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'rel'
            deco = 'face_orientation'
            cope = '5'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'rel'
            deco = 'object_orientation'
            cope = '6'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'rel'
            deco = 'letter_orientation'
            cope = '7'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'rel'
            deco = 'falsefont_orientation'
            cope = '8'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'face_vs_object'
            cope = '21'
            d_roi, f_roi = make_d_roi(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'letter_vs_falsefont'
            cope = '22'
            d_roi, f_roi = make_d_roi(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'face_orientation'
            cope = '9'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'object_orientation'
            cope = '10'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'letter_orientation'
            cope = '11'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)
            
            rel = 'irrel'
            deco = 'falsefont_orientation'
            cope = '12'
            d_roi, f_roi = make_d_roi_baseline(sub_id,a_roi,cope,n_voxels)
            f_name = f_name_pattern%{'sub':sub_id,'rel':rel,'deco':deco,'n_voxels':n_voxels,'roi':roi_name,'space':space}
            check_for_zeros(d_roi,f_roi,f_name)
            check_roi_size(sub_id,d_roi,f_name,n_voxels)
            save_d_roi(sub_id,d_roi,f_name)