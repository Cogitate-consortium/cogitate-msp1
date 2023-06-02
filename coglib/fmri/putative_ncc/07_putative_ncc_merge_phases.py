#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combines putative NCC results across optimization and replication phases for plotting purposes.

@author:Yamil Vidal (yamil.vidal@donders.ru.nl)

"""

# %% Imports & parameters
import numpy as np
import os, sys
import nibabel as nib

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

# paths
bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code/Yamil/fMRI' 

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type_1 = 'phase2_V1'
subject_list_type_2 = 'phase3_V1'
#subject_list_type = 'debug'

data_dir_1 = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type_1
data_dir_2 = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type_2

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask_1 = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type_1 + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'
group_mask_2 = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type_2 + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

output_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type_1 + '_' + subject_list_type_2
#output_dir = projectRoot + '/bids/derivatives/putative_ncc/'

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri


# %% run
if __name__ == '__main__':
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    conjunction_names = ['C_combined_and_Activation_conjunction',
                         'C_combined_and_Deactivation_conjunction',
                         'C_combined_or_Activation_conjunction',
                         'C_combined_or_Deactivation_conjunction',
                         'C_combined_sum_Activation_conjunction',
                         'C_combined_sum_Deactivation_conjunction',
                         'C_Face_conjunction',
                         'C_FalseFont_conjunction',
                         'C_Letter_conjunction',
                         'C_Object_conjunction',
                         'C_Face_Activation_conjunction',
                         'C_FalseFont_Activation_conjunction',
                         'C_Letter_Activation_conjunction',
                         'C_Object_Activation_conjunction',
                         'C_Face_Deactivation_conjunction',
                         'C_FalseFont_Deactivation_conjunction',
                         'C_Letter_Deactivation_conjunction',
                         'C_Object_Deactivation_conjunction',
                         'A_conjunction',
                         'B_conjunction']
    
    # Create a shared brain mask
    mask_img = nib.load(group_mask_1)

    M1 = np.asarray(nib.load(group_mask_1).get_fdata(), dtype=np.float32)
    M2 = np.asarray(nib.load(group_mask_2).get_fdata(), dtype=np.float32)
    
    group_mask = np.logical_or(M1, M2)
    
    group_mask = nib.Nifti1Image(group_mask.astype(np.int16), mask_img.affine)
    
    group_mask_dir = output_dir + os.sep + 'group_mask_combined.nii.gz'
    nib.save(group_mask, group_mask_dir)
    
    group_mask = np.squeeze(load_mri(group_mask_dir, group_mask_dir))
        
    for conjunction_name in conjunction_names:
    
        print(conjunction_name)
            
        data = np.zeros((len(group_mask),1))
        
        if conjunction_name[0] == 'C':
            
            C1 = np.squeeze(load_mri(data_dir_1 + os.sep + conjunction_name + '_not_A_or_B.nii.gz', group_mask_dir))
            C2 = np.squeeze(load_mri(data_dir_2 + os.sep + conjunction_name + '_not_A_or_B.nii.gz', group_mask_dir))
            
            fname = output_dir + os.sep + conjunction_name + '_not_A_or_B_merged_phases.nii.gz'
            
        else:
            
            C1 = np.squeeze(load_mri(data_dir_1 + os.sep + conjunction_name + '.nii.gz', group_mask_dir))
            C2 = np.squeeze(load_mri(data_dir_2 + os.sep + conjunction_name + '.nii.gz', group_mask_dir))
            
            fname = output_dir + os.sep + conjunction_name + '_merged_phases.nii.gz'
        
        data[C1 == 1] = 1
        data[C2 == 1] = 2
        data[np.logical_and(C1 == 1, C2 == 1)] = 3
        
        
        save_mri(data, group_mask_dir, fname)
        