#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes C (pNCCs) conjunction maps and excludes voxels that are included either
in an A (task goals) conjunction or a B (task relevance) conjunction.


@author:Yamil Vidal (hvidaldossantos@gmail.com)

"""

# %% Imports & parameters
import numpy as np
import os, sys

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

# paths
bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code/Yamil/fMRI' 

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

data_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri


# %% functions to load group copes and run conjunctions
def save_conjunction_map(conjunction_name, conjunction_map, group_mask, data_dir):
    """
    Save conjunction maps as binary map.
    conjunction_name: name of conjunction
    conjunction_map: binary conjunction map
    group_mask: group mask path
    output_dir: output dir where nifti files are written
    """
    
    save_fname = data_dir + os.sep + conjunction_name + '_not_A_or_B.nii.gz'
    save_mri(conjunction_map, group_mask, save_fname)
        


# %% run
if __name__ == '__main__':
    
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
                         'C_Object_Deactivation_conjunction']
    
    A = np.squeeze(load_mri(data_dir + os.sep + 'A_conjunction.nii.gz', group_mask))
    B = np.squeeze(load_mri(data_dir + os.sep + 'B_conjunction.nii.gz', group_mask))
    
    for conjunction_name in conjunction_names:
        C = np.squeeze(load_mri(data_dir + os.sep + conjunction_name + '.nii.gz', group_mask))
        
        # print(sum(np.logical_or(A, B)))
        print(conjunction_name + ': ' + str(sum(C[np.logical_or(A, B)])))
        
        C[np.logical_or(A, B)] = 0
        
        save_conjunction_map(conjunction_name, C, group_mask, data_dir)
    
    
