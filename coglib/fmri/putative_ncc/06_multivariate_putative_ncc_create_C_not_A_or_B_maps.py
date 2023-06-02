#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes C (pNCCs) conjunction maps and excludes voxels that are included either
in an A (task goals) conjunction or a B (task relevance) conjunction.


@author:Yamil Vidal (yamil.vidal@donders.ru.nl)

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

data_dir = projectRoot + '/bids/derivatives/putative_ncc_multivariate/' + subject_list_type
data_dir_AB_maps = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri


# %% functions to load group copes and run conjunctions
def save_conjunction_map(conjunction_name, conjunction_map, group_mask, data_dir):
    
    save_fname = data_dir + os.sep + conjunction_name + '_not_A_or_B.nii.gz'
    save_mri(conjunction_map, group_mask, save_fname)
        


# %% run
if __name__ == '__main__':
    
    conjunction_names = ['Multivariate_C_combined_and_conjunction',
                         'Multivariate_C_combined_or_conjunction',
                         'Multivariate_C_combined_sum_conjunction',
                         'Multivariate_C_face_baseline_conjunction',
                         'Multivariate_C_falseFont_baseline_conjunction',
                         'Multivariate_C_letter_baseline_conjunction',
                         'Multivariate_C_object_baseline_conjunction',
                         'Multivariate_C_face_object_conjunction',
                         'Multivariate_C_letter_falseFont_conjunction']
    
    A = np.squeeze(load_mri(data_dir_AB_maps + os.sep + 'A_conjunction.nii.gz', group_mask))
    B = np.squeeze(load_mri(data_dir_AB_maps + os.sep + 'B_conjunction.nii.gz', group_mask))
    
    for conjunction_name in conjunction_names:
        C = np.squeeze(load_mri(data_dir + os.sep + conjunction_name + '.nii.gz', group_mask))
        
        # print(sum(np.logical_or(A, B)))
        print(conjunction_name + ': ' + str(sum(C[np.logical_or(A, B)])))
        
        C[np.logical_or(A, B)] = 0
        
        save_conjunction_map(conjunction_name, C, group_mask, data_dir)
    
    
