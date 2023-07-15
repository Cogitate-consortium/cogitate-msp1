#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combines putative NCC subject level results across optimization and replication phases for plotting purposes.

@author:Yamil Vidal (hvidaldossantos@gmail.com)

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

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri, get_subject_list

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type_1 = 'phase2_V1'
subject_list_type_2 = 'phase3_V1'
#subject_list_type = 'debug'

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask_1 = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type_1 + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'
group_mask_2 = projectRoot + '/bids/derivatives/fslFeat/group/ses-V1/' + subject_list_type_2 + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

#data_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type
data_subject_dir = projectRoot + '/bids/derivatives/putative_ncc_subject_level'
output_dir = data_subject_dir + '/merged/'

# Create a shared brain mask
mask_img = nib.load(group_mask_1)

M1 = np.asarray(nib.load(group_mask_1).get_fdata(), dtype=np.float32)
M2 = np.asarray(nib.load(group_mask_2).get_fdata(), dtype=np.float32)

group_mask = np.logical_or(M1, M2)

group_mask = nib.Nifti1Image(group_mask.astype(np.int16), mask_img.affine)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

group_mask_dir = output_dir + os.sep + 'group_mask_combined.nii.gz'
nib.save(group_mask, group_mask_dir)

group_mask = np.squeeze(load_mri(group_mask_dir, group_mask_dir))

# %% functions to load group copes and run conjunctions
def save_conjunction_map(conjunction_name, conjunction_map, brain_mask, data_dir):
    """
    Save conjunction maps as binary map.
    conjunction_name: name of conjunction
    conjunction_map: binary conjunction map
    brain_mask: subject mask path
    output_dir: output dir where nifti files are written
    """
    
    save_fname = data_dir + os.sep + conjunction_name + '_merged.nii.gz'
    save_mri(conjunction_map, brain_mask, save_fname)
        


# %% run
if __name__ == '__main__':
    
    conjunction_names = ['C_Face_Activation_conjunction',
                          'C_FalseFont_Activation_conjunction',
                          'C_Letter_Activation_conjunction',
                          'C_Object_Activation_conjunction',
                          'C_Face_Deactivation_conjunction',
                          'C_FalseFont_Deactivation_conjunction',
                          'C_Letter_Deactivation_conjunction',
                          'C_Object_Deactivation_conjunction']
    
    #subject_list_type = 'debug'
    subjects1 = get_subject_list(bids_dir, subject_list_type_1)
    subjects2 = get_subject_list(bids_dir, subject_list_type_2)
    
    subjects = np.concatenate((subjects1,subjects2))
    
    remove_subjects = ['sub-SD122','sub-SD196']
    for r in remove_subjects:
        subjects = subjects[subjects != r]
    
    print('Removed subjects:',remove_subjects)
    print('Total subjects:',len(subjects))

    
    for conjunction_name in conjunction_names:
        
        data = np.zeros((len(group_mask),len(subjects)))
            
        for cidx in range(len(subjects)):
            
            data[:,cidx] = np.squeeze(load_mri(data_subject_dir + os.sep + subjects[cidx] + os.sep + conjunction_name + '_not_A_or_B.nii.gz', group_mask_dir))
            print('Loading ',conjunction_name,' subject: ',subjects[cidx])
            
        # merge data
        merged_c_map = np.sum(data,1)
        fname = output_dir + os.sep + conjunction_name + '_merged.nii.gz'
        save_mri(merged_c_map, group_mask_dir, fname)
