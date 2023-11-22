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

#data_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type
data_subject_dir = projectRoot + '/bids/derivatives/putative_ncc_subject_level'
data_glm_dir = projectRoot + '/bids/derivatives/fslFeat'
# subject level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
brain_mask_pattern = data_glm_dir + '/%(sub_id)s/ses-V1/%(sub_id)s_ses-V1_task-Dur_analysis-2ndGLM_space-MNI152NLin2009cAsym.gfeat/mask.nii.gz'

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri, get_subject_list


# %% functions to load group copes and run conjunctions
def save_conjunction_map(conjunction_name, conjunction_map, brain_mask, data_dir):
    """
    Save conjunction maps as binary map.
    conjunction_name: name of conjunction
    conjunction_map: binary conjunction map
    brain_mask: subject mask path
    output_dir: output dir where nifti files are written
    """

    save_fname = data_dir + os.sep + conjunction_name + '_not_A_or_B.nii.gz'
    save_mri(conjunction_map, brain_mask, save_fname)



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

    #subject_list_type = 'debug'
    subjects = get_subject_list(bids_dir, subject_list_type)

    remove_subjects = ['sub-CD122','sub-CD196']
    for r in remove_subjects:
        subjects = subjects[subjects != r]

    print('Removed subjects:',remove_subjects)
    print('Total subjects:',len(subjects))

    for sub_id in subjects:

        brain_mask = brain_mask_pattern%{'sub_id':sub_id}

        A = np.squeeze(load_mri(data_subject_dir + os.sep + sub_id + os.sep + 'A_conjunction.nii.gz', brain_mask))
        B = np.squeeze(load_mri(data_subject_dir + os.sep + sub_id + os.sep + 'B_conjunction.nii.gz', brain_mask))

        for conjunction_name in conjunction_names:
            C = np.squeeze(load_mri(data_subject_dir + os.sep + sub_id + os.sep + conjunction_name + '.nii.gz', brain_mask))

            # print(sum(np.logical_or(A, B)))
            print(conjunction_name + ': ' + str(sum(C[np.logical_or(A, B)])))

            C[np.logical_or(A, B)] = 0

            save_conjunction_map(conjunction_name, C, brain_mask, data_subject_dir + os.sep + sub_id)
