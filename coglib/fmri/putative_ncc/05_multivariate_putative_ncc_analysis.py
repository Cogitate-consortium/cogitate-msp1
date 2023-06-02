#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs the multivariate putative NCC (pNCC) analysis. Only C contrasts are 
performed, as A and B contrasts require Target data, and no decoding in possible
for target conditions due to the low number of trials.


Created on Wed Jan 02 2023

@author: Yamil Vidal (yamil.vidal@donders.ru.nl)

"""

# %% Imports & parameters
import numpy as np
import os, sys

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'
# import functions from helper_functions_MRI
code_dir = projectRoot + '/bids/code' 
bids_dir = projectRoot + '/bids'

sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri

# This should match the subject list that was used to perform the group level decoding
subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

conditions = ['face_baseline', 'object_baseline', 'letter_baseline',
                  'falseFont_baseline', 'face_object', 'letter_falseFont']

if subject_list_type == 'phase2_V1':
    data_dir = projectRoot + '/bids/derivatives/decoding/nibetaseries/searchlight_decoding/phase2/within_condition/'
elif subject_list_type == 'phase3_V1':
    data_dir = projectRoot + '/bids/derivatives/decoding/nibetaseries/searchlight_decoding/within_condition/'
else:
    raise Exception('Unknown subject list') 

decoding_group_results_template = data_dir + '/%(rel)s/SVM/4mm/category/%(category)s/searchlight_group_accuracy_map_nonparametric.nii'

fslFeat_dir = projectRoot + '/bids/derivatives/fslFeat'


output_dir = projectRoot + '/bids/derivatives/putative_ncc_multivariate/' + subject_list_type

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask = fslFeat_dir + '/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'




##############################################################################

# Functions to make various conjunctions
def make_C_conjunction(cond, group_mask, output_dir):
    """
    Make a C style conjunction map of decoding zmaps; i.e. 
    [Rel(id) > chance] & [Irrel(id) > chance]
    
    cond: Condition
    output_dir: output dir where nifti files are written
    group_mask: group mask path
    Returns:
    group_mask: utilized group mask path
    """
    print('. Creating C conjunction map for: ' + cond)
    
    # Relevant
    data_path = decoding_group_results_template%{'rel':'relevant','category':cond}
    data1 = load_mri(data_path, group_mask)
    
    # Irrelevant
    data_path = decoding_group_results_template%{'rel':'irrelevant','category':cond}
    data2 = load_mri(data_path, group_mask)
    
    
    conjunction_map = np.logical_and(data1>0,data2>0)
    # save maps
    fname = output_dir + os.sep + 'Multivariate_C_' + cond + '_conjunction.nii.gz'
    save_mri(conjunction_map, group_mask, fname)
    
    return group_mask
       
# create merged C conjunction map
def merge_C_conjunction_maps(conditions, group_mask, output_dir):
    """
    Merges C conjunction maps together; i.e. the C conjunction in the prereg. 
    specifies that this conjunction is run separately for each stimulus 
    category. However, for display purposes it can help to merge related
    stimuli togehter and output the union of the maps.
    
    """
    print('Creating merged C conjunction map')
    
    # load example data and preallocate array
    data = load_mri(group_mask,group_mask)
    data = np.zeros((len(data),len(conditions)))
    
    # load data
    for cidx in range(len(conditions)):
        fname = output_dir + os.sep + 'Multivariate_C_' + conditions[cidx] + '_conjunction.nii.gz'
        data[:,cidx] = np.squeeze(load_mri(fname, group_mask))
    
    # merge data
    merged_c_map = np.min(data,1)
    fname = output_dir + os.sep + 'Multivariate_C_combined_and_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.max(data,1)
    fname = output_dir + os.sep + 'Multivariate_C_combined_or_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.sum(data,1)
    fname = output_dir + os.sep + 'Multivariate_C_combined_sum_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    

# %% run
if __name__ == '__main__':
    # make output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # loop over pncc conjunctions and create conjunction maps
    for cond in conditions:
        print(cond)
        # make different conjunction styles depending on conjunction name; i.e. A, B or C style conjunctions (letters refer to preregistration conjunction labels)
        group_mask = make_C_conjunction(cond, group_mask, output_dir)
    
    
    # merge all C style conjunctions to one map
    conditions.remove('face_object')
    conditions.remove('letter_falseFont')
    merge_C_conjunction_maps(conditions, group_mask, output_dir)
