#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs the putative NCC (pNCC) analysis. Uses thresholded but uncorrected zmaps.
Requires that the corrected version of the pNCC analysis has been run.

Created on Feb 23 2023

@author: Yamil Vidal (yamil.vidal@donders.ru.nl)

"""

# %% Imports & parameters
import numpy as np
import os, sys

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

# paths
bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 
data_dir = projectRoot + '/bids/derivatives/fslFeat'


# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
#subject_list_type = 'phase2_V1'
subject_list_type = 'phase3_V1_subset'
#subject_list_type = 'debug'


output_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri, get_subject_list

# %% Parameters and analysis definitions
# define cope paths and cope templates; i.e. how the feat copes map to contrast labels and where to find them (number of feat dir)

# path template to group level cope results (i.e. from group level feat analyses)
group_results_template = data_dir + '/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM%(path_ext)s_space-MNI152NLin2009cAsym_desc-%(cope)s.gfeat/%(subCope)s.feat/'

# group level mask to be used for all analyses (i.e. all subject data should be normalized to this space)
group_mask = data_dir + '/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'

# Labels and cope numbers of all contrasts of interest
# CAUTION: contrast labels and cope numbers must correspond to those defined in the fsf file of the 1st/2nd level feat analyses!
copes = {
        'TarFace': {'cope': 'cope1', 'path_ext': 'TarFace'},
        'TarObject': {'cope': 'cope2', 'path_ext': 'TarObject'},
        'TarLetter': {'cope': 'cope3','path_ext': 'TarLetter'},
        'TarFalseFont': {'cope': 'cope4','path_ext': 'TarFalseFont'},
        'RelFace': {'cope': 'cope5', 'path_ext': ''},
        'RelObject': {'cope': 'cope6','path_ext': ''},
        'RelLetter': {'cope': 'cope7', 'path_ext': ''},
        'RelFalseFont': { 'cope': 'cope8', 'path_ext': ''},
        'IrrelFace': {'cope': 'cope9', 'path_ext': ''},
        'IrrelObject': {'cope': 'cope10', 'path_ext': ''},
        'IrrelLetter': {'cope': 'cope11','path_ext': ''},
        'IrrelFalseFont': {'cope': 'cope12', 'path_ext': ''},
        'TarAll': {'cope': 'cope13', 'path_ext': ''},
        'RelAll': { 'cope': 'cope14', 'path_ext': ''},
        'IrrelAll': {'cope': 'cope15', 'path_ext': ''}
        }

# sub copes; sub contrasts per cope separately testing significant activation or deactivation to allow for directional or bidirectional conjunctions
subCopes = {
        'Activation': {'subCope':'cope1'},
        'Deactivation': {'subCope':'cope2'}
        }

"""
Define contrasts run for pNCC analysis (note these are different 
contrasts then those run on the 1st/2nd level of FEAT analyses, but rather use 
2nd level feat contrasts as their input). Subsequently these serve as input to 
the conjunction analyses.

Adjusted from Preregistration:
A)	[Tar > bsl] & [Rel = bsl] & [Irrel = bsl] 
    [TarAll > 0] && [RelAll == 0] && [IrrelAll == 0]
B)	[Tar > bsl] & [Rel != bsl] & [Irrel = bsl]
    [TarAll > 0] && [(RelAll > 0) || (RelAll < 0)] && [IrrelAll == 0]
C)	[Rel(id) != bsl] & [Irrel(id) != bsl]
    [(Rel(id) > 0) && (Irrel(id) > 0)] || [(Rel(id) < 0) && (Irrel(id) < 0)]
"""
# C conjunctions are performed as: conjunction_map = (con1A || con1B) && (con2A || con2B)
# note: the first letter of the conjunctio name defines the type of conjunction performed; e.g. C_face will run a C type conjunction (see above)
pncc_conjunctions = {
        'C_Face_uncorrected': {
        'con1A': {'cope_of_interest':'RelFace', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelFace', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelFace', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelFace', 'test_direction':'Deactivation'}
        },
        'C_Object_uncorrected': {
        'con1A': {'cope_of_interest':'RelObject', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelObject', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelObject', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelObject', 'test_direction':'Deactivation'}
        },
        'C_Letter_uncorrected': {
        'con1A': {'cope_of_interest':'RelLetter', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelLetter', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelLetter', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelLetter', 'test_direction':'Deactivation'}
        },
        'C_FalseFont_uncorrected': {
        'con1A': {'cope_of_interest':'RelFalseFont', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelFalseFont', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelFalseFont', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelFalseFont', 'test_direction':'Deactivation'}
        },
        'A_uncorrected': {
        'con1': {'cope_of_interest':'TarAll', 'test_direction':'Activation'},
        'con2': {'cope_of_interest':'RelAll', 'test_direction':'Equivalence'},
        'con3': {'cope_of_interest':'IrrelAll', 'test_direction':'Equivalence'}
        },
        'B_uncorrected': {
        'con1': {'cope_of_interest':'TarAll', 'test_direction':'Activation'},
        'con2A': {'cope_of_interest':'RelAll', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'RelAll', 'test_direction':'Deactivation'},
        'con3': {'cope_of_interest':'IrrelAll', 'test_direction':'Equivalence'}
        }
        }


##############################################################################

# %% functions to load group copes and run conjunctions
def save_conjunction_map(conjunction_name, conjunction_map, group_mask, output_dir):
    """
    Save conjunction maps as binary map & associated z stat map as nifti files.
    conjunction_name: name of conjunction
    zmap: z statistic map
    conjunction_map: binary conjunction map
    group_mask: group mask path
    output_dir: output dir where nifti files are written
    """
    save_fname = output_dir + os.sep + conjunction_name + '_conjunction.nii.gz'
    save_mri(conjunction_map, group_mask, save_fname)
        
def load_group_zmap(cope_of_interest, test_direction, group_mask_fname='mask.nii.gz',  return_bool=False):
    """
    Gets uncorrected zmaps for C conjunctions.
    cope_of_interest: cope name of interest to be loaded (e.g. RelFace to load relevant faces > baseline contrast)
    test direction: direction of statistical test to load; corresponds to subCopes (e.g. Activation to load RelFace>Baseline)
    return_bool: if true returns a boolean map instead of z values (default false)
    group_mask_fname: (optional) full path to group mask used to mask the MRI data. If left empty the gfeat mask will be used corresponding to the loaded contrast 
    Returns:
    data: Uncorrected zmap as vector
    group_mask: utilized group mask path
    """
    # get information from copes & subcopes
    cope = copes[cope_of_interest]['cope']
    path_ext = copes[cope_of_interest]['path_ext']
    subCope = subCopes[test_direction]['subCope']
    print('. . loading ' + cope + '.gfeat, ' + subCope + '.feat | ' + test_direction + ' to relative baseline' )
    # make data path
    data_path = group_results_template%{'path_ext':path_ext,'cope':cope,'subCope':subCope}
    
    # Uses data_fname to pick the zmap
    data_fname='stats/zstat1.nii.gz'
    data_final_path = data_path + data_fname
    if group_mask_fname == 'mask.nii.gz':
        group_mask = data_path + group_mask_fname
    else:
        group_mask = group_mask_fname
    # load mri data
    data = load_mri(data_final_path, group_mask)
    if return_bool:
        data = np.asarray(data,dtype='bool')
    
    # Thresholds the zmap (z < 3.10 = p < 0.001)
    # Thresholds the zmap (z < 2.57 = p < 0.005)
    data[data < 2.57] = 0
    
    return data, group_mask

def load_group_bayesian(cope_of_interest, test_direction, group_mask, data_fname='thresh.nii.gz',  return_bool=False):
    """
    Gets thresholded maps for group bayesian test; i.e. significance maps 
    for A or B conjunctions.
    cope_of_interest: cope name of interest to be loaded (e.g. RelFace to load relevant faces == baseline contrast)
    test direction: direction of statistical test to load; corresponds to subCopes (must be Equivalence to load e.g. RelFace==Baseline)
    return_bool: if true returns a boolean map instead of z values (default false)
    data_fname: (optional) file name for contast of interest; appended to path. If left empty z stat will be used
    group_mask: full path to group mask used to mask the MRI data. If left empty the gfeat mask will be used corresponding to the loaded contrast 
    Returns:
    data: mri map as vector
    group_mask: utilized group mask path
    """
    # get information from copes & subcopes
    cope = copes[cope_of_interest]['cope']
    print('. . loading ' + cope + ' | ' + test_direction + ' relative to baseline')
    # make data path
    bayesian_group_results_path = output_dir + os.sep + 'bayesian_maps' 
    data_final_path = bayesian_group_results_path + os.sep + cope_of_interest + '_bayesian_map_'  + data_fname
    # load mri data
    data = np.squeeze(load_mri(data_final_path, group_mask))
    if return_bool:
        data = np.asarray(data,dtype='bool')
    return data, group_mask

# Functions to make various conjunctions
def make_C_directional_conjunction(conjunction_name, conjunction_info, group_mask, output_dir):
    """
    Make a C style conjunction map, but only for contrasts of same 
    direction (activation or deactivation) for rel and irrel; i.e.:
    [Rel(id) > bsl] & [Irrel(id) > bsl]
    and
    [Rel(id) < bsl] & [Irrel(id) < bsl]
    This is an addition to the conjunction in the prereg.
    Save map as binary & associated z stat map as nifti files.
    conjunction_name: name of conjunction with activation and deactivation suffix
    conjunction_info: contains information about copes of interest and test direction to identiy correct gfeat data to be loaded for conjunction
    output_dir: output dir where nifti files are written
    group_mask: group mask path
    Returns:
    group_mask: utilized group mask path
    """
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con1A']['cope_of_interest']
    test_direction_1A = conjunction_info['con1A']['test_direction']
    data1A, group_mask = load_group_zmap(cope_of_interest, test_direction_1A, group_mask)
    # get map for conjunction part 1B, deactivation
    cope_of_interest = conjunction_info['con1B']['cope_of_interest']
    test_direction_1B = conjunction_info['con1B']['test_direction']
    data1B, group_mask = load_group_zmap(cope_of_interest, test_direction_1B, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction_2A = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_zmap(cope_of_interest, test_direction_2A, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction_2B = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_zmap(cope_of_interest, test_direction_2B, group_mask)
    # make directional C conjunction: Activation
    if test_direction_1A == test_direction_2A:
        conjunction_name_A = conjunction_name + '_' + test_direction_1A
        print('. Creating C directional conjunction map for: ' + conjunction_name_A)
        
        conjunction_map = np.logical_and(data1A>0, data2A>0)
        # save maps
        save_conjunction_map(conjunction_name_A, conjunction_map, group_mask, output_dir)
    else:
        print('. ! Caution: conjunction test direction do not match. No map created for C Directional conjunction C.A')
    # make directional C conjunction: Deactivation
    if (test_direction_1B == test_direction_2B):
        conjunction_name_B = conjunction_name + '_' + test_direction_1B
        print('. Creating C directional conjunction map for: ' + conjunction_name_B)
        
        conjunction_map = np.logical_and(data1B>0, data2B>0)
        # save maps
        save_conjunction_map(conjunction_name_B, conjunction_map, group_mask, output_dir)
    else: 
        print('. ! Caution: conjunction test direction do not match. No map created for C Directional conjunction C.B')
    return group_mask

# Functions to make various conjunctions
def make_C_conjunction(conjunction_name, conjunction_info, group_mask, output_dir):
    """
    Make a C style conjunction map; i.e. 
    [Rel(id) != bsl] & [Irrel(id) != bsl]
    conjunction_map = (con1A || con1B) && (con2A || con2B)
    Save map as binary & associated z stat map as nifti files.
    conjunction_name: name of conjunction
    conjunction_info: contains information about copes of interest and test direction to identiy correct gfeat data to be loaded for conjunction
    output_dir: output dir where nifti files are written
    group_mask: group mask path
    Returns:
    group_mask: utilized group mask path
    """
    print('. Creating C conjunction map for: ' + conjunction_name)
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con1A']['cope_of_interest']
    test_direction = conjunction_info['con1A']['test_direction']
    data1A, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 1B, deactivation
    cope_of_interest = conjunction_info['con1B']['cope_of_interest']
    test_direction = conjunction_info['con1B']['test_direction']
    data1B, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # make binary conjunction map
    
    conjunction_map = np.logical_or(np.logical_and(data1A>0,data2A>0), np.logical_and(data1B>0,data2B>0))
    # save maps
    save_conjunction_map(conjunction_name, conjunction_map, group_mask, output_dir)
    return group_mask
       
# create merged C conjunction map
def merge_C_conjunction_maps(pncc_conjunctions, group_mask, output_dir):
    """
    Merges C conjunction maps together; i.e. the C conjunction in the prereg. 
    specifies that this conjunction is run separately for each stimulus 
    category. However, for display purposes it can help to merge related
    stimuli togehter and output the union of the maps.
    
    """
    print('Creating merged C conjunction map')
    # load example data
    data = load_mri(group_mask,group_mask)
    # find relevant C conjunctions
    conjunctions_of_interest = []
    for conjunction_name in pncc_conjunctions:
        if conjunction_name[0] == 'C':
            conjunctions_of_interest.append(conjunction_name)
    # make empty data array
    data = np.zeros((len(data),len(conjunctions_of_interest)))
    
    ## Non-directional conjunctions
    # load data
    for cidx in range(len(conjunctions_of_interest)):
        fname = output_dir + os.sep + conjunctions_of_interest[cidx] + '_conjunction.nii.gz'
        data[:,cidx] = np.squeeze(load_mri(fname, group_mask))
    
    # merge data
    merged_c_map = np.min(data,1)
    fname = output_dir + os.sep + 'C_combined_and_uncorrected_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.max(data,1)
    fname = output_dir + os.sep + 'C_combined_or_uncorrected_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.sum(data,1)
    fname = output_dir + os.sep + 'C_combined_sum_uncorrected_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # Directional conjunctions
    for direction in ['Activation', 'Deactivation']:
        # load data
        for cidx in range(len(conjunctions_of_interest)):
            fname = output_dir + os.sep + conjunctions_of_interest[cidx] + '_' + direction + '_conjunction.nii.gz'
            data[:,cidx] = np.squeeze(load_mri(fname, group_mask))
        
        # merge data
        merged_c_map = np.min(data,1)
        fname = output_dir + os.sep + 'C_combined_and_uncorrected_' + direction + '_conjunction.nii.gz'
        save_mri(merged_c_map, group_mask, fname)
        
        # merge data
        merged_c_map = np.max(data,1)
        fname = output_dir + os.sep + 'C_combined_or_uncorrected_' + direction + '_conjunction.nii.gz'
        save_mri(merged_c_map, group_mask, fname)
        
        # merge data
        merged_c_map = np.sum(data,1)
        fname = output_dir + os.sep + 'C_combined_sum_uncorrected_' + direction + '_conjunction.nii.gz'
        save_mri(merged_c_map, group_mask, fname)
        

def make_B_conjunction(conjunction_name, conjunction_info, group_mask, output_dir):
    """
    Make a B style conjunction map; i.e. 
    [Tar > bsl] & [Rel != bsl] & [Irrel = bsl]
    conjunction_map = (con1) && (con2A || con2B) && (con3_equivalence)
    Save map as binary & associated z stat map as nifti files.
    conjunction_name: name of conjunction
    conjunction_info: contains information about copes of interest and test direction to identiy correct gfeat data to be loaded for conjunction
    output_dir: output dir where nifti files are written
    group_mask: group mask path
    Returns:
    group_mask: utilized group mask path
    """
    print('. Creating B conjunction map for: ' + conjunction_name)
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con1']['cope_of_interest']
    test_direction = conjunction_info['con1']['test_direction']
    data1, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con3']['cope_of_interest']
    test_direction = conjunction_info['con3']['test_direction']
    data3, group_mask = load_group_bayesian(cope_of_interest, test_direction, group_mask)
    # make binary conjunction map
    
    conjunction_map = np.logical_and.reduce([data1>0, np.logical_or(data2A>0,data2B>0), data3>0])
    # save maps
    save_conjunction_map(conjunction_name, conjunction_map, group_mask, output_dir)
    return group_mask

def make_A_conjunction(conjunction_name, conjunction_info, group_mask, output_dir):
    """
    Make an A style conjunction map; i.e. 
    [Tar > bsl] & [Rel = bsl] & [Irrel = bsl] 
    conjunction_map = (con1) && (con2_equivalence) && (con3_equivalence)
    Save map as binary & associated z stat map as nifti files.
    conjunction_name: name of conjunction
    conjunction_info: contains information about copes of interest and test direction to identiy correct gfeat data to be loaded for conjunction
    output_dir: output dir where nifti files are written
    group_mask: group mask path
    Returns:
    group_mask: utilized group mask path
    """
    print('. Creating A conjunction map for: ' + conjunction_name)
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con1']['cope_of_interest']
    test_direction = conjunction_info['con1']['test_direction']
    data1, group_mask = load_group_zmap(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2']['cope_of_interest']
    test_direction = conjunction_info['con2']['test_direction']
    data2, group_mask = load_group_bayesian(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 1A, activation
    cope_of_interest = conjunction_info['con3']['cope_of_interest']
    test_direction = conjunction_info['con3']['test_direction']
    data3, group_mask = load_group_bayesian(cope_of_interest, test_direction, group_mask)
    # make conjunction map
    
    conjunction_map = np.logical_and.reduce([data1>0, data2>0, data3>0])
    # save maps
    save_conjunction_map(conjunction_name, conjunction_map, group_mask, output_dir)
    return group_mask


# %% run
if __name__ == '__main__':
    # make output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create group maps for all bayesian tests
    # get subjects
    subjects = get_subject_list(bids_dir,subject_list_type)
    
    # loop over pncc conjunctions and create conjunction maps
    for conjunction_name in pncc_conjunctions:
        print(conjunction_name)
        conjunction_info = pncc_conjunctions[conjunction_name]
        # make different conjunction styles depending on conjunction name; i.e. A, B or C style conjunctions (letters refer to preregistration conjunction labels)
        if conjunction_name[0] == 'C':
            group_mask = make_C_conjunction(conjunction_name, conjunction_info, group_mask, output_dir)
            group_mask = make_C_directional_conjunction(conjunction_name, conjunction_info, group_mask, output_dir)
        elif conjunction_name[0] == 'B':
            make_B_conjunction(conjunction_name, conjunction_info, group_mask, output_dir)
        elif conjunction_name[0] == 'A':
            make_A_conjunction(conjunction_name, conjunction_info, group_mask, output_dir)
    
    # merge all C style conjunctions to one map
    merge_C_conjunction_maps(pncc_conjunctions, group_mask, output_dir)
