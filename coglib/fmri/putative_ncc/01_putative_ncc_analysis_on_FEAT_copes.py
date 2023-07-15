#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs the putative NCC (pNCC) analysis.
Uses output of several FSL FEAT analyses to perform conjunctions. 
These 1st (run), 2nd (subject) and 3rd (group) level feat analyses must be 
submitted using the script glm/02_run_fsf_feat_analyses.py and must be 
completed by the time this script is run! The cope paths, cope numbers and 
labels must be defined below to correspond to exactly the same cope numbers
and labels used during the FSL FEAT setup (i.e. as defined in the fsf files).

In addition to using the existing FSL FEAT outputs this script performs 
bayes factor tests to establish equivalence to baseline for relevant contrasts.
Settings for the bayes factor test can be adjusted below.

Finally, it runs the actual conjunctions for the pNCC analysis as defined below.
The cope labels in pncc_conjunctions must correspond to those used in
the FSL FEAT cope setup.
The conjunction maps, intermediate equivalence maps and additional conjunction 

Created on Fri Jun 18 17:18:01 2021

@author: David Richter (david.richter.work@gmail.com), Yamil Vidal (hvidaldossantos@gmail.com)

"""

# %% Imports & parameters
import numpy as np
from scipy import stats
import os, sys
import pingouin as pn

##### Paths #####

# project root path; assumed to contains raw folder and bids folder (following bids specification)
projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

# paths
bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 
data_dir = projectRoot + '/bids/derivatives/fslFeat'


# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase2_V1'
#subject_list_type = 'phase3_V1'
#subject_list_type = 'debug'

output_dir = projectRoot + '/bids/derivatives/putative_ncc/' + subject_list_type

# import functions from helper_functions_MRI
sys.path.append(code_dir)
from helper_functions_MRI import load_mri, save_mri, get_subject_list

# %% Parameters and analysis definitions
# define cope paths and cope templates; i.e. how the feat copes map to contrast labels and where to find them (number of feat dir)

# path template to group level cope results (i.e. from group level feat analyses)
cope_group_results_template = data_dir + '/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM%(path_ext)s_space-MNI152NLin2009cAsym_desc-%(cope)s.gfeat/%(subCope)s.feat/'

# path template to individual subjects cope data (2nd level gfeat copes)
cope_path_template = data_dir + '/%(sub)s/ses-V1/%(sub)s_ses-V1_task-Dur_analysis-2ndGLM%(label)s_space-MNI152NLin2009cAsym.gfeat/%(cope)s.feat/stats/cope1.nii.gz'

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


# Bayesian test parameters/settings
# labels of copes for which an bayesian test should be performed (i.e. those that include an == statement in the conjunction analysis)
copes_of_interest_for_bayesian_test = ['RelAll','IrrelAll']

# stat threshold for bayes factor map
bf_threshold = 3

# whether to apply additional spatial smoothing (in mm) to bayes factor maps; if <=0 no smoothing is applied
smoothing_in_mm = 0


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
        'C_Face': {
        'con1A': {'cope_of_interest':'RelFace', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelFace', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelFace', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelFace', 'test_direction':'Deactivation'}
        },
        'C_Object': {
        'con1A': {'cope_of_interest':'RelObject', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelObject', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelObject', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelObject', 'test_direction':'Deactivation'}
        },
        'C_Letter': {
        'con1A': {'cope_of_interest':'RelLetter', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelLetter', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelLetter', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelLetter', 'test_direction':'Deactivation'}
        },
        'C_FalseFont': {
        'con1A': {'cope_of_interest':'RelFalseFont', 'test_direction':'Activation'},
        'con1B': {'cope_of_interest':'RelFalseFont', 'test_direction':'Deactivation'},
        'con2A': {'cope_of_interest':'IrrelFalseFont', 'test_direction':'Activation'},
        'con2B': {'cope_of_interest':'IrrelFalseFont', 'test_direction':'Deactivation'}
        },
        'A': {
        'con1': {'cope_of_interest':'TarAll', 'test_direction':'Activation'},
        'con2': {'cope_of_interest':'RelAll', 'test_direction':'Equivalence'},
        'con3': {'cope_of_interest':'IrrelAll', 'test_direction':'Equivalence'}
        },
        'B': {
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
    conjunction_map: binary conjunction map
    group_mask: group mask path
    output_dir: output dir where nifti files are written
    """

    save_fname = output_dir + os.sep + conjunction_name + '_conjunction.nii.gz'
    save_mri(conjunction_map, group_mask, save_fname)
        
def load_group_cope(cope_of_interest, test_direction, group_mask_fname='mask.nii.gz', data_fname='thresh_zstat1.nii.gz',  return_bool=False):
    """
    Gets thresholded maps for group copes; i.e. significance maps for C 
    conjunctions.
    cope_of_interest: cope name of interest to be loaded (e.g. RelFace to load relevant faces > baseline contrast)
    test direction: direction of statistical test to load; corresponds to subCopes (e.g. Activation to load RelFace>Baseline)
    return_bool: if true returns a boolean map instead of z values (default false)
    data_fname: (optional) file name for contast of interest; appended to path. If left empty thresholded z stat from gfeat dirs will be used
    group_mask_fname: (optional) full path to group mask used to mask the MRI data. If left empty the gfeat mask will be used corresponding to the loaded contrast 
    Returns:
    data: mri map as vector
    group_mask: utilized group mask path
    """
    # get information from copes & subcopes
    cope = copes[cope_of_interest]['cope']
    path_ext = copes[cope_of_interest]['path_ext']
    subCope = subCopes[test_direction]['subCope']
    print('. . loading ' + cope + '.gfeat, ' + subCope + '.feat | ' + test_direction + ' to relative baseline' )
    # make data path
    data_path = cope_group_results_template%{'path_ext':path_ext,'cope':cope,'subCope':subCope}
    data_final_path = data_path + data_fname
    if group_mask_fname == 'mask.nii.gz':
        group_mask = data_path + group_mask_fname
    else:
        group_mask = group_mask_fname
    # load mri data
    data = load_mri(data_final_path, group_mask)
    if return_bool:
        data = np.asarray(data,dtype='bool')
    
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
    data1A, group_mask = load_group_cope(cope_of_interest, test_direction_1A, group_mask)
    # get map for conjunction part 1B, deactivation
    cope_of_interest = conjunction_info['con1B']['cope_of_interest']
    test_direction_1B = conjunction_info['con1B']['test_direction']
    data1B, group_mask = load_group_cope(cope_of_interest, test_direction_1B, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction_2A = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_cope(cope_of_interest, test_direction_2A, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction_2B = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_cope(cope_of_interest, test_direction_2B, group_mask)
    # make directional C conjunction Activation
    if test_direction_1A == test_direction_2A:
        conjunction_name_A = conjunction_name + '_' + test_direction_1A
        print('. Creating C directional conjunction map for: ' + conjunction_name_A)

        conjunction_map = np.logical_and(data1A>0, data2A>0)
        # save maps
        save_conjunction_map(conjunction_name_A, conjunction_map, group_mask, output_dir)
    else:
        print('. ! Caution: conjunction test direction do not match. No map created for C Directional conjunction C.A')
    # make directional C conjunction Deactivation
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
    data1A, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 1B, deactivation
    cope_of_interest = conjunction_info['con1B']['cope_of_interest']
    test_direction = conjunction_info['con1B']['test_direction']
    data1B, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
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
    fname = output_dir + os.sep + 'C_combined_and_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.max(data,1)
    fname = output_dir + os.sep + 'C_combined_or_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # merge data
    merged_c_map = np.sum(data,1)
    fname = output_dir + os.sep + 'C_combined_sum_conjunction.nii.gz'
    save_mri(merged_c_map, group_mask, fname)
    
    # Directional conjunctions
    for direction in ['Activation', 'Deactivation']:
        # load data
        for cidx in range(len(conjunctions_of_interest)):
            fname = output_dir + os.sep + conjunctions_of_interest[cidx] + '_' + direction + '_conjunction.nii.gz'
            data[:,cidx] = np.squeeze(load_mri(fname, group_mask))
        
        # merge data
        merged_c_map = np.min(data,1)
        fname = output_dir + os.sep + 'C_combined_and_' + direction + '_conjunction.nii.gz'
        save_mri(merged_c_map, group_mask, fname)
        
        # merge data
        merged_c_map = np.max(data,1)
        fname = output_dir + os.sep + 'C_combined_or_' + direction + '_conjunction.nii.gz'
        save_mri(merged_c_map, group_mask, fname)
        
        # merge data
        merged_c_map = np.sum(data,1)
        fname = output_dir + os.sep + 'C_combined_sum_' + direction + '_conjunction.nii.gz'
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
    data1, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2A, activation
    cope_of_interest = conjunction_info['con2A']['cope_of_interest']
    test_direction = conjunction_info['con2A']['test_direction']
    data2A, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
    # get map for conjunction part 2B, deactivation
    cope_of_interest = conjunction_info['con2B']['cope_of_interest']
    test_direction = conjunction_info['con2B']['test_direction']
    data2B, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
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
    data1, group_mask = load_group_cope(cope_of_interest, test_direction, group_mask)
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


# %% functions for bayesian tests

def get_cope_data(subjects, group_mask, cope_info, cope_path_template):
    """
    Gets 2nd level FEAT cope data from feat dir. Sets all data outside of mask 
    to NaN.
    subjects: list of subjects for analysis
    group_mask: group level mask (e.g. MNI) for masking maps.
    cope_info: cope number (cope) and path extensions (path_ext) information
    cope_path_template: template string for cope path with place holders for
    subjects, labels, cope numbers.    
    Returns:
    data: cope maps
    """
    print('. loading 2nd level data for all subjects: ' + cope_info['cope'])
    # template data shape from group mask
    data_template = load_mri(group_mask, group_mask)
    # empty array for data
    data = np.empty((len(subjects), len(data_template)))
    data[:] = np.NaN
    # loop over subjects
    for sub_idx in range(len(subjects)):
        subject = subjects[sub_idx]
        # get all paths of interest
        current_cope_path = cope_path_template%{'sub':subject, 'label':cope_info['path_ext'], 'cope':cope_info['cope']}
        # load data
        data[sub_idx,:] = load_mri(current_cope_path, group_mask)
    # set all zeros to NaN (to drop voxels containing missing data from analysis)
    data[data==0] = np.NaN
    # count and display n NaN
    nanCount = np.zeros((data.shape[1]))
    for idx in range(data.shape[1]):
        nanCount[idx] = any(np.isnan(data[:,idx]))
    print('. . ' + str(int(sum(nanCount))) + ' of ' + str(data.shape[1]) + ' voxels (per cope map) contain at least 1 subject with NaN')
    return data

def make_bayesian_map(cope_for_bayesian_test, data, group_mask, output_dir, smoothing_in_mm=0):
    """
    Run equivalence tests on MRI data (copes) and write t + p maps as output.
    for_equivalence_test: label of the cope of interest
    data: cope data [subjects by voxels]
    group_mask: group level mask (e.g. MNI) for masking maps.
    output_dir: output dir for maps
    smoothing_in_mm: spatial smoothing in mm (if 0, no smoothing is applied)
    """
    # make empty data arrays
    bf01 = np.empty((data.shape[1]))
    bf01[:] = np.NaN
    
    # loop over voxels and run bayesian tests
    print('. . running bayes factor (threshold: ' + str(bf_threshold) + ') for: ' + cope_for_bayesian_test + ' | total n voxels: ' + str(data.shape[1]))
    for idx in range(data.shape[1]):
        if idx%10000 == 0:
            print('. . . processing voxel: ' + str(idx) + ' of ' + str(data.shape[1]))
        
        # skip if there are any nans
        if np.isnan(data[:,idx]).any():
            bf01[idx] = np.NaN
            
        else:
            bf01[idx] = one_sample_bayes_factor(data[:,idx])
            
    # save resulting maps
    fname_bf01 = save_bayesian_map(cope_for_bayesian_test, bf01, group_mask, output_dir, bf_threshold)
    # apply additional smoothing to bayesian maps
    if smoothing_in_mm > 0:
        print('Smoothing bayesian map (' + str(smoothing_in_mm) + 'mm)')
        smoothed_bf01_map = smooth_maps(fname_bf01, smoothing_in_mm, group_mask)
        save_bayesian_map(cope_for_bayesian_test, smoothed_bf01_map, group_mask, output_dir, bf_threshold)
    
    return bf01
        
def one_sample_bayes_factor(x):
    T = stats.ttest_1samp(x, 0)
    bf = pn.bayesfactor_ttest(T[0], x.shape[0])
    
    return 1/bf

def smooth_maps(map_fname, smoothing_in_mm, group_mask):
    """
    Smooth bayesian maps given smoothing size in mm.
    map_fanme: path to map to be smoothed
    smoothing_in_mm: smoothing kernel size in mm
    group_mask: path to group mask
    Returns
    masked_data: smoothed & masked data
    """
    from nilearn import image
    import nibabel as nib
    print('. smoothing map: ' + map_fname)
    # apply smoothing
    map_to_smooth = nib.load(map_fname)
    smoothed_img = image.smooth_img(map_to_smooth, smoothing_in_mm)
    # mask 
    m = nib.load(group_mask).get_fdata()
    masked_data = np.asarray(smoothed_img.get_fdata()[m != 0], dtype=np.float32)
    return masked_data

def save_bayesian_map(cope_for_bayesian_test, bf01, group_mask, output_dir, bf_threshold=3):
    """
    Save bayesian maps as niftis
    for_equivalence_test: label of the cope of interest
    equivalence_t: t statistics of equivalence test map
    equivalence_p: p values of equivalence test map
    group_mask: group level mask (e.g. MNI) for masking map.
    output_dir: output dir for maps
    equivalence_threshold: z statistic threshold for evidence maps (all below will be set to 0)
    Returns
    fname_t: filename of t map
    fname_p: filename of p map
    """
    # set nans to 0
    # bf01[np.isnan(bf01)] = 0
    
    # make output path
    fpath = output_dir + os.sep + 'bayesian_maps' 
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    # save bf
    fname = fpath + os.sep + cope_for_bayesian_test + '_bayesian_map.nii.gz'
    save_mri(bf01, group_mask, fname)
    
    # threshold evidence
    bf01[bf01<bf_threshold] = 0
    fname = fpath + os.sep + cope_for_bayesian_test + '_bayesian_map_thresh.nii.gz'
    save_mri(bf01, group_mask, fname)
    
    return fname
    

# %% run
if __name__ == '__main__':
    # make output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create group maps for all bayesian tests
    # get subjects
    subjects = get_subject_list(bids_dir,subject_list_type)
    
    remove_subjects = ['sub-SD122','sub-SD196']
    for r in remove_subjects:
        subjects = subjects[subjects != r]
    
    print('Removed subjects:',remove_subjects)
    print('Total subjects:',len(subjects))
    
    # # loop over copes of interest for which an bayesian test should be performed
    for cope_for_bayesian_test in copes_of_interest_for_bayesian_test:
        cope_info = copes[cope_for_bayesian_test]
        # get MRI data from all copes for which an bayesian test should be run
        data = get_cope_data(subjects, group_mask, cope_info, cope_path_template)
        # run bayesian test and create map
        make_bayesian_map(cope_for_bayesian_test, data, group_mask, output_dir, smoothing_in_mm)
    
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
