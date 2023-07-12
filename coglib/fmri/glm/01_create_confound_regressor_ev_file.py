#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates confound regressor file, as required by first level GLMs (e.g. using 
FSL FEAT). Extracts confounds of interest (defined in confounds_of_interest) 
from fmriprep confound tsv file. The function get_confound_regressor_list
contains several common variations of confound regressor sets. Select one by
adjusting the variable confound_regressor_set to one of the implemented options
or add a new variation to get_confound_regressor_list. After extracting 
confound regressor time series, script discards x dummy volumes  (defined in 
n_dummy_volumes) from beginning of confound regressor file. Finally, creates 
one confound regressor file per run for each subject and session in output dir
defined in event_file_output_path + event_file_output_pattern

Note: this script assumes that fmriprep already successfully completed for all 
subjects of interest!

Created on Wed Mar 14 12:39:58 2021

@author: David Richter
@tag: prereg_v4.2
"""

import pandas as pd
import numpy as np
import os, glob, sys


#%% Paths and Parameters
# root project path
root_dir = '/project/3018050.01/twcf_code_review'

#subject_list_type = 'phase3_V1'
subject_list_type = 'demo'


### Confound regressors of interest ###
# define which confound regressor set should be extracted. See 
# get_confound_regressor_list infro for details.
# options: 6motion, 24motion, 6motion_FD_CSF_WM, 24motion_CSF_WM, 6motion_FD_5aCompCor
confound_regressor_set = '24motion_CSF_WM'

# BIDS path
bids_dir = root_dir + '/bids'

# input file (from fmriprep)
fmriprep_confound_file_pattern = bids_dir + '/derivatives/fmriprep/%(sub)s/%(ses)s/func/%(sub)s_%(ses)s_%(task)s_desc-confounds_timeseries.tsv'

# output file path
event_file_output_path = bids_dir + '/derivatives/regressoreventfiles/%(sub)s/%(ses)s/confound_event_files'
# output file name
event_file_output_pattern = '%(sub)s_%(ses)s_%(task)s_confounds.txt'


# session list
session_labels = ['ses-V1']

# dummy volumes (removed from beginning of confound tsv file)
n_dummy_volumes = 3


# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/code'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import get_subject_list


# %% functions
def get_confound_regressor_list(confound_regressor_set):
    """
    Get list of confound regressor header labels (corresponding to the header
    names output by fmriprep in *_desc-confounds_timeseries.tsv).
    :param
    confound_regressor_set: name of the regressor set to be returned
        Options include:
        6motion = Standard 6 motion parameters
        24motion = FSL's extended set of 24 motion parameters
        6motion_FD_CSF_WM = Standard 6 motion parameters plus framsewise 
            displacement, CSF and WM
        24motion_CSF_WM = FSL's extended set of 24 motion parameters plus  
            CSF and WM
        6motion_FD_5aCompCor = Standard 6 motion parameters, framewise 
            displacement, top 5 aCompCor. This is largely based on a suggestion 
            by Chris Gorgolewski (https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326)
    :returns
    confounds_of_interest: list of confound regressor labels of interest 
    
    """
    if confound_regressor_set == '6motion':
        # Standard 6 motion params only
        confounds_of_interest = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'] 
        
    # FSL's extended set of 24 motion params
    elif confound_regressor_set == '24motion':
        confounds_of_interest = ['trans_x', 'trans_x_derivative1', 'trans_x_power2', 'trans_x_derivative1_power2',
                                 'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
                                 'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
                                 'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
                                 'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
                                 'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2']
    
    # 6 motion params, framewise displacement, CSF and WM
    elif confound_regressor_set == '6motion_FD_CSF_WM':
        confounds_of_interest = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                 'framewise_displacement',
                                 'csf', 'white_matter']
        
    # FSL's extended set of 24 motion params
    elif confound_regressor_set == '24motion_CSF_WM':
        confounds_of_interest = ['trans_x', 'trans_x_derivative1', 'trans_x_power2', 'trans_x_derivative1_power2',
                                 'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
                                 'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
                                 'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
                                 'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
                                 'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2',
                                 'csf', 'white_matter']
    
    # 6 motion params, framewise displacement, 5 aCompCor (Suggestion by Chris Gorgolewski (https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326))        
    elif confound_regressor_set == '6motion_FD_5aCompCor':
        # make sure to use cosines_XX in model too, if you use this option!! see: https://fmriprep.org/en/stable/outputs.html -> "fMRIPrep does high-pass filtering before running anatomical or temporal CompCor. Therefore, when using CompCor regressors, the corresponding cosine_XX regressors should also be included in the design matrix."
        confounds_of_interest = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                 'framewise_displacement',
                                 'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04'] 
    
    return confounds_of_interest


#%% main functions to extract confound regressors

def create_confound_regressor_file(input_fname, output_fname, confounds_of_interest, n_dummy_volumes):
    """
    Creates confound regressor file, as required by e.g. FSL FEAT, extracting 
    confounds of interest from fmriprep confound tsv file.
    :param 
    input_fname: input file name (confound file from fmriprep)
    output_fname: output file name
    onfounds_of_interest: list of confound regressor labels of interest 
    n_dummy_volumes: number of dummy volumes to be discared
    :returns: nothing, saves output_fname
    """
    # load fmriprep confound list
    conf = pd.read_csv(input_fname, sep='\t')
    
    # get regressors of intrest and remove dummy vols
    confound_regressor = np.asarray((conf[confounds_of_interest].iloc[n_dummy_volumes::]),dtype='float')
    
    # save confound regressors as txt file (e.g. for fsl feat)
    np.savetxt(output_fname, confound_regressor, delimiter="\t")


def run_conf_for_all_subjects(subjects, session_labels, confounds_of_interest):
    """
    Run confound regressor creation for all subjects and sessions. 
    Finds fmriprep confound tsv file per run, given subject and session, then 
    runs confound regressor creation per run.
    :param 
    subjects: list of subject ID
    session_labels: list of session labels
    confounds_of_interest: list of confound regressor labels of interest
    :returns: nothing - writes confound regressors to output dir
    """
    # loop over subjects
    for sub in subjects:
        print('--------------------------------------------------------------')
        
        # loop over sessions
        for ses in session_labels:
            print('')
            
            # find fmriprep confound tsv files for current session
            input_fname = fmriprep_confound_file_pattern%{'sub':sub, 'ses':ses, 'task':'*'}
            runs = glob.glob(input_fname)
            if len(runs) == 0:
                print('!!! No confounds_timeseries.tsv found for: ' + sub  + ' | Session: ' + ses + ' !!!')
            
            # create output folder
            output_dir = event_file_output_path%{'sub':sub, 'ses':ses}
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            
            # loop over existing runs 
            for run in runs:
                task = run[run.find('task-'):run.find('_desc')]
                print('Processing: ' + sub  + ' | Session: ' + ses + ' | Run: ' + task)
                
                input_fname = fmriprep_confound_file_pattern%{'sub':sub, 'ses':ses, 'task':task}
                output_fname = event_file_output_path%{'sub':sub, 'ses':ses} + os.sep + event_file_output_pattern%{'sub':sub, 'ses':ses, 'task':task}
                
                create_confound_regressor_file(input_fname, output_fname, confounds_of_interest, n_dummy_volumes)


# %% run
if __name__ == '__main__':
    # get subject list
    subjects = get_subject_list(bids_dir,subject_list_type)
    
    # get confound regressor set
    confounds_of_interest = get_confound_regressor_list(confound_regressor_set)
    
    # run confound regressor creation for all subjects & sessions
    run_conf_for_all_subjects(subjects, session_labels, confounds_of_interest)
