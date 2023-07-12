#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code is used for QC motion threshold.
It checks MRI data quality using IQMs from MRIQC. 

Takes MRIQC outputs, assuming bids specifications (bids_dir).
Calculates mean FD, FD_perc, DVARS and tsnr per run. Uses FD_perc and DVARS 
to reject bad participants exceeding X SD above (FD_perc, DVARS) the group 
mean (X defined in SdThreshold).

Utilized metrics are specified in more detail here: 
    https://mriqc.readthedocs.io/en/latest/measures.html
FD = Framewise Displacement: expresses instantaneous head-motion. 
    MRIQC reports the average FD, labeled as fd_mean. 
    Rotational displacements are calculated as the displacement on the surface 
    of a sphere of radius 50 mm [Power2012]:
FD_perc = the percent of FDs above the FD threshold w.r.t. the full timeseries 
    FD threshold is set at 0.20mm.
DVARS = D: temporal derivative of timecourses. VARS: RMS variance over voxels 
    ([Power2012] dvars_nstd). Indexes the rate of change of BOLD signal across 
    the entire brain at each frame of data.
TSNR = Temporal SNR is a simplified interpretation of the tSNR definition 
    [Kruger2001]. Reports the median value of the tSNR map calculated by:
    average BOLD signal (across time) divided by the corresponding temporal 
    standard-deviation map.

Writes 3 output csv files:
2 csv files with subjects rejected per session, averaged over runs 
(IQM-summary-ses).
1 csv file with rows per run, allowing for run specific rejection (IQM-allRuns)
Each csv file contains subject IDs, session labels if applicable and 4 IQMs
(FD, FD_perc, DVARS and tsnr), and a rejection flag, indicating if the 
run/subject should be rejected from analysis. Outputs are written to the MRIQC 
directory (mriQcSubdir).

@author: David Richter, first created 01/27/2021
@ Modified by Urszula Gorska (gorska@wisc.edu)
Last modified 06/15/2023
"""

import os, sys, json
import numpy as np
import pandas as pd


#%% Paths and Parameters

# root project path
root_dir = '/project_root_path'

# threshold to mark subjects as rejected if they exceed X SD above group mean
SdThreshold = 2

###############################################################################

# bids path
bids_dir = root_dir + '/bids'

# mri qc sub dir
mriQcSubdir = '/derivatives/mriqc'
# sub dir pattern with sub and ses key-value pairs
dataDirPattern = bids_dir + mriQcSubdir + os.sep + '%(sub)s' + os.sep + '%(ses)s' + os.sep + 'func' + os.sep 

# session labels
sesLabels = ['ses-V1']

participants_dir = bids_dir
subject_list_type = 'demo'

# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/coglib/fmri'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import get_subject_list


# %% support functions
def getJsonFnames(fPath):
    """
    Get MRIQC output json files for current subject and session 
    fPath: file path + name
    Returns: jsonFiles per run
    """
    jsonFiles = []
    for root, dirs, files in os.walk(fPath):
        for file in files:
            if file.endswith(".json"):
                jsonFiles.append(os.path.join(root, file))
    return jsonFiles

def saveIQMs(df, ses=None):
    """
    Save summary IQM as csv file, either per session with data averaged across 
    runs if ses arg is passed or for all runs separately if ses arg is not 
    passed
    df: data frame with IQMs to be written as csv file
    ses: session label 
    """
    outputDir = bids_dir + mriQcSubdir
    # outputDir = bids_dir + mriQcSubdir
    if ses is None:
        fname = outputDir + os.sep + 'IQM-perRun_' + subject_list_type + '.csv'
    else:
        fname = outputDir + os.sep + 'IQM-summary_' + ses + '_' + subject_list_type + '.csv'
    df.to_csv(fname, sep=',', index=False)  


# %% data processing functions
def getDataFromJson(jsonFiles):
    """
    Get relevant IQM data from json files (output of MRIQC) per run
    jsonFiles: json file paths + names
    Returns: df_sub pandas data frame with relevant IQMs per run
    """
    sub_id = []
    ses_label = []
    run_label = []
    fd_mean = []
    fd_perc = []
    dvars_nstd = []
    tsnr = []
    for fname in jsonFiles:
        sub_id.append(fname[fname.find('sub-')+4:fname.find('sub-')+9])
        ses_label.append(fname[fname.find('ses-')+4:fname.find('ses-')+6])
        run_label.append(fname[fname.find('task-')+5:fname.find('run-')+5])
        with open(fname) as json_file:
            data = json.load(json_file)
            fd_mean.append(data['fd_mean'])
            fd_perc.append(data['fd_perc'])
            dvars_nstd.append(data['dvars_nstd'])
            tsnr.append(data['tsnr'])
    df_sub = pd.DataFrame({"sub_id":sub_id, "session":ses_label, "run":run_label, "fd_mean":fd_mean, "fd_perc":fd_perc, "dvars_nstd":dvars_nstd, "tsnr":tsnr, "rejected":[False]*len(sub_id)})
    return df_sub
    
def processQMs(df):
    """
    Process IQMs by adding the mean and adding a rejection flag per subject
    based on the subject's IQM being x SD (defined in SdThreshold) worse than 
    the group mean. 
    IQMs used for rejection are: 
    fd_perc = the percent of FDs above the FD threshold w.r.t. the full 
        timeseries FD threshold is set at 0.20mm (MRI QC default).
    dvars_nstd = D: temporal derivative of timecourses. VARS: RMS variance over 
        voxels ([Power2012] dvars_nstd). Indexes the rate of change of BOLD 
        signal across the entire brain at each frame of data.
    tsnr = Temporal SNR is a simplified interpretation of the tSNR definition 
        [Kruger2001]. Reports the median value of the tSNR map calculated by:
        average BOLD signal (across time) divided by the corresponding 
        temporal standard-deviation map.
    df: data frame to be processed containing the IQMs
    Returns df with added mean and rejection flags (rejected == 1 indicating 
                                                    rejected participants/runs)
    """
    # add mean
    sd = df.std().copy()
    df.loc['mean'] = df.mean().copy()
    df.loc['mean', ['subIDs']]='average'
    # add rejection flag for 'bad' subjects if any IQM does not pass check
    # fd percentage
    rejThresh_fd_perc = df.fd_perc['mean'] + SdThreshold * sd['fd_perc']
    df.loc[df.fd_perc > rejThresh_fd_perc,['rejected']] = 1
    # dvars
    rejThresh_dvars = df.dvars_nstd['mean'] + SdThreshold * sd['dvars_nstd']
    df.loc[df.dvars_nstd > rejThresh_dvars,['rejected']] = 1
    ## tsnr (add lines below back in if tsnr is also to be used to reject data)
    #rejThresh_tsnr = df.tsnr['mean'] - SdThreshold * sd['tsnr']
    #df.loc[df.tsnr < rejThresh_tsnr,['rejected']] = 1
    return df

#%%
if __name__ == '__main__':
    """
    Gather IQMs (image quality metrics) from MRIQC output per subject 
    (averaging over runs) and session.
    Write output csv file containing IQMs of interest
    subjects: list of subjects to be processed (MRI QC must be finished for all 
    sessions for these subjects)
    """
    # subjects = get_subject_list(bids_dir,subject_list_type)
    subjects = get_subject_list(participants_dir,subject_list_type)
    df_all = pd.DataFrame()
    
    # loop over sessions
    for sesIdx in range(len(sesLabels)):
        ses = sesLabels[sesIdx]
        subIDs = []
        av_fd_mean = []
        av_fd_perc = []
        av_dvars_nstd =[]
        av_tsnr = []
        
        # loop over subjects
        for sub in subjects:
            print('Processing | subject: ' + sub + ' | session: ' + ses)
            fPath = dataDirPattern%{'sub':sub, 'ses':ses}
            # get json files
            jsonFiles = getJsonFnames(fPath)
            print(fPath)
            # check if json files exist; otherwise throw warning and skip sub
            if not jsonFiles:
                print('! CAUTION: No MRIQC json files found for subject: ' + sub + ' | session: ' + ses + ' ! Skipping ! Make sure to run MRIQC for all subjects first !')
                continue
            df_sub = getDataFromJson(jsonFiles)
            # append sub df to df with all runs and sessions
            df_all = df_all.append(df_sub, ignore_index=True)
            # calc mean over runs and add to list
            av_fd_mean.append(np.mean(df_sub['fd_mean']))
            av_fd_perc.append(np.mean(df_sub['fd_perc']))
            av_dvars_nstd.append(np.mean(df_sub['dvars_nstd']))
            av_tsnr.append(np.mean(df_sub['tsnr']))
            subIDs.append(sub)
            
        # gather data to df (QMs averaged over runs per subject)
        df = pd.DataFrame({"subIDs":subIDs, "fd_mean":av_fd_mean, "fd_perc":av_fd_perc, "dvars_nstd":av_dvars_nstd, "tsnr":av_tsnr, "rejected":[False]*len(av_tsnr)})
        df = processQMs(df)
        saveIQMs(df, ses)
        
    # process data frame containing all subjects & session IQMs per run
    df_all = processQMs(df_all)
    saveIQMs(df_all)
    