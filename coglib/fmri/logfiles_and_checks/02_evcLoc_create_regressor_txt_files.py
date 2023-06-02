#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads MRI specific, bids compliant *_event.tsv file per run, then writes
various event files .txt for 1st level GLMs (FSL FEAT compliant format; 
i.e. 1 file per regressor) for EVC localizer data.
Requires events.tsv file to exist; i.e. first run associated 
'*_create_events_tsv_file.py' script.

Inputs:
    - bids compliant *_event.tsv file per run with all information relevant for 
    analysis of EVC localizer data

Outputs: 
    - 3 column event txt files for use in fMRI analysis per regressor 
    (timestamp, ev duation, parametric modulator)

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 12.10.2020

@author: David Richter (d.richter@donders.ru.nl)
@tag: prereg_v4.2
"""

# %% Imports & parameters
import pandas as pd 
import numpy as np
import sys
pd.options.mode.chained_assignment = None  # default='warn'


##### Parameters #####
# session label
sessionLabel = 'V1'

# stimulus & response parameters
relevantEvents = ['TRBL','TLBR','buttonPress']      # relevant event types of interest (creates regressor per type); i.e. stimulus onsets (TRBL = top right + bottom left; TLBR = top left + bottom right; buttonPress = participant's button press)

# expected durations & trials (only used for checks)
stimulusDuration = 15.25                            # expected duration of stimulus events in seconds (actual duration taken from log file)

# TR & number of dummy volumes (required to adjust timestamps of events.tsv file to INCLUDE non-removed dummy scans, as per bids specification)
TR = 1.5
nDummyVols = 3

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase2_V1'


##### Paths #####

# bid dir & code dir
bidsDir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'  
codeDir = bidsDir + '/code'  

# output paths
outputLogFilePattern = bidsDir + '/sub-%(sub)s/ses-%(ses)s/func/sub-%(sub)s_ses-%(ses)s_%(runType)s_events'
outputEventFilePattern = bidsDir + '/derivatives/regressoreventfiles/sub-%(sub)s/ses-%(ses)s/sub-%(sub)s_ses-%(ses)s_%(runType)s_%(eventType)s_EV.txt'


# import functions from helper_functions_MRI
sys.path.append(codeDir)
from helper_functions_MRI import saveEventFile, get_subject_list


# %% create 3 column event files for GLM regressors
# event file for stimulus events of interest
def createRegressorEvFiles_EvcLocalizer(dataLog, outputEventFilePattern, relevantEvents, runType, sub):
    """
    Create 3 column event files (onset, duration, parametric mod.) for stimulus 
    events of interest for EVC localizer.
    dataLog: MRI log file with events as returned by createMriLog_exp1
    outputEventFilePattern: file name pattern for event file
    relevantEvents: list of events of interest 
    runType: run type label
    sub: subject ID
    Returns: nothing, but writes event file .txt to output path per relevant
    event type
    """
    # get timestamps and durations of all stimulus events (i.e., excluding button presses and null events)
    allStimEvents_noPresses = (dataLog['trial_type']=='TLBR') | (dataLog['trial_type']=='TRBL') | (dataLog['trial_type']=='Null')
    allEventTS = dataLog['onset'][allStimEvents_noPresses].to_numpy()
    allDurations = np.append(allEventTS[1::] - allEventTS[0:-1], stimulusDuration)
    allEvents = dataLog['trial_type'][allStimEvents_noPresses].to_numpy()
    # loop over events of interest    
    for stimType in relevantEvents:
        # if event type is not a button press (i.e. a stimulus event), create regressor with duration = actual stimulus duration
        if not stimType == 'buttonPress':
             # get onsets and durations from those calculated above
            onsets = allEventTS[allEvents==stimType]
            durations = allDurations[allEvents==stimType]
        # if event type is button presses, create stick function regressor (0 duration)
        else:
            # get onsets and set duration
            onsets = dataLog['onset'][dataLog['trial_type']=='buttonPress']
            durations = np.zeros((len(onsets)))
        # remove dummy volume duration
        onsets = onsets - (nDummyVols*TR)
        # set parametric mod to 1s
        parametricMod = np.ones((len(onsets)))
        # make 3 column format
        event = np.vstack((onsets,durations,parametricMod)).T
        # save event file
        fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType, 'ses':sessionLabel}        
        saveEventFile(fname,event)
    
    
# %%
if __name__ == '__main__':
    """
    Create regressors event txt files for each subject from events.tsv file.
    Writing FSL FEAT compliant 3 column regressor event files for each relevant 
    event type of the EVC localizer.
    bidsDir: bids directory
    """
    # get subject list
    subjects = get_subject_list(bidsDir,subject_list_type)
    
    # loop over subjects    
    for sub in subjects:     
        sub = sub[4::]
        print('========== SUBJECT: ' + sub + ' ==========')
        
        # label for run type; note we assume that only the correct run has been transformed to BIDS and only multiple log files are stored
        runType = 'task-EVCLoc_run-1'
        
        fname = outputLogFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel} + '.tsv'
        try:
             dataLog = pd.read_csv(fname, sep='\t')
        except Exception as e:
            print(e)
            print(' - CHECK EVENTS.TSV FILE ! -')
            print(' - SKIPPING RUN ! - ')
            continue
        
        ##### Create event files for GLM per run#####
        print('... Creating regressor event files')
        createRegressorEvFiles_EvcLocalizer(dataLog, outputEventFilePattern, relevantEvents, runType, sub)
        
    