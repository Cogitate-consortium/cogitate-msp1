#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads MRI specific, bids compliant *_event.tsv file per run, then writes
various event files .txt for 1st level GLMs (FSL FEAT compliant format; 
i.e. 1 file per regressor) for experiment 1.
Requires events.tsv file to exist; i.e. first run associated 
'*_create_events_tsv_file.py' script.

Inputs:
    - bids compliant *_event.tsv file per run with all information relevant for 
    analysis of exp.1 data

Outputs: 
    - 3 column event txt files for use in fMRI analysis per regressor 
    (timestamp, ev duation, parametric modulator)

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 23.10.2020

@author: David Richter (d.richter@donders.ru.nl)
@tag: prereg_v4.2
"""

# %% Imports & parameters
import pandas as pd 
import numpy as np
import sys
pd.options.mode.chained_assignment = None  # default='warn'

# root project path
root_dir = '/project/3018050.01/twcf_code_review'


##### Parameters #####

 # if true fills empty event files with a row of zeros (useful to avoid feat crashing if empty event files are supplied)
fill_empty = True

# session label
sessionLabel = 'V1'

# stimulus & response parameters
relevantEvents = ['baseline','face','object','letter','falseFont','response','targetScreen']             # relevant event types of interest (creates regressor per type)
relevantEventsWithTaskRelevance = ['target_face','target_object','target_letter','target_falseFont',
                  'relevant_face','relevant_object','relevant_letter','relevant_falseFont',
                  'irrelevant_face','irrelevant_object','irrelevant_letter','irrelevant_falseFont']             # relevant event types of interest (creates regressor per type)

# experiment design parameters
nRuns = 8                               # expected number of runs

# TR & number of dummy volumes (required to adjust timestamps of events.tsv file to INCLUDE non-removed dummy scans, as per bids specification)
TR = 1.5
nDummyVols = 3

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'demo'
#subject_list_type = 'debug'


##### Paths #####

# bid dir & code dir
bidsDir = root_dir + '/bids'  
codeDir = bidsDir + '/code' 

# inputs: events.tsv file
eventsTsvFilePattern = bidsDir + '/sub-%(sub)s/ses-%(ses)s/func/sub-%(sub)s_ses-%(ses)s_%(runType)s_events.tsv'

# output paths & file names; event regressor txt file
outputEventFilePattern = bidsDir + '/derivatives/regressoreventfiles/sub-%(sub)s/ses-%(ses)s/sub-%(sub)s_ses-%(ses)s_%(runType)s_%(eventType)s_EV.txt'


# import functions from helper_functions_MRI
sys.path.append(codeDir)
from helper_functions_MRI import saveEventFile, get_subject_list


# %% create 3 column event files
def createRegressorEvFiles_exp1(dataLog, outputEventFilePattern, relevantEvents,  runType, sub):
    """
    Create 3 column regressor event files (onset, duration, parametric mod.) 
    for stimulus events of interest for exp.1. 
    Removes the dummy volume duration from onsets! I.e. onsets are 
    relative to run onset, not imaging sequence onset.
    dataLog: MRI log file with events as returned by createMriLog_exp1
    outputEventFilePattern: file name pattern for event file
    relevantEvents: list of events of interest 
    runType: run type label
    sub: subject ID
    Returns: nothing, but writes event file .txt to output path per relevant
    event type
    """
    # loop over events of interest    
    for stimType in relevantEvents:
        # get onsets and durations from mri log
        onsets = dataLog['onset'][dataLog['trial_type']==stimType]
        # remove dummy volume duration
        onsets = onsets - (nDummyVols*TR)
        # get duration of event
        durations = dataLog['duration'][dataLog['trial_type']==stimType]
        # set parametric mod to 1s
        parametricMod = np.ones((len(onsets)))
        # make 3 column format
        event = np.vstack((onsets,durations,parametricMod)).T
        # save event file
        fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType, 'ses':sessionLabel}
        saveEventFile(fname,event,fill_empty)
    
    # create extra event file for end of run screen
    addRegressorEvFiles_exp1_endOfRun(dataLog, outputEventFilePattern, runType, sub)


def createRegressorEvFiles_exp1_withTaskRelevance(dataLog, outputEventFilePattern, relevantEvents,  runType, sub):
    """"
    Create 3 column regressor event files (onset, duration, parametric mod.) 
    for stimulus events of interest for exp.1 while also splitting events into 
    task relevance conditions. 
    Removes the dummy volume duration from onsets! I.e. onsets are 
    relative to run onset, not imaging sequence onset.
    dataLog: MRI log file with events as returned by createMriLog_exp1
    outputEventFilePattern: file name pattern for event file
    relevantEvents: list of events of interest 
    runType: run type label
    sub: subject ID
    Returns: nothing, but writes event file .txt to output path per relevant
    event type
    """
    # loop over events of interest    
    for stimType in relevantEvents:
        # get onsets and durations from mri log
        onsets = dataLog['onset'][dataLog['task_relevance']+'_'+dataLog['trial_type']==stimType]
        # remove dummy volume duration
        onsets = onsets - (nDummyVols*TR)
        # get duration of event
        durations = dataLog['duration'][dataLog['task_relevance']+'_'+dataLog['trial_type']==stimType]
        # set parametric mod to 1s
        parametricMod = np.ones((len(onsets)))
        # make 3 column format
        event = np.vstack((onsets,durations,parametricMod)).T
        # save event file
        fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':stimType, 'ses':sessionLabel}
        saveEventFile(fname,event,fill_empty)
    
    # create extra event file for end of run screen
    addRegressorEvFiles_exp1_endOfRun(dataLog, outputEventFilePattern, runType, sub)


def addRegressorEvFiles_exp1_endOfRun(dataLog, outputEventFilePattern, runType, sub):    
    """
    Create 3 column regressor event files (onset, duration, parametric mod.) 
    for end of run event, which is otherwise not explicitly logged. Duration 
    is fixed in function.
    Removes the dummy volume duration from onsets! I.e. onsets are 
    relative to run onset, not imaging sequence onset. 
    dataLog: MRI log file with events as returned by createMriLog_exp1
    outputEventFilePattern: file name pattern for event file
    runType: run type label
    sub: subject ID
    Returns: nothing, but writes event file .txt to output path
    """
    # create end of run event
    onsets = dataLog['onset'][len(dataLog)-1]
    # remove dummy volume duration
    onsets = onsets - (nDummyVols*TR)
    # set duration of event
    durations = 30 # set duration to 30 sec (i.e. sufficiently long), because we don't have an 'end of run' event logged, nor any other means to estimate the actual run offset
    parametricMod = 1
    # make 3 column format
    event = np.vstack((onsets,durations,parametricMod)).T
    # save event file
    fname = outputEventFilePattern%{'sub':sub, 'runType':runType, 'eventType':'endOfRun', 'ses':sessionLabel}
    saveEventFile(fname,event)


# %% run
if __name__ == '__main__':
    """
    Create regressors event txt files for each subject from events.tsv file.
    Writing FSL FEAT compliant 3 column regressor event files for each relevant 
    event type of Exp.1.
    bidsDir: bids directory
    """
    # get subject list
    subjects = get_subject_list(bidsDir,subject_list_type)
    # loop over subjects
    for sub in subjects:
        sub = sub[4::]
        print('========== SUBJECT: ' + sub + ' ==========')
        
        # loop over runs for current subject
        for runIdx in range(nRuns):
            
            # label for run
            runType = 'task-Dur_run-' + str(runIdx+1)
            
            ##### load events.tsv file per run #####
            fname = eventsTsvFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel}
            try:
                log = pd.read_csv(fname, sep='\t')
            except Exception as e:
                print(e)
                print(' - CHECK EVENTS.TSV FILE ! -')
                print(' - SKIPPING RUN ! - ')
                continue
            
            ##### Create event files for GLM per run#####
            print('... Creating regressor txt files for run: ' + str(runIdx+1))
            createRegressorEvFiles_exp1(log, outputEventFilePattern, relevantEvents,  runType, sub)
            createRegressorEvFiles_exp1_withTaskRelevance(log, outputEventFilePattern, relevantEventsWithTaskRelevance,  runType, sub)
            
    
