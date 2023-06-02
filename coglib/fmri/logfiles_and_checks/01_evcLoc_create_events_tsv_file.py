#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts events relevant for fMRI analysis from log file of EVC Localizer, then
creates events.tsv file per subject and run. Also writes assocaited json 
sidecars.
Performs various tests on log files.
Outputs MRI specific, bids compliant *_event.tsv file per run. 

Inputs:
    - log files from EVC localizer

Outputs: 
    - one log file checks table with warning flags 
    - one MRI specific, bids compliant *_event.tsv file per run with all 
    information relevant for analysis (bids compliant) 

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 12.10.2020

@author: David Richter (d.richter@donders.ru.nl)
@tag: prereg_v4.2
"""

# %% Imports & parameters
import pandas as pd 
import numpy as np
from glob import glob
import sys
from shutil import copyfile
pd.options.mode.chained_assignment = None  # default='warn'


##### Parameters #####
# session label
sessionLabel = 'V1'

# expected durations & trials (only used for checks)
stimulusDuration = 15.25                            # expected duration of stimulus events in seconds (actual duration taken from log file)
nExpectedStimuli = 20                               # expected number of stimulus trials
nExpectedNullEVs = 2                                # expected number of null events

# TR & number of dummy volumes (required to adjust timestamps of events.tsv file to INCLUDE non-removed dummy scans, as per bids specification)
TR = 1.5
nDummyVols = 3

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase2_V1'


##### Paths #####

# raw folder
rawDir = '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII'

# bid dir & code dir
bidsDir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'  
codeDir = bidsDir + '/code' 

# input log file paths & file names
logPathPattern =  rawDir + '/%(sub)s/%(sub)s_MR_%(ses)s/RESOURCES/BEHEVCLoc/'
logFile = '%(sub)sE%(run)s_Beh_EVCLoc.csv'

# json sidecar template (template file of json sidecar defining columns in events.tsv)
jsonSidecarTemplate = codeDir + '/logfiles_and_checks/events-json_file-templates/events-json_file-template_ses-V2_task-EVCLoc.json'

# output paths
# events.tsv file
outputLogFilePattern = bidsDir + '/sub-%(sub)s/ses-%(ses)s/func/sub-%(sub)s_ses-%(ses)s_%(runType)s_events'
# error flags csv file
outputErrorFlagPattern = bidsDir + '/derivatives/logfilechecks/sub-%(sub)s_ses-%(ses)s-EVCLoc_errorFlags.csv'


# import functions from helper_functions_MRI
sys.path.append(codeDir)
from helper_functions_MRI import saveErrorFlags, get_subject_list


# %% Log file checks 
def logChecks_EvcLocalizer(dataLog, TR, nExpectedNullEVs, nExpectedStimuli, stimulusDuration):
    """
    Perform various checks on the evc localizer log files.
    dataLog: MRI log file with events as returned by createMriLog_EvcLocalizer
    TrTime: duration of TR
    nExpectedNullEVs: number of null events expected per run
    nExpectedStimuli: range of expected number of stimuli per run
    stimulusDuration: stimulus duration per block
    Returns: list with error flags (see below for error codes)
    """
    # timestamps of all events (except for button presses)
    allEventTS = dataLog['Time'][(dataLog['EventID']=='TLBR') | (dataLog['EventID']=='TRBL') | (dataLog['EventID']=='Null')].to_numpy()
    # create error flags
    errLogFlag = []
    # 1. check the number of null events
    if np.sum(dataLog['EventID']=='Null') != nExpectedNullEVs:
        print('!!! WARNING: Number of null events does not correspond to expected number of null events. Obs=' + str(np.sum(dataLog['EventID']=='Null')) + '. Check log files !!!')
        errLogFlag.append(1)
    # 2. check that TLBR has 1/2 of the total expected stimulus events
    if np.sum(dataLog['EventID']=='TLBR') != nExpectedStimuli/2:
        print('!!! WARNING: Number of TLBR events does not correspond to expected number of TLBR events. Obs=' + str(np.sum(dataLog['EventID']=='TLBR')) + '. Check log files !!!')
        errLogFlag.append(2)
    # 3. check that TRBL has 1/2 of the total expected stimulus events
    if np.sum(dataLog['EventID']=='TRBL') != nExpectedStimuli/2:
        print('!!! WARNING: Number of TRBL events does not correspond to expected number of TLBR events. Obs=' + str(np.sum(dataLog['EventID']=='TRBL')) + '. Check log files !!!')
        errLogFlag.append(3)
    # 4. check that events take place in expected intervals (defined by stimulusDuration parameter); allowed error = 100ms (given blocked design with 15.25 stim duration)
    if any((abs(np.diff(allEventTS)-stimulusDuration)) > 0.1):
        print('!!! WARNING: Event(s) took place at an unexpected interval. Incorrect event intervals: n=' + str(np.sum(abs((np.diff(allEventTS)-stimulusDuration))>0.1)) + '. Max divergence in ms=' + str(int(np.max(abs(np.diff(allEventTS)-stimulusDuration))*1000)) + '  Check log files !!!')
        errLogFlag.append(4)
    # 5. check that the delay between time 0 and the first event is approx 1 TR; accuracy of 100ms
    if (allEventTS[0]-TR) > 0.1:
        print('!!! WARNING: First stimulus event is unexpectedly delayed. Obs time=' + str(allEventTS[0]) + '. Check log files !!!')
        errLogFlag.append(5)
    # 6. check that there are button presses at all
    if np.sum(dataLog['EventID']=='buttonPress')==0:
        print('!!! WARNING: No button presses found. Check log files !!!')
        errLogFlag.append(6)
    # note if no error was found
    if not errLogFlag:
        print('... EVC localizer run passed checks without error.')
        errLogFlag = 0
    return errLogFlag


# %% create bids compliant tsv log/event file
def createEventsTsv_EvcLocalizer(dataLog, outputLogFilePattern, sub, runType):
    """
    Create events.tsv log file per run with events relevant for MRI analysis.
    dataLog: raw log array with events, onsets, etc., as saved by evc localizer
    outputLogFilePattern: file name pattern for event tsv file
    sub: subject ID
    runType: run type label
    Returns: MRI log array with onsets, event types, etc. (bids compliant)
    """
    print('... Creating MRI events.tsv file')
    # get onsets and labels for all stimulus (and null) events and calculate their durations 
    allStimEvents_noPresses = (dataLog['EventID']=='TLBR') | (dataLog['EventID']=='TRBL') | (dataLog['EventID']=='Null')
    allEventTS = dataLog['Time'][allStimEvents_noPresses].to_numpy()
    allDurations = np.append(allEventTS[1::] - allEventTS[0:-1], stimulusDuration) 
    allEvents = dataLog['EventID'][allStimEvents_noPresses].to_numpy()
    # add onsets and labels and set durations of all button press events
    allEventTS = np.append(allEventTS, dataLog['Time'][dataLog['EventID']=='buttonPress'].to_numpy())
    allDurations = np.append(allDurations, np.zeros(np.sum([dataLog['EventID']=='buttonPress'])))
    allEvents = np.append(allEvents, dataLog['EventID'][dataLog['EventID']=='buttonPress'].to_numpy())
    # offset all onset times by the number of dummy volumes * TR. 
    # This step is required for the events.tsv file, because BIDS specificiation requires time 0 = first volume of BIDS imaging data file. 
    # However, all onset times recorded in the exp.1 log file are referenced to the first event per run, the run onset time, which is triggered by the 4th volume per run (i.e. after 3 dummy volumes)
    onset = allEventTS + (nDummyVols*TR)
    # create output df
    outputDF = pd.DataFrame({'onset':onset, 'duration':allDurations, 'trial_type':allEvents})
    # sort by onsets
    outputDF = outputDF.sort_values(by=['onset'])
    outputDF = outputDF.reset_index(drop=True)
    # save df
    fname = outputLogFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel} + '.tsv'
    try:
        outputDF.to_csv(fname, sep='\t', index=False, na_rep='Null')
    except Exception as e:
        print(e)
    
# %%
if __name__ == '__main__':
    """
    Create MRI events.tsv log files for current run & subject. First, get log
    file path+name, count number of dicoms per run. Then performs various log 
    file checks. Next, create an MRI relevant events.tsv file from the native 
    EVC localizer log file. 
    bidsDir: bids directory
    """
    # get subject list
    subjects = get_subject_list(bidsDir,subject_list_type)
    
    # loop over subjects    
    for sub in subjects:        
        sub = sub[4::]
        print('========== SUBJECT: ' + sub + ' ==========')
        
        # import relevant log files
        logPath = logPathPattern%{'sub':sub, 'ses':sessionLabel}
        fname = logPath + logFile%{'sub':sub, 'run':'*'}
        
        nRuns = len(glob(fname))
        # check if multiple runs exist and if so request which log file to use
        if not nRuns == 1:
            print(' - CAUTION ! --> ' + str(nRuns) + ' <-- EVC localizer runs found for subject: ' + sub + ', while exactly 1 run was expected.')
            print(' - SKIPPING PARTICIPANT ! - ')
            continue
        fname = logPath + logFile%{'sub':sub, 'run':'*'}
        fname = glob(fname)[0]
        dataLog = pd.read_csv(fname) 
        
        # label for run type; note we assume that only the correct run has been transformed to BIDS and only multiple log files are stored
        runType = 'task-EVCLoc_run-1'
        
        ##### Checks #####
        errLogFlag = logChecks_EvcLocalizer(dataLog, TR, nExpectedNullEVs, nExpectedStimuli, stimulusDuration)
        
        ##### Create log file with relevant events per run #####
        createEventsTsv_EvcLocalizer(dataLog, outputLogFilePattern, sub, runType)
        
        # create json sidecar for events.tsv file from template json file
        fname = outputLogFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel} + '.json'
        try:
            copyfile(jsonSidecarTemplate, fname)
        except Exception as e:
            print(e)
        
        # save log file warning flags
        fname = (outputErrorFlagPattern%{'sub':sub,'ses':sessionLabel}) 
        errLogDf = pd.DataFrame({"WarningCode": [errLogFlag]})
        saveErrorFlags(fname, errLogDf)