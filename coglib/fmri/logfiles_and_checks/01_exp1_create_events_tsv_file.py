#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts events relevant for fMRI analysis from log file of exp.1 and creates
events.tsv file per subject and run. Also writes assocaited json sidecars.
Performs various tests on log files.
Outputs MRI specific, bids compliant *_event.tsv file per run. 

Inputs:
    - log files from exp.1

Outputs: 
    - one log file checks table with warning flags
    - one MRI specific, bids compliant *_event.tsv file per run with all 
    information relevant for analysis (bids compliant) 

Tested on python v3.7.4, pandas v0.25.2, numpy v1.17.2

Created 23.10.2020

@author: David Richter (david.richter.work@gmail.com)
@tag: prereg_v4.2
"""

# %% Imports & parameters
import pandas as pd 
import numpy as np
import sys, os
from shutil import copyfile

pd.options.mode.chained_assignment = None  # default='warn'


##### Parameters #####
# session label
sessionLabel = 'V1'

# experiment design parameters
nRuns = 8                               # expected number of runs
nExpectedStimuli = [17,18,19]           # expected number of stimuli per miniblock
nExpectedNullEVs = 3                    # expected number of null events per run

# TR & number of dummy volumes (required to adjust timestamps of events.tsv file to INCLUDE non-removed dummy scans, as per bids specification)
TR = 1.5
nDummyVols = 3

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase2_V1'
#subject_list_type = 'debug'


##### Paths #####

# raw folder
rawDir = '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII'

# bid dir & code dir
bidsDir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'  
codeDir = bidsDir + '/code' 

# input log file paths & file names
logPathPattern =  rawDir + '/%(sub)s/%(sub)s_MR_%(ses)s/RESOURCES/BEH/'
logFileName = '%(sub)s_Beh_V1_RawDurR%(runIdx)s%(restartedFlag)s.csv'

# json sidecar template (template file of json sidecar defining columns in events.tsv)
jsonSidecarTemplate = codeDir + '/logfiles_and_checks/events-json_file-templates/events-json_file-template_ses-V1_task-Dur.json'

# output paths & file names
# events.tsv file
outputLogFilePattern = bidsDir + '/sub-%(sub)s/ses-%(ses)s/func/sub-%(sub)s_ses-%(ses)s_%(runType)s_events'
# error flags csv file
outputErrorFlagPattern = bidsDir + '/derivatives/logfilechecks/sub-%(sub)s_ses-%(ses)s_errorFlags.csv'


# import functions from helper_functions_MRI
sys.path.append(codeDir)
from helper_functions_MRI import saveErrorFlags, get_subject_list


"""
Exp1. stimulus log code:
% 1st digit = stimulus type (1 = face; 2 = object; 3 = letter; 4 = false font)
% 2nd digit = stimulus orientation (1 = center; 2 = left; 3 = right)
% 3rd & 4th digits = stimulus id (1...20; for faces 1...10 is male, 11...20 is female)
% e.g., "1219" = 1 is face, 2 is left orientation and 19 is a female stimulus #19
% The decimal is for duration
"""

#%%
def createEventsTsv_exp1(dataLog):
    """
    Creates events.tsv file per run with events relevant for MRI analysis in a 
    bids compliant fashion.
    events.tsv file contains columns for: 
        - onsets = Onset (in seconds) of the event measured from the beginning 
        of the acquisition of the first volume in the corresponding task 
        imaging data file. Note time 0 refers here to the onset of the imaging 
        sequence, not the onset of the run, which is delayed by 3 dummy TRs 
        (i.e., 4.5s). 
        - duration = Duration of the event (measured from onset) in seconds.
        - trial_type = Primary categorisation of each trial to identify them as 
        instances of the experimental conditions. Contains stimulus category
        such as face, object, letter, falseFont, targetScreen, etc.
        - task_relevance = Task relevance of stimulus.
        - stimulus_orientation = Orientation of stimulus.
        - stimulus_id = identity of stimulus (see log code above).
        - response = participant's response (hit, FA, etc.).
    dataLog: raw log array with events, onsets, etc., as saved by exp1.
    Returns: MRI log array with onsets, event types, etc. (bids compliant).
    """
    # fields for log file
    onset = []
    duration = []
    types = []
    taskRelevance = []
    stimId = []
    stimOrientation = []
    response = []
    
    # loop over log file entries and handle by type
    for idx in range(len(dataLog)):
        
        # handle normal stimulus events
        if dataLog['eventType'][idx] == 'Stimulus':
            onset = np.append(onset, dataLog['time'][idx] - dataLog['time'][dataLog['eventType'] == 'RunOnset'][0])
            stimId = np.append(stimId, str(int(dataLog['event'][idx])))
            
            # determine stimulus class
            if str(dataLog['event'][idx])[0] == '1': 
                types = np.append(types, 'face')
            elif str(dataLog['event'][idx])[0] == '2': 
                types = np.append(types, 'object')
            elif str(dataLog['event'][idx])[0] == '3': 
                types = np.append(types, 'letter')
            elif str(dataLog['event'][idx])[0] == '4': 
                types = np.append(types, 'falseFont')
                
            # determine task relevance of stimulus
            # remove orientation indicator since orientation does not matter for targets
            targets = [str(dataLog['targ1'][idx])[0] + str(dataLog['targ1'][idx])[2:4], str(dataLog['targ2'][idx])[0] + str(dataLog['targ2'][idx])[2:4]]
            curStim = str(int(dataLog['event'][idx]))[0] + str(int(dataLog['event'][idx]))[2::] 
            if curStim in targets:
                taskRelevance = np.append(taskRelevance, 'target')
            elif curStim[0] in [str(targets[0])[0], str(targets[1])[0]]:
                taskRelevance = np.append(taskRelevance, 'relevant')
            elif not curStim[0] in [str(targets[0])[0], str(targets[1])[0]]:
                taskRelevance = np.append(taskRelevance, 'irrelevant')
            else:
                taskRelevance = np.append(taskRelevance, 'x')
                
            # determine stimulus orientation
            if str(dataLog['event'][idx])[1] == '1': 
                stimOrientation = np.append(stimOrientation, 'center')
            elif str(dataLog['event'][idx])[1] == '2': 
                stimOrientation = np.append(stimOrientation, 'left')
            elif str(dataLog['event'][idx])[1] == '3': 
                stimOrientation = np.append(stimOrientation, 'right')
            
            # determine response to stimulus 
            # check if there is no response in the current miniblock and trial when no response is expected -> correct rejection
            if (dataLog['dsrdResponse'][idx] == 0) & (not np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response'))):
                response = np.append(response, 'correctRejection')
            # check if there is no response in the current miniblock and trial when a response is expected -> miss
            elif (dataLog['dsrdResponse'][idx] == 1) & (not np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response'))):
                response = np.append(response, 'miss')
            # check if there is response in the current miniblock and trial when no response is expected -> false alarm
            elif (dataLog['dsrdResponse'][idx] == 0) & np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response')):
                response = np.append(response, 'falseAlarm')
            # check if there is response in the current miniblock and trial when a response is expected -> hit
            elif (dataLog['dsrdResponse'][idx] == 1) & np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response')):
                response = np.append(response, 'hit')
            
            # determine duration by checking next fixation event
            nextFixIdx = dataLog.index[dataLog['eventType'] == 'Fixation'][[dataLog.index[dataLog['eventType'] == 'Fixation'] > idx][0]][0]
            duration = np.append(duration, dataLog['time'][nextFixIdx] - dataLog['time'][idx])
            
        # handle response events
        elif dataLog['eventType'][idx] == 'Response':
            onset = np.append(onset, dataLog['time'][idx] - dataLog['time'][dataLog['eventType'] == 'RunOnset'][0])
            types = np.append(types, 'response')
            taskRelevance = np.append(taskRelevance, 'response')
            stimId = np.append(stimId, str(dataLog['eventType'][idx]))
            stimOrientation = np.append(stimOrientation, dataLog['eventType'][idx])
            duration = np.append(duration, 0)
            if (dataLog['dsrdResponse'][idx] == 0) & np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response')):
                response = np.append(response, 'falseAlarm')
            elif (dataLog['dsrdResponse'][idx] == 1) & np.any((dataLog['trial'] == dataLog['trial'][idx]) & (dataLog['miniBlock'] == dataLog['miniBlock'][idx]) & (dataLog['eventType'] == 'Response')):
                response = np.append(response, 'hit')
        
        # handle null events
        elif dataLog['eventType'][idx] == 'Baseline':
            onset = np.append(onset, dataLog['time'][idx] - dataLog['time'][dataLog['eventType'] == 'RunOnset'][0])
            types = np.append(types, 'baseline')
            taskRelevance = np.append(taskRelevance, 'baseline')
            stimId = np.append(stimId, str(dataLog['eventType'][idx]))
            stimOrientation = np.append(stimOrientation, dataLog['eventType'][idx])
            # determine duration by checking next saving event
            nextFixIdx = dataLog.index[dataLog['eventType'] == 'Save'][[dataLog.index[dataLog['eventType'] == 'Save'] > idx][0]][0]
            duration = np.append(duration, dataLog['time'][nextFixIdx] - dataLog['time'][idx])
            response = np.append(response, 'baseline')
        
        # handle target screens
        elif dataLog['eventType'][idx] == 'TargetScreenOnset':
            onset = np.append(onset, dataLog['time'][idx] - dataLog['time'][dataLog['eventType'] == 'RunOnset'][0])
            types = np.append(types, 'targetScreen')
            taskRelevance = np.append(taskRelevance, 'targetScreen')
            stimId = np.append(stimId, 'targetScreen')
            stimOrientation = np.append(stimOrientation, 'targetScreen')
            # determine duration by checking next fixation event
            nextFixIdx = dataLog.index[dataLog['eventType'] == 'Fixation'][[dataLog.index[dataLog['eventType'] == 'Fixation'] > idx][0]][0]
            duration = np.append(duration, dataLog['time'][nextFixIdx] - dataLog['time'][idx])
            response = np.append(response, 'targetScreen')
    
    # offset all onset times by the number of dummy volumes * TR. 
    # This step is required for the events.tsv file, because BIDS specificiation requires time 0 = first volume of BIDS imaging data file. 
    # However, all onset times recorded in the exp.1 log file are referenced to the first event per run, the run onset time, which is triggered by the 4th volume per run (i.e. after 3 dummy volumes)
    onset = onset + (nDummyVols*TR)
    
    # create and return sorted df with columns listed below
    mriLog = pd.DataFrame({'onset':onset, 'duration':duration, 'trial_type':types, 'task_relevance':taskRelevance, 'stimulus_orientation':stimOrientation, 'stimulus_id':stimId, 'response':response})
    mriLog = mriLog.sort_values(by=['onset'])
    mriLog = mriLog.reset_index(drop=True)
    return mriLog


# %% Handle double restarts
def get_log_exp1(sub, runIdx, logPath):
    """
    Restarted runs should use csv files with _RESTARTED flag in the filename. 
    However, runs which were restarted more than once result in information 
    from multiple runs in one log file (without restarted flag). This 
    information is disentangled here and only the relevant runs information is 
    used for event file construction.
    sub: Subject ID
    runIdx: Run index
    logPath: Path to raw log file
    Returns: dataLog; array with log file data
    """
    
    # import relevant log files (if a restarted log file exists, load that, otherwise load log without suffix)
    fname_restarted = logPath + logFileName%{'sub':sub, 'runIdx':runIdx+1, 'restartedFlag':'_RESTARTED'}
    fname_normal = logPath + logFileName%{'sub':sub, 'runIdx':runIdx+1, 'restartedFlag':''}
    if (not os.path.isfile(fname_restarted)) and os.path.isfile(fname_normal):
        # if restarted log files do NOT exist, but normal log file exists, simply load log file and return 
        dataLog = load_log_file(fname_normal)
    elif os.path.isfile(fname_restarted) and (not os.path.isfile(fname_normal)):
        # if restarted log files do exist, check if file without any restart flag (_ABORT or _RESTARTED exist), if not simply load restarted log file and return
        dataLog = load_log_file(fname_restarted)
    elif os.path.isfile(fname_restarted) and os.path.isfile(fname_normal):
        # if both restarted and normal logs exist for same run this may indicate problems with the log file due to double restart of a run. Such log files need to be cleaned up before processing.
        dataLog = load_log_file(fname_normal)
        # find all RunOnset events and remove all events before the last run onset event (i.e. only use the final; correct run's data). Note: this assumes that the final is always the correct run after restarts (if there can be other cases this needs to be added)
        dataLog = dataLog[(dataLog.eventType == 'RunOnset').cumsum().idxmax():].reset_index(drop=True)
    else:
        # if no log file exists at all, return empty array
        dataLog = []
    return dataLog

def load_log_file(fname):
    """
    Try loading log file.
    Throw expections if not successful and return empty dataLog instead.
    Returns: dataLog; array with log file data
    """
    # load log
    try:
        dataLog = pd.read_csv(fname)
    except Exception as e:
        print(e)
        dataLog = []
    return dataLog

    
# %% Log file checks 
def logChecks_exp1(dataLog, nExpectedNullEVs, nExpectedStimuli, runIdx):
    """
    Perform various checks on the exp1 log files.
    dataLog: MRI log file with events as returned by createMriLog_exp1
    nExpectedNullEVs: number of null events expected per run
    nExpectedStimuli: range of expected number of stimuli per run
    runIdx: Index of run 
    Returns: list with error flags (see below for error codes)
    """
    # create error flags
    errLogFlag = []
    # 1. check the number of null events
    if np.sum(dataLog['eventType']=='Baseline') != nExpectedNullEVs:
        print('!!! WARNING: Number of null events does not correspond to expected number of null events. Obs=' + str(np.sum(dataLog['EventID']=='Null')) + '. Check log files !!!')
        errLogFlag.append(1)
    # 2. check the number of stimulus events
    for blkIdx in dataLog['miniBlock'].unique():
        if not np.any(np.sum((dataLog['eventType']=='Stimulus') & (dataLog['miniBlock']==blkIdx)) == nExpectedStimuli):
            print('!!! WARNING: Number of stimulus events does not correspond to expected number of stimulus events per miniblock. Obs=' + str(np.sum((dataLog['eventType']=='Stimulus') & (dataLog['miniBlock']==blkIdx))) + '. Check log files !!!')
            errLogFlag.append(2)
    # 3. check the total duration of the run (6min < duration < 8min)
    if not (360 < (dataLog['time'][len(dataLog)-1]-dataLog['time'][0]) < 480):
        print('!!! WARNING: Unexpected total run duration: Obs=' + str(round(dataLog['time'][len(dataLog)-1]-dataLog['time'][0],2)) + 'sec. Check log files !!!')
        errLogFlag.append(3)
    # 4. check that there are responses
    if np.sum((dataLog['eventType']=='Response') & (dataLog['event']==1)) == 0:
        print('!!! WARNING: No button presses found. Check log files !!!')
        errLogFlag.append(4)
    # note if no error was found
    if not errLogFlag:
        print('... Run: ' + str(runIdx+1) + ' passed checks without error.')
        errLogFlag = 0
    return errLogFlag


# %% run
if __name__ == '__main__':
    """
    Create MRI log files (events.tsv file) for each subject and run. 
    First, gets exp1 native log file path+name, then loops over runs 
    performing various log file checks before creating events.tsv file in the
    correct bids subdirector. Finally saves result of log file checks to
    error log csv file, summarizing any errors in the input log file to
    derivatives dir.
    bidsDir: bids directory
    """
    # get subject list
    subjects = get_subject_list(bidsDir,subject_list_type)
    # loop over subjects
    for sub in subjects:
        sub = sub[4::]
        print('========== SUBJECT: ' + sub + ' ==========')
        logPath = logPathPattern%{'sub':sub,'ses':sessionLabel}
        
        errLogFlag = [[],[]]
        
        # loop over runs for current subject
        for runIdx in range(nRuns):
            
            # get data log and handle possible restarts
            dataLog = get_log_exp1(sub, runIdx, logPath)
            
            # label for run
            runType = 'task-Dur_run-' + str(runIdx+1)
            
            ##### Checks #####
            errLogFlag[0].append('Run%s' %(runIdx+1))
            # check if dataLog is empty; if so, add error flag, if not submit to additional log checks
            if len(dataLog) == 0:
                print(' - NO DATA/LOG FOUND ! SKIPPING RUN ! - ')
                errLogFlag[1].append('9')
                continue
            else:
                errLogFlag[1].append(logChecks_exp1(dataLog, nExpectedNullEVs, nExpectedStimuli, runIdx))
            
            # write files
            print('... Creating MRI events.tsv file for run: ' + str(runIdx+1))
            
            ##### Create events.tsv log file with relevant events per run #####
            log = createEventsTsv_exp1(dataLog)
            # save log file
            fname = outputLogFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel} + '.tsv'
            try:
                log.to_csv(fname, sep='\t', index=False, na_rep='Null')
            except Exception as e:
                print(e)
            
            # create json sidecar for events.tsv file from template json file
            fname = outputLogFilePattern%{'sub':sub, 'runType':runType, 'ses':sessionLabel} + '.json'
            try:
                copyfile(jsonSidecarTemplate, fname)
            except Exception as e:
                print(e)
                
        # save log file warning flags
        fname = (outputErrorFlagPattern%{'sub':sub,'ses':sessionLabel}) 
        errLogDf = pd.DataFrame({"RunNo": errLogFlag[0], "WarningCode": errLogFlag[1]})
        saveErrorFlags(fname, errLogDf)
    
