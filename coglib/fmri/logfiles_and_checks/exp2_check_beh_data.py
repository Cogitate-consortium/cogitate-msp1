#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check behavioral data of exp2 (VGR & Replay) - seen & FA
Finds exp2 sessions in raw data path.
Extracts summary files from VGR and Replay.
Writes output csv file (exp2_beh_data_checks) with seen, unseen & FA to raw data path.

NOTE: This code is NOT relevant for the final analysis pipeline, 
but was temporarily used to assess behavioral data during data acquisition

@author: David Richter
"""

import pandas as pd
import os, glob

# project root folder
rawDataPath = '/project/3018050.01/raw'

# subject & session folder prefixes & labels for exp2
subjectPrefix = 'sub-SC'
expTwoSessionFolder = 'ses-mri02'

# %% find data to be processed 
def findRawData(rawDataPath):
    """
    Find file names + paths per subject for exp2 summary json files.
    Assumes subjects are prefixed with sub-SC and exp2 sessions with ses-mri02
    """
    fNames_Prescreen = []
    fNames_VGR = []
    fNames_Replay = []
    # find subjects (raw and processed)
    for (_, rawDirs, _) in os.walk(rawDataPath):break    
    n_sub_raw = 0
    for sid in rawDirs:
        if sid[0:len(subjectPrefix)] == subjectPrefix:
            n_sub_raw += 1
    print('Found RAW data of   ' + str(n_sub_raw) + '   subjects in path: ' + rawDataPath)
    # get paths to data per subj
    for sid in sorted(rawDirs):
        # make sure it's a completed subject, indicated by the subjectPrefix
        if sid[0:len(subjectPrefix)] == subjectPrefix:
            for (rootSubDir, sesDirs, _) in os.walk(rawDataPath + os.sep + sid):break
            # find sessions to be processed, indicated by the expTwoSessionFolder
            for curSes in sesDirs:
                if curSes==expTwoSessionFolder:
                    print('. found exp2 data for subject: ' + sid)
                    fNames_Prescreen.append(glob.glob(rootSubDir + os.sep + curSes + os.sep + sid[4::] + os.sep + 'A' + os.sep + 'Summaries'  + os.sep +  '*_ProbeSummary_Game.json')[0])
                    fNames_VGR.append(glob.glob(rootSubDir + os.sep + curSes + os.sep + sid[4::] + os.sep + '1' + os.sep + 'Summaries'  + os.sep +  '*_ProbeSummary_Game.json')[0])
                    fNames_Replay.append(glob.glob(rootSubDir + os.sep + curSes + os.sep + sid[4::] + os.sep + '1' + os.sep + 'Summaries'  + os.sep +  '*_ProbeSummary_Replay.json')[0])
    # return file path list
    return fNames_Prescreen, fNames_VGR, fNames_Replay


# %% get data from json files
def getBehGame(fNames):
    """
    Calculate seen, unseen & FA percent for VGR summary json files
    """
    seen = []
    unseen = []
    seen_percent = []
    unseen_percent = []
    fa = []
    fa_percent = []
    for fName in fNames:
        df = pd.read_json(fName, typ='series')
        seen.append(df['TotalNumberOfAllStimuliSeen'])
        unseen.append(df['TotalNumberOfAllStimuliUnseen'])
        fa.append(df['TotalNumberOfBlanksFalseAlarms'])
        seen_percent.append(100 * (df['TotalNumberOfAllStimuliSeen'] / (df['TotalNumberOfFacesProbed'] + df['TotalNumberOfObjectsProbed'])))
        unseen_percent.append(100 * (df['TotalNumberOfAllStimuliUnseen'] / (df['TotalNumberOfFacesProbed'] + df['TotalNumberOfObjectsProbed'])))
        fa_percent.append(100 * (df['TotalNumberOfBlanksFalseAlarms'] / df['TotalNumberOfBlanksProbed']))
    return seen, unseen, fa, seen_percent, unseen_percent, fa_percent

def getBehReplay(fNames):
    """
    Calculate seen, missed & FA percent for Replay summary json files
    """
    sub_id = []
    hits = []
    hits_percent = []
    misses = []
    fa = []
    fa_percent = []
    for fName in fNames:
        sub_id.append(fName[fName.find('sub-')+4 : fName.find('sub-')+9])
        df = pd.read_json(fName, typ='series')
        hits.append(df['TotalNumberOfHitsInFaceTarget'] + df['TotalNumberOfHitsInObjectTarget'])
        misses.append(df['TotalNumberOfMissesInFaceTarget'] + df['TotalNumberOfMissesInObjectTarget'])
        fa.append(df['TotalNumberOfFalseAlarmsInFaceTarget'] + df['TotalNumberOfFalseAlarmsInObjectTarget'])
        hits_percent.append(100 * (hits[-1] / (hits[-1]+misses[-1])))
        fa_percent.append(100 * (fa[-1] / (df['TotalNumberOfBlanksPresentedInFaceTarget'] + df['TotalNumberOfBlanksPresentedInObjectTarget'] + df['TotalNumberOfObjectsPresentedInFaceTarget'] + df['TotalNumberOfFacesPresentedInObjectTarget'])))
    return sub_id, hits, misses, fa, hits_percent, fa_percent


# %% load json files
[fNames_Prescreen, fNames_VGR, fNames_Replay] = findRawData(rawDataPath)

# get data
[vgr_seen, vgr_unseen, vgr_fa, vgr_seen_percent, vgr_unseen_percent, vgr_fa_percent] = getBehGame(fNames_VGR)
[sub_id, replay_hits, replay_misses, replay_fa, replay_hits_percent, replay_fa_percent] = getBehReplay(fNames_Replay)

# create output df
dat = pd.DataFrame({'Subject_ID':sub_id, 'Replay_Hits_Percent':replay_hits_percent, 'Replay_FA_Percent':replay_fa_percent, 'VGR_Seen_Percent':vgr_seen_percent, 'VGR_Unseen_Percent':vgr_unseen_percent, 'VGR_FA_Percent':vgr_fa_percent})
outputName = rawDataPath + '/exp2_beh_data_checks_' + subjectPrefix + '.csv'
dat.to_csv(outputName, sep=',', index=False, na_rep='Null')
print('Writing output file as: ' + outputName)