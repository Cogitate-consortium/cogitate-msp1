import os
import numpy as np
import pandas as pd
import warnings
import copy
import ET_param_manager
import datetime
from ET_param_manager import Triggers
from progress.bar import IncrementalBar as Ibar
from based_noise_blinks_detection import based_noise_blinks_detection

""" Data Parsing Module

This module includes all the methods used to extract, parse and filter data.

@authors: RonyHirsch, AbdoSharaf98
"""


# Eyelink logs' line tags
OTHER = "OTHER"
EMPTY = "EMPTY"
COMMENT = 'COMMENT'
SAMPLE = "SAMPLE"
START = "START"
END = "END"
MSG = "MSG"
EFIX = "EFIX"
ESACC = 'ESACC'
EBLINK = "EBLINK"
START_REC_MARKER = '!MODE RECORD'


DF_REC = "dfRec"
DF_MSG = "dfMsg"
DF_FIXAT = "dfFix"
DF_SACC = "dfSacc"
DF_SACC_EK = f"{DF_SACC}EK"
DF_BLINK = "dfBlink"
DF_SAMPLES = "dfSamples"
EYE = "Eye"
EYELINK = "Eyelink"
EK = "EK"
HERSHMAN = "Hershman"
HERSHMAN_PAD = "HershmanPad"
HAS_ET_DATA = "has_ET_data"

# data columns
ONSET = 'stimOnset'
T_START = 'tStart'
T_END = 'tEnd'
T_SAMPLE = 'tSample'
VPEAK = 'vPeak'
AMP_DEG = 'ampDeg'
BLOCK_COL = "block"
MINIBLOCK_COL = "miniBlock"
ORIENTATION_COL = "stimOrientation"
TRIAL_COL = "trial"
RESP_TIME_COL = "responseTime"
STIM_TYPE_COL = "stimType"
STIM_COL = "stimID"
IS_TASK_RELEVANT_COL = "isTaskRelevant"
STIM_DUR_PLND_SEC = "plndStimulusDur"
STIM_DUR_COL = 'stimDur'
REAL_FIX = "RealFix"
REAL_SACC = "RealSacc"
REAL_PUPIL = "RealPupil"
TRIAL_NUMBER = "trialNumber"

# duration windows

STIM_LOC = "stimulusLocation"
PRE_STIM_DUR = 'PreStim'
FIRST_WINDOW = 'First'
SECOND_WINDOW = 'Second'
THIRD_WINDOW = 'Third'
EPOCH = 'Epoch'
TRIAL = "Trial"
BASELINE_START = "Baseline"

LOC = 'Location'
VIS = 'Visibility'
WORLD_G = 'Game World'
WORLD_R = 'Replay World'
WORLD = 'World'
CATEGORY = 'Category'
IS_LEVEL_ELIMINATED = "IS_LEVEL_ELIMINATED"
SACC_DIRECTION_RAD = "sacc_direction_rad"

SAMPLING_FREQ = 'SamplingFrequency'
WINDOW_START = 'WindowStart'
WINDOW_END = 'WindowEnd'


class Error(Exception):
    pass


class InputError(Error):
    def __init__(self, msg):
        self.message = msg


def is_probed(row):
    # PROBED: IF IN GAME LOCATION ENDS WITH 1 || IF IN REPLAY, ANY STIMULUS
    value = int(row["Location"])
    if value % 10:
        return True
    if row["WorldName"].endswith("_A") or row["WorldName"].endswith("_B"):
        return True
    return False


def set_visibility(category, response):
    """
    A function mapping between a stimulus type and the stimulus response AS RECEIVED FROM THE EYELINK TRIGGER MESSAGES.
    :param category:
    :param response:
    :return:
    """
    TP = "True Positive"
    TN = "True Negative"
    FP = "False Positive"
    FN = "False Negative"
    vis_dict = {("Blank", "No"): TN, ("Blank", "Yes"): FP, ("Blank", "NoResp"): TN,  ("Blank", "Resp"): FP,
                ("Face", "No"): FN, ("Object", "No"): FN, ("Face_target", "NoResp"): FN, ("Obj_target", "NoResp"): FN,
                ("Face", "Yes"): TP, ("Object", "Yes"): TP, ("Face_target", "Resp"): TP, ("Obj_target", "Resp"): TP,
                ("Face_non_target", "Resp"): FP, ("Obj_non_target", "Resp"): FP,
                ("Face_non_target", "NoResp"): TN, ("Obj_non_target", "NoResp"): TN,
                ("Obj_target", "No"): FN, ("Object", "Resp"): TP}  # THIS CASE SHOULDN'T EVEN HAPPEN, For some reason it was spotted in CD154 only
    return vis_dict[(category, response)]


def identify_missing_triggers(beh_stim, et_stim):
    # we take BEH as ground truth, as we reach this method only if BEH trials > ET trials
    ind_beh = 0
    ind_et = 0
    missing_trials = list()
    while ind_beh < beh_stim.shape[0] and ind_et < et_stim.shape[0]:
        if beh_stim.iloc[ind_beh, :].equals(et_stim.iloc[ind_et, :]):
            ind_beh += 1
            ind_et += 1
        else:  # there is a missing trial appearing in beh_stim that does not appear in et_stim
            missing_trials.append(ind_beh)
            ind_beh += 1
    return missing_trials


def compare_beh_triggers(beh_trial_data, et_trial_data, sub_code):
    """
    Compare the behavioral log outputs to eyetracking trigger messages to make sure that they agree on all trials'
    locations and stimulus identities. In case there is a mismatch, this is an issue that needs to be raised.
    If there is a match, then we can rely on beh_trial_data and simply add things from et_trial_data to it.
    :param beh_trial_data: output of behavioral data as it was parse in the behavioral QC from the log files
    :param et_trial_data: output of behavioral data as it was parsed by get_trial_info from eyelink trigger messages
    :param sub_code: subject code
    :return: whether they match or not
    """
    error = 0
    relevant_cols = [BLOCK_COL, MINIBLOCK_COL, TRIAL_COL, IS_TASK_RELEVANT_COL, STIM_TYPE_COL, ORIENTATION_COL, STIM_COL, STIM_DUR_COL]
    # prepare beh data for column comparison
    beh_trial_data[STIM_DUR_COL] = beh_trial_data.loc[:, STIM_DUR_PLND_SEC] * 1000  # seconds to ms
    beh_trial_data[STIM_DUR_COL] = beh_trial_data[STIM_DUR_COL].astype(np.int64)
    beh_trial_data[STIM_COL] = beh_trial_data[STIM_COL].astype(np.int64)
    # prepare et data for column comparison
    et_trial_data[BLOCK_COL] = et_trial_data[BLOCK_COL].astype(np.int64)

    # create beh df to test matching to trigger
    beh_stim = beh_trial_data[relevant_cols]
    et_stim = et_trial_data[relevant_cols]

    # compare
    comparison = et_stim.equals(beh_stim)
    # add a column in beh that marks for each trial whether it has ET data or not
    beh_trial_data.loc[:, "isTrigger"] = 1  # initialize as true

    if not comparison:
        if et_stim.shape[0] < beh_stim.shape[0]:
            if beh_stim.loc[:et_stim.shape[0] - 1, :].compare(et_stim).empty:
                # This is a case where the ET data is trimmed from the end. This means that the ET and BEH data
                # are identical if the BEH data is trimmed to include the same number of trials as the ET data
                beh_trial_data = beh_trial_data.loc[:et_stim.shape[0] - 1, :]  # trim manually
                print(f"{sub_code} ET DATA INCOMPLETE - TRIMMING BEH DATA TO MATCH")
                error = 1
                return error, beh_trial_data
            else:
                # missing triggers for BEH trials; need to locate them
                print(f"{sub_code} ISSUE WITH TRIAL MAPPING: SOME BEHAVIORAL DATA LOG TRIALS MISSING THEIR CORRESPONDING TRIGGERS")
                missing_trial_indices = identify_missing_triggers(beh_stim, et_stim)
                beh_trial_data.loc[missing_trial_indices, "isTrigger"] = 0  # trials missing ET triggers
                error = 1
                return error, beh_trial_data

        elif et_stim.shape[0] > beh_stim.shape[0]:
            print(f"{sub_code} ISSUE WITH TRIAL MAPPING: STIMULI TRIGGERS CONTRADICT THE BEHAVIORAL DATA LOGS: CHECK MANUALLY (B)")
            error = 1
            return error, beh_trial_data
        else:
            diff = et_stim.compare(beh_stim, keep_shape=True, keep_equal=True)  # for debugging, see what's wrong
            for col in relevant_cols:
                df1 = et_stim.loc[:, col]
                df2 = beh_stim.loc[:, col]
                mini_comparison = et_stim.equals(beh_stim)
                if not(mini_comparison):
                    diff_specific = et_stim.compare(beh_stim, keep_shape=False, keep_equal=False)  # this keeps ONLY the non-equal instances
                    for ind, row in diff_specific.iterrows():  # for each instance where et != beh
                        if np.isnan(row[diff_specific.columns[0]]):  # if this is because the ET df is MISSING that data
                            if not np.isnan(row[diff_specific.columns[1]]):  # if the behavioral df HAS that data
                                """
                                This case happened with subject CA125, where in one of the trials in miniblock 14
                                a trigger for the DURATION of a stimulus event is just missing, probably failed to send.
                                As the duration of that stimulus is also logged in the behavioral files, we can
                                complete that information based on the behavior.
                                """
                                # this means that the trigger with the information for that event was MISSING
                                col_name = diff_specific.columns[0][0]  # the original column name (trigger)
                                print(f"WARNING: {sub_code} ET DATA MISSING TRIGGER {col_name} - FIXED BY BEH DATA")
                                et_stim.loc[ind, col_name] = row[diff_specific.columns[1]]
                        else:
                            raise Exception("ISSUE WITH TRIAL MAPPING: STIMULI TRIGGERS CONTRADICT THE BEHAVIORAL DATA LOGS: CHECK MANUALLY (C)")  # THIS SHOULD NOT HAPPEN
                            #print(f"{sub_code} ISSUE WITH TRIAL MAPPING: STIMULI TRIGGERS CONTRADICT THE BEHAVIORAL DATA LOGS: CHECK MANUALLY (C)")
                            #error = 1
            return error, beh_trial_data
    else:
        print(f"{sub_code} ET triggers match BEH logs")
    return error, beh_trial_data


def set_trial_info(et_trial_data, beh_trial_data, sub_code, save_path):
    """
    Following the behavioral analysis of the video game, two key changes happen to the behavioral responses:
    1. Responses might be flipped : "negative" responses with a recodgnized delayed button press might have flipped
    to positive responses.
    2. Levels might have been eliminated: levels in which subjects' response is unacceptable (e.g., no button presses
    at all during the entire level) are eliminated from further analysis.
    Thus, the ET analysis should take the corrected behavioral data as the ground truth. We first compare the stimulus
    identity and location with compare_beh_triggers to see that ET triggers matched BEH logs, then we unify the
    information to return a comprehensive, correct, trial df
    :param et_trial_data: output of behavioral data as it was parsed by get_trial_info from eyelink trigger messages
    :param beh_trial_data: output of behavioral data as it was parse in the behavioral QC from the log files
    :return: beh_trial_data with timestamps from ET triggers, after making sure the sources agree
    """

    # Check equality between the stimuli in BEH and ET:
    beh_trial_data.reset_index(inplace=True, drop=True)
    et_trial_data.reset_index(inplace=True, drop=True)

    error, beh_trial_data = compare_beh_triggers(beh_trial_data, et_trial_data, sub_code)
    # after making sure the dfs are equal in terms of trial information, add the ET derived timings into BEH
    # note that now beh_trial_data has an additional column for each trial, whether or not it has a corresponding ET trigger
    beh_trial_data.to_csv(os.path.join(save_path, "trial_data_full.csv"), index=False)
    all_beh_trials = beh_trial_data.shape[0]  # total number of trials
    # as this is ET parsing, trials that for some reason don't have ET data are not interesting, so drop them but make sure to have the correct trial numbers
    beh_trial_data = beh_trial_data.loc[beh_trial_data["isTrigger"] == 1, :].reset_index(drop=False, inplace=False, names=[TRIAL_NUMBER])
    print(f"{beh_trial_data.shape[0]} of {all_beh_trials} trials have ET trigger data")
    # TODO: timestamps are taken as eyelink timestamps; make sure this is ok
    beh_trial_data.loc[:, ONSET] = et_trial_data.loc[:, ONSET]
    beh_trial_data.loc[:, ET_param_manager.STIM_OFFSET] = et_trial_data.loc[:, ET_param_manager.STIM_OFFSET]
    beh_trial_data.loc[:, ET_param_manager.JITTER_ONSET] = et_trial_data.loc[:, ET_param_manager.JITTER_ONSET]
    beh_trial_data.rename(columns={"responseEvaluation": VIS}, inplace=True)
    return beh_trial_data


def get_trial_info(allMsgDF, params, is_tobii=False):  # params
    """
    This function extracts the timestamps and basic information about trials (trials are basically probed stimuli).
    The information about all stimuli (location, id, whether they are probed, AND RESPONSES TO THE PROBES) is extracted
    directly from the TRIGGERS sent to the eyelink by the video game. The mapping of these triggers is an instance of
    ET_param_manager.Triggers class.
    It outputs a dataframe in which each line is a single trial, and each column is some information about it
    (e.g., world, timestamps of beginning and end of this trial, which stimulus was in this trial etc).
    :param allMsgDF: dataframe containing all of the Eye-Tracker messages (raw data)
    :return: trialInfo
    """
    triggers = Triggers()

    # get the messages sent from LPT
    is_LPT_message = [x.isdigit() for x in allMsgDF["text"].values]
    msgDF = allMsgDF.loc[is_LPT_message, :].reset_index(inplace=False, drop=True)
    # Remove lines with 0 in the text because they mean nothing
    msgDF = msgDF.loc[msgDF["text"] != "0"]

    curr_stim = None
    stim_counter = 0
    stims = []
    curr_miniblock = None
    for ind, msg in msgDF.iterrows():
        if curr_stim is None:  # This is only for the first time in this loop
            block_number = msg['BlockID'].split("Block ")[1]
            curr_stim = {BLOCK_COL: block_number}
        if int(msg["text"]) in triggers.MsgTypes:
            msg_type = triggers.MsgTypes[int(msg["text"])]
        else:
            if is_tobii:
                msg_type = "NONE"
                continue
            else:
                print("ERROR: This is an unexpected case; inspect manually")
            break
        if msg_type == ET_param_manager.STIMULUS:
            stim_counter += 1
            curr_stim[STIM_TYPE_COL] = triggers.Stimuli[int(msg["text"])]
            curr_stim[STIM_COL] = int(msg["text"]) - triggers.StimCode[curr_stim[STIM_TYPE_COL]]
            curr_stim[ONSET] = msg["time"]
        elif msg_type == ET_param_manager.STIM_DURATION:
            curr_stim[STIM_DUR_COL] = triggers.DurationMS[msg["text"]]
        elif msg_type == ET_param_manager.ORIENTATION:
            curr_stim[ORIENTATION_COL] = triggers.Orientation[msg["text"]]
        elif msg_type == ET_param_manager.TR:
            curr_stim[ET_param_manager.TR] = triggers.TaskRelevance[msg["text"]]
        elif msg_type == ET_param_manager.TRIAL:
            curr_stim[TRIAL_COL] = triggers.Trials[int(msg["text"])]
            stims.append(curr_stim)
            block_number = msg['BlockID'].split("Block ")[1]
            curr_stim = {BLOCK_COL: block_number, MINIBLOCK_COL: curr_miniblock}
        elif msg_type == ET_param_manager.RESP:
            """
            If we have seen the current stimulus, this is its' response time.
            Else, this is attributed to the last seen stimulus (previous stimulus, one before curr) and the response
            time will be listed there. The field is NaN if no response was made.
            """
            if stim_counter > len(stims):
                curr_stim[RESP_TIME_COL] = msg["time"]
            else:
                stims[-1][ET_param_manager.RESP] = msg["time"]
        elif msg_type == ET_param_manager.STIM_OFFSET:
            # In some subjects, there is an additional offset at the beggining, which makes a bad shift for stim timings.
            # In order to solve this, we assure that there was an onset before that offset - if there was no onset prior to this offset -DO NOTHING
            """
            If we have seen the current stimulus, this is its' response time.
            Else, this is attributed to the last seen stimulus (previous stimulus, one before curr) and the response
            time will be listed there. The field is NaN if no response was made.
            """
            if stim_counter > len(stims) or stim_counter == 0:
                curr_stim[ET_param_manager.STIM_OFFSET] = msg["time"]
            else:
                stims[-1][ET_param_manager.STIM_OFFSET] = msg["time"]
        elif msg_type == ET_param_manager.JITTER_ONSET:
            """
            Same logic as above, because the trial message is received before the jitter
            """
            if stim_counter > len(stims) or stim_counter == 0:
                curr_stim[ET_param_manager.JITTER_ONSET] = msg["time"]
            else:
                stims[-1][ET_param_manager.JITTER_ONSET] = msg["time"]
        elif msg_type == ET_param_manager.EXP_ONSET:
            x = 5  # do nothing
        elif msg_type == ET_param_manager.REC_ONSET:
            x = 5  # do nothing
        elif msg_type == ET_param_manager.REC_OFFSET:
            x = 5  # do nothing
        elif msg_type == ET_param_manager.MINIBLOCK:
            curr_miniblock = triggers.Miniblocks[int(msg["text"])]
            block_number = msg['BlockID'].split("Block ")[1]
            curr_stim[BLOCK_COL] = block_number
            curr_stim[MINIBLOCK_COL] = curr_miniblock
    trial_info = pd.DataFrame.from_records(stims)
    trial_info[IS_TASK_RELEVANT_COL] = trial_info[ET_param_manager.TR].map(ET_param_manager.TASK_RELEVANCE_MAP)

    """
    BLOCK NUMBER CORRECTION FOR MEG:
    As per the pre-registration (https://osf.io/gm3vd), "For M-EEG, stimuli were administered over 10 different runs,
    for fMRI over 8 runs, and for the iEEG, over 5 runs."

    For fMRI and iEEG, run = block = a separate durR file. Therefore, each file we parsed corresponded to a block.
    For MEG, a file contained more than one block in it (10 runs in 5 files). Therefore, when parsing ET data based on
    triggers and file numbers alone, creates a discrepancy between the counted and actual block numbers in the et df.

    Here, we implement a correction for that, based on the knowledge that EACH BLOCK CONTAINS 4 MINIBLOCKS IN IT.
    Pre-registration: "The experiment was divided into runs, with four blocks each. During each block, a ratio of 34-38 trials was presented"
    slab for trigger coding scheme: https://twcf-arc.slab.com/posts/eyetracker-and-meg-coding-scheme-pf78qe5y
    """
    if params['Modality'] == ET_param_manager.MEG:
        trial_info.loc[:, BLOCK_COL] = ((trial_info[MINIBLOCK_COL] - 1) / 4).astype(int) + 1
    return trial_info


def SequenceEyeData(et_sample_data, trial_info, params):
    """
    This function sequences the eye tracking samples into trials defined by EPOCH_START and EPOCH_END.
    Meaning, based on the trialInfo dataframe (which given information about the start and end of each trial),
    the current function splits the ET data samples df into a list of trials: trialData.
    Each element in trial data represents a single trial (for 4 worlds + 8 replay levels we'll have 400 trials=400
    elements). Each trial in this list is a dataframe, containing all the samples in this trial.
    :param et_sample_data: dataframe containing all ET "SAMPLE" lines (all samples in the experiment)
    :param trial_info: dataframe containing all the TRIAL information in the exp (including timestamps of trial beginning
    and end)
    :param params: the subjects' parameter set, including things like the sampling frequency of the Eye Tracker.
    :return: (1) trialData = a list of dataframes of samples. Each df is a single trial's sample collection.
    (2) trialInfo = the same dataframe as the input one, with extra columns.
    """
    # get the trial window start and end indices for each trial
    stimOnset = np.array(trial_info[ONSET])  # column of stimuli onsets for each trial (=probed stimulus)
    onsetInds = np.array([np.where(et_sample_data.tSample == onset)[0][0] for onset in stimOnset])  # index of eye sample which matches each stimulus onset
    # index of eye sample which matches the beginning of the epoch
    epoch_start_inds = list(np.ceil(onsetInds - (ET_param_manager.EPOCH_START / 1000) * params[SAMPLING_FREQ]).astype(int))
    # index of eye sample which matches the end of the epoch: EPOCH ENDS WITH RELATION TO STIMULUS ONSET!!!
    epoch_end_inds = list((np.ceil(onsetInds + (ET_param_manager.EPOCH_END / 1000) * params[SAMPLING_FREQ]) + 1).astype(int))
    # index of eye sample which matches the prestimulus beginning (end is stimonset)
    prestim_start_inds = list(np.ceil(onsetInds - (ET_param_manager.PRE_STIM_DUR / 1000) * params[SAMPLING_FREQ]).astype(int))
    # index of eye sample which matches the stimulus duration end (beginning is stimonset)
    stimdur_end_inds = list((np.ceil(onsetInds + (ET_param_manager.STIM_DUR / 1000) * params[SAMPLING_FREQ]) + 1).astype(int))

    epoch_begin = np.array(et_sample_data.iloc[epoch_start_inds, 0])
    epoch_end = np.array(et_sample_data.iloc[epoch_end_inds, 0])
    prestim_begin = np.array(et_sample_data.iloc[prestim_start_inds, 0])  # prestim end = stim Onset
    stimdur_end = np.array(et_sample_data.iloc[stimdur_end_inds, 0])  # stimdur begin = stim Onset

    # initialize outputs
    epoch_data = [None] * len(stimOnset)
    epoch_info = trial_info.copy(deep=True)

    epoch_info['EpochWindowStart'] = epoch_begin
    epoch_info['EpochWindowEnd'] = epoch_end
    epoch_info[PRE_STIM_DUR+"WindowStart"] = prestim_begin
    epoch_info[STIM_DUR + "WindowEnd"] = stimdur_end

    print('num trials = ' + str(len(stimOnset)))

    # loop through trials
    # start tracking progress
    bar = Ibar('Sequencing', max=len(stimOnset), suffix='%(percent)d%%')
    for trial in range(0, len(stimOnset)):  # for each (probed) stimulus

        # get the indices for the start and end with which to index the data array
        stIdx = epoch_start_inds[trial]
        endIdx = epoch_end_inds[trial]

        # get the trial's data
        epoch_data[trial] = et_sample_data.iloc[stIdx:endIdx, :]
        epoch_data[trial].reset_index(drop=True, inplace=True)

        # raise an error if the trial data is empty
        if epoch_data[trial].empty:
            raise InputError(f'Trial no. {trial}: Epoch has no samples, check the raw data')

        bar.next()
    bar.finish()

    epoch_info = epoch_info.reset_index(drop=True, inplace=False)  # get the first non-world-0 index

    return epoch_data, epoch_info


def GetEuclideanDistance(data, trialData_no_blinks, trialInfo, params):
    """
    This function gets the euclidean distance from (1) the fixation cross and (2) the stimulus. Based on that, it
    calculates the fixation proportion and mean distance for each condition.

    :param data: dataframe containing all ET "SAMPLE" lines (all samples) - meaning, this is all the sample data
    from the eye tracker messages
    :param trialData_no_blinks: a list of dfs where each element corresponds all the samples of a single trial
    WITHOUT THE BLINKS (meaning, AFTER filtering out of blinks).
    :param trialInfo: a df containing all trial info (should be the output of SequenceEyeData)
    :param params: the parameters dict of the subject. Should be the output of InitParams/UpdateParams

    :return:
    """
    trial_info_res = pd.DataFrame()
    trial_info_res = pd.concat([trial_info_res, trialInfo])

    refLocs = ['StimReference', 'CenterReference']
    timePeriods = [PRE_STIM_DUR, STIM_DUR, EPOCH]  # the analysis time periods relative to stimulus onset

    xcol = 'LX' if params['Eye'] == 'L' else 'RX'
    ycol = 'LY' if params['Eye'] == 'L' else 'RY'

    # initialize the onset time/index for all trials
    trialOnset = np.array(trialInfo[ONSET])

    # the starting inds for prestim
    # DIV by 1000 is because ET_param_manager.PRE_STIM_DUR is in MILLISECONDS and sampling frequency is the
    # number of samples per SECOND. So this is to convert from ms to sec, which then is converted to number of samples
    prestim_onset = np.array(trialInfo[PRE_STIM_DUR+"WindowStart"])

    # the ending indcs for during stim
    stimdur_offset = np.array(trialInfo[STIM_DUR + "WindowEnd"])

    # area used to determine fixation bounds for center
    refAngle = ET_param_manager.FIX_REF_ANGLE_RADIUS  # the radius of ° of visual angle which was defined as fixation stability
    fixAreaCenter = refAngle / params['DegreesPerPix']

    # get the stimulus locations and coordinates for all trials
    allStimLocs = trialInfo.Location
    allStimCoords = np.array([params['StimulusCoords'][n] for n in allStimLocs])
    allFixAreas = np.array([refAngle / params['DegreesPerPix']] * len(list(allStimLocs)))  # Fixation area in PIXELS

    centerCoords = params['ScreenCenter']

    # initialize outputs
    MeanFixationDist = dict.fromkeys(timePeriods)
    FixationProportion = dict.fromkeys(timePeriods)

    for k in MeanFixationDist.keys():
        MeanFixationDist[k] = dict.fromkeys(refLocs)
        FixationProportion[k] = dict.fromkeys(refLocs)
        for r in refLocs:
            MeanFixationDist[k][r] = []
            FixationProportion[k][r] = []

    # do the epochs first since we have sequenced data
    for tr in range(0, len(trialData_no_blinks)):  # loop over the trials
        epoch = trialData_no_blinks[tr]

        for ky in timePeriods:
            # get the relevant gaze data
            if ky == PRE_STIM_DUR:
                gaze = epoch.iloc[epoch[epoch.tSample == prestim_onset[tr]].index[0]:
                                  epoch[epoch.tSample == trialOnset[tr]].index[0]]
            elif ky == STIM_DUR:
                gaze = epoch.iloc[epoch[epoch.tSample == trialOnset[tr]].index[0]:
                                  epoch[epoch.tSample == stimdur_offset[tr]].index[0]]
            else:
                gaze = epoch

            gazeX = gaze[xcol]
            gazeY = gaze[ycol]

            for rf in refLocs:  # for each reference type
                coords = allStimCoords[tr, :] if rf == 'StimReference' else centerCoords
                fixArea = allFixAreas[tr] if rf == 'StimReference' else fixAreaCenter

                # calculate mean distance from reference, and fixation proportion
                distance = np.sqrt(((gazeX - coords[0]) ** 2) + ((gazeY - coords[1]) ** 2))  # dist in PIXELS!!!
                distance_in_degrees = distance * params['DegreesPerPix']  # convert to degrees
                meanDist = np.nanmean(distance_in_degrees)  # mean dists from target IN DEGREEES
                fixating = np.zeros(len(distance))  # BINARY array indicating: 1 = fixating (in area), 0 = not
                # NOTE: we assume fixArea is defined by RADIUS already so we DON'T divide by 2. If this size is DIAMETER then this needs to be divided by 2
                fixating[distance <= fixArea] = 1  # fixation area in pixels, distance in pixels
                fixProp = np.nansum(fixating) / len(fixating)

                # store
                # MEAN DISTANCES FROM CENTER OF TARGET (STIMULUS/FIX) in DEGREES
                trial_info_res.at[tr, 'DistFrom' + rf + "_" + ky] = meanDist
                # % of timestamps where gaze is within the TARGET RADIUS, 0 = OUTSIDE, 1 = INSIDE
                trial_info_res.at[tr, 'FixProp' + rf + "_" + ky] = fixProp

                if ky == EPOCH:  # when we calculate this for the entire epoch (all samples in trial) - save everything
                    trialData_no_blinks[tr]['DistFrom' + rf] = distance_in_degrees
                    trialData_no_blinks[tr]['IsFixating' + rf] = fixating

    return trialData_no_blinks, trial_info_res


def ParseEyeLinkAsc(elFilename, last_end_time, total_prev_diff):
    """
    This method reads in a single eyelink data file in an .asc file format, and produces readable dataframes for further
    analysis.
    :param elFilename: path to the eyelink data file
    :return: res_dict, which contains:
     -dfRec contains information about recording periods (often trials)
     -dfMsg contains information about messages (usually sent from stimulus software)
     -dfFix contains information about fixations
     -dfSacc contains information about saccades
     -dfBlink contains information about blinks
     -dfSamples contains information about individual samples
    """

    # Read in EyeLink file
    print('Reading in EyeLink file %s' % elFilename)
    f = open(elFilename, 'r')
    fileTxt0 = f.read().splitlines(True)  # split into lines
    fileTxt0 = list(filter(None, fileTxt0))  # remove emptys
    fileTxt0 = np.array(fileTxt0)  # concert to np array for simpler indexing
    f.close()

    # Separate lines into samples and messages
    print('Sorting lines')
    nLines = len(fileTxt0)
    lineType = np.array([OTHER] * nLines, dtype='object')
    iStartRec = list()
    for iLine in range(nLines):
        if fileTxt0[iLine] == "**\n" or fileTxt0[iLine] == "\n":
            lineType[iLine] = EMPTY
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = COMMENT
        elif bool(len(fileTxt0[iLine][0])) and fileTxt0[iLine][0].isdigit():
            fileTxt0[iLine] = fileTxt0[iLine].replace('.\t', 'NaN\t')
            lineType[iLine] = SAMPLE
        else:  # the type of this line is defined by the first string in the line itself (e.g. START, MSG)
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if START in fileTxt0[iLine]:
            # from EyeLink Programmers Guide: "The "START" line and several following lines mark the start of recording, and encode the recording conditions for the trial."
            iStartRec.append(iLine + 1)

    iStartRec = iStartRec[0]

    # ===== PARSE EYELINK FILE ===== #
    # Trials
    """ DEPRECATED
    print('Parsing recording markers')
    iNotStart = np.nonzero(lineType != START)[0]
    dfRecStart = pd.read_csv(elFilename, skiprows=iNotStart, header=None, delim_whitespace=True, usecols=[1])
    dfRecStart.columns = [T_START]
    iNotEnd = np.nonzero(lineType != END)[0]
    """
    """
    END lines mark the end of a block of data. The two values following the "RES" keyword are the average resolution
    for the block: if samples are present, it is computed from samples, else it summarizes any resolution data in the
    events. Note that resolution data may be missing: this is represented by a dot (".") instead of a number for the
    resolution.
    """
    """ DEPRECATED
    dfRecEnd = pd.read_csv(elFilename, skiprows=iNotEnd, header=None, delim_whitespace=True, usecols=[1, 5, 6])
    dfRecEnd.columns = [T_END, 'xRes', 'yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart, dfRecEnd], axis=1)
    nRec = dfRec.shape[0]
    print('%d recording periods found.' % nRec)
    """

    # Import Messages
    print('Parsing stimulus messages')
    iMsg = np.nonzero(lineType == MSG)[0]
    # set up
    tMsg = []
    txtMsg = []
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    """
    Convert dict to dataframe:
    The "MSG"s in the experiment's Ascii file are of 2 types: at the beginning of the recroding there are a lot of
    messages from EYELINK about the parameters of the ET and so on. Then, After the debugging ends ("---DEBUG END---")
    And the actual experiment starts, The video game SENDS TRIGGERS that are written as "MSG" lines for all types of
    events (which are coded in the Triggers class)
    """
    dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})

    # Import Fixations
    print('Parsing fixations')
    # From Eyelink Programmer's guide: "Fixation end events ("EFIX") are read by asc_read_efix() which fills the
    # variable a_efix with the start and end times, and average gaze position, pupil size,"
    # the information is: eye, start time, end time, duration, X position, Y position, pupil
    iNotEfix = np.nonzero(lineType != EFIX)[0]
    try:
        dfFix = pd.read_csv(elFilename, skiprows=iNotEfix, header=None, delim_whitespace=True, usecols=range(1, 8))
        dfFix.columns = ['eye', T_START, T_END, 'duration', 'xAvg', 'yAvg', 'pupilAvg']
    except Exception:  # meaning, NO FIXATIONS IN THIS FILE AT ALL (i.e., if we were to skip all iNotFIx rows, we were to be left with nothing)
        print(f"No fixations in {elFilename}")
        dfFix = pd.DataFrame(columns=['eye', T_START, T_END, 'duration', 'xAvg', 'yAvg', 'pupilAvg'])

    # Saccades
    print('Parsing saccades')
    # From Eyelink Programmer's guide: "Saccade end events ("ESACC") are read by asc_read_esacc() which fills the
    # variable a_esacc with the start and end times, start and end gaze position, duration, amplitude, and peak velocity."
    # the information is: eye, start time, end time, duration IN MILLISECONDS, start X position, start Y position,
    # end X position, end Y position, amplitude in DEGREES, peak velocity in DEGREES PER SECOND.
    # The total visual angle covered in the saccade is reported by the 'amplitude' parameter,
    # which can be divided by (<dur>/1000) to obtain the average velocity.
    iNotEsacc = np.nonzero(lineType != ESACC)[0]
    dfSacc = pd.read_csv(elFilename, skiprows=iNotEsacc, header=None, delim_whitespace=True, usecols=range(1, 11))
    dfSacc.columns = ['eye', T_START, T_END, 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', AMP_DEG, VPEAK]

    # Blinks
    print('Parsing blinks')
    # From Eyelink Programmer's guide:
    # The STARTBLINK and ENDBLINK events bracket parts of the eye-position data where the pupil size is very small,
    # or the pupil in the camera image is missing or severely distorted by eyelid occlusion.
    # Only the time of the start and end of the blink are recorded." (4.5.3.5 Blinks, Eyelink 1000 Plus user manual)
    # "Blink end events ("EBLINK") mark the reappearance of the eye pupil. These are
    # read by asc_read_-eblink() which fills the variable a_eblink with the start and end times, and duration. Blink
    # events may be used to label the next "ESACC" event as being part of a blink and not a true saccade."
    # more from EDF2ASC documentation: Blinks are always embedded in saccades, caused by artifactual motion as the
    # eyelids progressively occlude the pupil of the eye. Such artifacts are best eliminated by labeling an
    # SSACC...ESACC pair with one or more CBLINK events between them as a blink, not a saccade. The data contained in
    # the ESACC event will be inaccurate in this case, but the "tStart", "tEnd", and "duration" data will be accurate.
    # It is also useful to eliminate any short (less than 120 millisecond duration) fixations that precede or follow
    # a blink. These may be artificial or be corrupted by the blink.
    # right now we're just parsing everything so order doesn't matter
    iNotEblink = np.nonzero(lineType != EBLINK)[0]
    dfBlink = pd.read_csv(elFilename, skiprows=iNotEblink, header=None, delim_whitespace=True, usecols=range(1, 5))
    dfBlink.columns = ['eye', T_START, T_END, 'duration']

    # determine sample columns based on eyes recorded in file
    #eyesInFile = np.unique(dfFix.eye)
    eyesInFile = dfMsg[dfMsg["text"].str.contains("RECCFG")].iloc[0, 1].split()[-1]
    if len(eyesInFile) == 2:
        print('binocular data detected.')
        cols = [T_SAMPLE, 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile
        print(f"monocular data detected {eye}")
        cols = [T_SAMPLE, f"{eye}X", f"{eye}Y", f"{eye}Pupil"]
    # Import samples
    print('Parsing samples')
    iNotSample = np.nonzero(np.logical_or(lineType != SAMPLE, np.arange(nLines) < iStartRec))[0]
    dfSamples = pd.read_csv(elFilename, skiprows=iNotSample, header=None, delim_whitespace=True,
                            usecols=range(0, len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L', 'R']:
        if eye in eyesInFile:
            dfSamples[f"{eye}X"] = pd.to_numeric(dfSamples[f"{eye}X"], errors='coerce')
            dfSamples[f"{eye}Y"] = pd.to_numeric(dfSamples[f"{eye}Y"], errors='coerce')
            dfSamples[f"{eye}Pupil"] = pd.to_numeric(dfSamples[f"{eye}Pupil"], errors='coerce')
        else:
            dfSamples[f"{eye}X"] = np.nan
            dfSamples[f"{eye}Y"] = np.nan
            dfSamples[f"{eye}Pupil"] = np.nan

    # These variables are used in the case of a new file which starts BEFORE an older file.
    # In such a case, we need to add the endtime to each file we read from now on.
    is_problematic = False
    next_last_end_time = dfMsg.loc[dfMsg.shape[0] - 1, "time"]

    if dfMsg.loc[0, "time"] < last_end_time:
        is_problematic = True
        total_prev_diff += last_end_time
    # This means that we will add total_prev_diff to each of the dfs.
    # In a regular case, total_prev_diff will stay 0.
    """ DEPRECATED
    dfRec[T_START] = dfRec[T_START] + total_prev_diff
    dfRec[T_END] = dfRec[T_END] + total_prev_diff
    """

    dfMsg["time"] = dfMsg["time"] + total_prev_diff

    dfFix[T_START] = dfFix[T_START] + total_prev_diff
    dfFix[T_END] = dfFix[T_END] + total_prev_diff

    dfSacc[T_START] = dfSacc[T_START] + total_prev_diff
    dfSacc[T_END] = dfSacc[T_END] + total_prev_diff

    dfBlink[T_START] = dfBlink[T_START] + total_prev_diff
    dfBlink[T_END] = dfBlink[T_END] + total_prev_diff

    dfSamples[T_SAMPLE] = dfSamples[T_SAMPLE] + total_prev_diff

    res_dict = {DF_MSG: dfMsg, DF_FIXAT: dfFix, DF_SACC: dfSacc,  # DF_REC: dfRec,
                DF_BLINK: dfBlink, DF_SAMPLES: dfSamples}  # EYE: sub_eye
    # Return new compilation dataframe
    return res_dict, is_problematic, next_last_end_time


def et_data_mark_Eyelink(et_data_dict):
    """
    Mark blinks AS THEY APPEAR IN EYELINK (i.e., periods of missing data // Eylink-calculated blinks).
    The code  labels each sample in the samples dataframe as a EYELINK fixation or EYELINK saccade or
    EYELINK blink if it is one of these. - ALL SAMPLES MUST BE EITHER A BLINK, A FIXATION, OR A SACCADE.
    IF A SAMPLE IS NOT A BLINK, OR A SACCADE - IT MUST BE A FIXATION.

    This is based on Eyelink guidelines (from programmers' guide and chm file):
    In the Eyelink eye tracker (4.5.3.5. Eyelink 1000 Plus User Manual):
    Blinks are always preceded and followed by partial occlusion of the pupil, causing artificial changes in pupil
    position. These are sensed by the EyeLink 1000 Plus parser, and marked as saccades. The sequence of events produced is always:
    • STARTSACC
    • STARTBLINK
    • ENDBLINK
    • ENDSACC
    Note that the position and velocity data recorded in the ENDSACC event is not valid. All data between the STARTSACC
    and ENDSACC events should be discarded.
    - Labeling an SSACC...ESACC pair with one or more CBLINK events between them as a blink, not a saccade.
    - Eliminating any short (less than 120 millisecond duration) fixations that precede or follow a blink as these may
    be artificial or be corrupted by the blink. The end of fixation events will be marked by EFIX events, and those
    markers will immediately precede the SSACC event marker for the saccade surrounding the blink.  Similarly, the start
    of fixations immediately following blinks will be marked by an SFIX marker immediately after the ESACC marker
    containing the blink.  So, one strategy could be to find the EFIX/SFIX markers before/after each blink event and
    check the duration of those associated fixations to determine whether you want to discard them.


    :param et_data_dict: the dictionary of the subject's entire eye tracker information:
    - samples
    - blinks
    - fixations
    - saccades

    :return: a dictionary that contains
    - samples: each sample line labeled as a fixation/saccade/blink based on cleaned fixations and saccades
    - blinks
    - saccades: cleaned from artificial
    - fixations: cleaned from artificial
    """

    res_et_data = {DF_FIXAT: None, DF_SACC: None, DF_BLINK: et_data_dict[DF_BLINK], DF_SAMPLES: None,
                   DF_MSG: et_data_dict[DF_MSG]}

    # mark on Eyelink data guidelines and their stamping of FIX, SACC, BLINK
    fixs = et_data_dict[DF_FIXAT].sort_values(by=[T_START])
    saccs = et_data_dict[DF_SACC].sort_values(by=[T_START])
    bls = et_data_dict[DF_BLINK].sort_values(by=[T_START])
    samps = et_data_dict[DF_SAMPLES]

    for index, blink in bls.iterrows():
        # Labeling an SSACC...ESACC pair with one or more CBLINK events between them as BLINKS
        saccs.loc[(saccs[T_START] <= blink[T_START]) & (saccs[T_END] >= blink[T_START]) & (
                saccs['eye'] == blink['eye']), f"is_{EYELINK}Blink"] = blink[T_START]

    """
    note that fixations CANNOT contain blink events according to Marcus from eyelink, so this check should be satisfactory
    Labeling fixations that precede or follow a blink : "The end of fixation events will immediately precede the
    SSACC event marker for the saccade surrounding the blink. Similarly, the start of fixations immediately following
    blinks will be marked by an SFIX marker immediately after the ESACC marker containing the blink."
    - short
    - fixation tEnd immediately before the tStart of a false saccade (that surrounds a blink), or
    fixation tStart immediately after false saccade's tEnd
    """
    only_false_saccs = saccs[~pd.isna(saccs[f"is_{EYELINK}Blink"])]  # only saccades that have blinks in them
    min_fixation_dur = 120  # Based on Eyelink data guidelines
    for index, sacc in only_false_saccs.iterrows():
        fixs.loc[(fixs['duration'] < min_fixation_dur) &
                ((fixs[T_END] == sacc[T_START] - 1) | (fixs[T_START] == sacc[T_END] + 1)) &
                (fixs['eye'] == sacc['eye']), f"is_{EYELINK}Blink"] = sacc[f"is_{EYELINK}Blink"]  # if sacc['is_EyeLinkBlink'] is somehow nan, that would be the case in the fixs as well

    # put it in result dict
    res_et_data[DF_FIXAT] = fixs
    saccs.loc[:, ['xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg']] = saccs[['xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg']].replace("^.$", np.nan, regex=True)
    res_et_data[DF_SACC] = saccs.astype({'xStart': 'float64', 'yStart': 'float64', 'xEnd': 'float64', 'yEnd': 'float64', 'ampDeg': 'float64'})

    # MARK dfSamples as fixations and saccades ONLY samples that are NOT EYELINK BLINKS
    only_saccs = saccs[pd.isna(saccs[f"is_{EYELINK}Blink"])]
    only_fixs = fixs[pd.isna(fixs[f"is_{EYELINK}Blink"])]

    # add the (Eyelink + NOT EYELINK BLINK) saccade/fixation information to the sample data
    print(f"Blink information {datetime.datetime.now()}")
    for blink in bls.itertuples():
        samps.loc[samps[T_SAMPLE].between(blink.tStart, blink.tEnd), f"{blink.eye}{EYELINK}Blink"] = 1

    print(f"Saccade information {datetime.datetime.now()}")
    for sacc in only_saccs.itertuples():
        samps.loc[samps[T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"{sacc.eye}{EYELINK}Sacc"] = 1

    print(f"Adding blink, fixation, and saccade information to samples dataframe {datetime.datetime.now()}")
    for fix in only_fixs.itertuples():
        samps.loc[samps[T_SAMPLE].between(fix.tStart, fix.tEnd), f"{fix.eye}{EYELINK}Fix"] = 1

    res_et_data[DF_SAMPLES] = samps.reset_index(drop=True)
    print(f"Finished all {datetime.datetime.now()}")

    return res_et_data


def et_data_to_trials(et_data_prepro, trial_info, params, is_tobii=False):
    """
    In-place, change the preprocessed et_data dataframes (FIX, SACC, BLINK, SAMPLES) and trial_info to include columns
     indicating about trials. NOTE: trials are recognized by EPOCH start and end. Meaning, anything that's outside a
     trial's EPOCH is marked in et_data_prepro's dataframes column TRIAL as -1.

    :param is_tobii: whether the data is from a tobii eye tracker (or eyelink). All but one lab had Eyelink, so false.
    :param et_data_prepro: output of preprocess_et_data method: a dictionary containing:
    {DF_FIXAT: fixation dataframe with "is_blink" column labeling false fixations,
    DF_SACC: saccade dataframe with "is_blink" column labeling false saccade,
    DF_BLINK: et_data_dict[DF_BLINK],
    DF_SAMPLES: sample dataframe with a column per fix/sacc/blink per eye (e.g., Rblink)}
    :param trial_info: information about the trial and subject behavior as derived from the eyetracker TRIGGERs
    :param params: subject's parameters
    :return:
    - et_data_prepro: same dict of 4 ET dataframes, with a column named TRIAL marking the trial the sample belongs to.
    - trial_info: same dataframe with additional columns for the important time windows of each trial
    """

    # get the inds of the eye tracking data samples (DF_SAMPLES) that match stimulus onset
    probed_stim_onsets = np.array(trial_info[ONSET])  # remember, these onsets are derived from eyelink trigger messages
    trial_info.loc[:, HAS_ET_DATA] = True
    try:
        probed_stim_onsets_sample_inds = np.array([np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0][0] for onset in probed_stim_onsets])
    except IndexError:
        print(f"WARNING!: {params['SubjectName']} HAS EYELINK MGSS W/O CORRESPONDING SAMPLES")
        if params["SubjectLab"] == "CE":
            """
            Example: SE103
            For ECoG subjects, where the sampling rate was less than 1000Hz (500), trigger messages timestamps reflected
            to Eyelink device times that had no corresponding sample. Meaning, the stimulus trigger had a timestamp that
            did not appear in any sample. Therefore, to be able to compare Eyelink events with behavioral log events,
            we implement the following correction for these exceptional cases:
            """
            et_data_prepro[DF_SAMPLES]["aligned_timings"] = et_data_prepro[DF_SAMPLES][T_SAMPLE]//2
            probed_stim_onsets_sample_inds = np.array([np.where(et_data_prepro[DF_SAMPLES]["aligned_timings"] == onset//2)[0][0] for onset in probed_stim_onsets])
        else:
            """
            Example: CD118, CA125
            For some subjects (or at least this one), the eyetracking device did not track eye data during
            some interval mid-experiment. The result of this is that a trigger was sent to Eyelink, but no eye data
            was sampled at the time. For the analysis, the missing trial (i.e., a trial as was indicated in the
            trigger messages, that is missing gaze data) will be nullified completely to preserve dataframe coherence.
            This means that less trials are analyzed for this subject (only trials were gaze data was collected).

            *NOTE* : this is based on the stimOnset trigger - if the moment of stimOnset does not have any
            corresponding sample, then we assume this trial is lost, as we have no trackin of the eyes during the onset.
            """
            for onset in probed_stim_onsets:
                try:
                    ind = np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0][0]
                except Exception:
                    trial_info.loc[trial_info[ONSET] == onset, HAS_ET_DATA] = False

            probed_all = np.array([np.where(et_data_prepro[DF_SAMPLES][T_SAMPLE] == onset)[0] for onset in probed_stim_onsets])
            probed_stim_onsets_sample_inds = np.array([x[0] for x in probed_all if len(x) != 0])

    # get the inds of the eye tracking data samples that match different interesting events (e.g.epoch beginning)
    # and then add to trial_info the timestamps of the samples that match these events for each trial
    interesting_events = {EPOCH + WINDOW_START: ET_param_manager.EPOCH_START,  # EPOCH start (with respect to stim ONSET)
                          EPOCH + WINDOW_END: ET_param_manager.EPOCH_END,  # EPOCH end (with respect to stim ONSET)
                          BASELINE_START + WINDOW_START: ET_param_manager.BASELINE_START,
                          FIRST_WINDOW + WINDOW_END: ET_param_manager.FIRST_WINDOW_END,  # first window starts on stimOnset!
                          SECOND_WINDOW + WINDOW_END: ET_param_manager.SECOND_WINDOW_END,  # second window starts when first ends!
                          THIRD_WINDOW + WINDOW_END: ET_param_manager.THIRD_WINDOW_END}

    trial_info_valid_idx = trial_info[HAS_ET_DATA] == True
    for event in interesting_events:
        event_time = interesting_events[event] / 1000  # div by 1000 to turn MILLISECONDS TO SECONDS
        rel = event_time * params[SAMPLING_FREQ] if "End" in event else event_time * params[SAMPLING_FREQ] * (-1)
        event_sample_inds = list(np.ceil(probed_stim_onsets_sample_inds + rel).astype(int))
        event_samples = np.array(et_data_prepro[DF_SAMPLES].iloc[event_sample_inds, 0])
        trial_info.loc[trial_info_valid_idx, event] = event_samples

    # prepare a column in all ET dataframes to contain the trial number of samples within each epoch
    time_windows = {TRIAL: (EPOCH + WINDOW_START, EPOCH + WINDOW_END),
                    BASELINE_START: (BASELINE_START + WINDOW_START, ONSET),
                    FIRST_WINDOW: (ONSET, FIRST_WINDOW + WINDOW_END),
                    SECOND_WINDOW: (FIRST_WINDOW + WINDOW_END, SECOND_WINDOW + WINDOW_END),
                    THIRD_WINDOW: (SECOND_WINDOW + WINDOW_END, THIRD_WINDOW + WINDOW_END)}

    for key in et_data_prepro.keys():
        for window in time_windows:
            et_data_prepro[key][window] = -1

    """
    TOBII data does not include pupil and blink data. Therefore, these are not marked in subjects who have tobii data.
    """
    if not is_tobii:
        dfs_to_mark = [DF_BLINK, DF_FIXAT, DF_SACC]
    else:
        dfs_to_mark = []

    """
    The following loop marks samples/saccades/fixations as belonging to specific trials.
    Notably, trials where the gaze data was lost ("has_et_data" column is False) ARE NOT MARKED, so that they will not
    be part of the analysis.
    """
    for index, trial in trial_info.iterrows():  # iterate trials
        if not trial[HAS_ET_DATA]:
            continue
        for window in time_windows:
            window_start = trial[time_windows[window][0]]
            window_end = trial[time_windows[window][1]]
            for key in dfs_to_mark:
                all_key_data = et_data_prepro[key]
                all_key_data.loc[all_key_data[T_START].between(window_start, window_end) |
                                 all_key_data[T_END].between(window_start, window_end), window] = trial[TRIAL_NUMBER]
            et_data_prepro[DF_SAMPLES].loc[(et_data_prepro[DF_SAMPLES][T_SAMPLE] <= window_end) & (et_data_prepro[DF_SAMPLES][T_SAMPLE] >= window_start), window] = trial[TRIAL_NUMBER]

    return et_data_prepro, trial_info
