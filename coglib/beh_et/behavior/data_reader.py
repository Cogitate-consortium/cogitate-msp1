import pandas as pd
import numpy as np
import os
import re

"""
This module uploads subject data and organizes it for future checks and analyses.
The "read_data" function expects to receive a path to a directory containing subject folders in the exp.1 structure.
Each subject directory is then read (and verified) by the "read_sub_data" function, which expects to receive a path
to a subject folder that contains session folders. From session, the extracted data are the "Details" files (within
the "Details" folder) and the Analyzer output for this session - which is expected to be nested right under the session
folder. This information is then used to instantiate class instance Subject, which contains instances of class Session,
and within each session the Details file class and the analyzer output dataframe.

@author: RonyHirsch
"""

" Constants of the structure of exp.2 : assumptions on subject code names, session code names and file names:"
BEH = "BEH"
SUBLEN = 5
DETAILS = "Details"
SEQ = "Sequence"
STIM_TYPE_MAP = {"Object": "SO", "Face": "SF", 'None': 'None'}
PRESCREEN = "A"
PRESCREEN_ALT = "B"
STIMANALYZER = "AnalyzerOutput"
STIM_ANALYZER_SUFFIX = "Stimulus.csv"
FILLERD = "FillerDetails"
LOCALIZERD = "LocalizerDetails"
PROBED = "ProbeDetails"
STIMD = "StimulusDetails"
SUBD = "SubjectDetails"
# column names as they appear in the raw exp.2 output
# fillerDetails
FILLER_ONSET_TS = "fillerOnsetTS"
FILLER_ONSET_TS_NP = "fillerOnsetTS_NoPauses"
DURING_SAFE_END = "duringSafeEndOfGame"
STIM_LOC = "LOCALIZER_STIMULUS"
STIM_GAME = "GAME_STIMULUS/PROBE"
CURR_LEVEL_ID = "currentLevelID"

# localizerDetails
STIM_ONSET_TS = "stimulusOnsetTS"
STIM_ONSET_TS_NP = "stimulusOnsetTS_NoPauses"
TYPE_ = "type"
EVAL = 'responseEvaluation'
TRUENEGATIVE = 'TrueNegative'

# localizer sequence
LOC_LEVEL_TITLE = "LevelToReplayID_1Based"

# Paths and dicts for HPC
COGITATE_PATH = "/mnt/beegfs/XNAT/COGITATE"
RESOURCES = "RESOURCES"
FMRI = 'fMRI'
MR = "MR"
MEEG = 'MEEG'
MEG = 'MEG'
ECOG = "ECoG"
ECOG_BIG = "ECOG"
RAW = "Raw"
PROJ = "projects"
MODALITY_MAPPING = {FMRI: MR, MEG: MEEG, ECOG: ECOG_BIG}
LAB = "lab"

BEH_RAW_FILE = "RawDurR"
RAW_FILENUM = {FMRI: 8, MEG: 5, ECOG: 5}
ABORTED = "_ABORTED"
RESTARTED = "_RESTARTED"
SUM_DUR_SHEETS = 7

# "time" column
TIME_COL = "time"

# "event" code column
EVENT_CODE_COL = "event"
# event coding scheme, ref: https://twcf-arc.slab.com/posts/behavioral-data-coding-scheme-aa3w6d6w
STIM_TYPE_DICT = {"1": "face", "2": "object", "3": "letter", "4": "falseFont"}
STIM_ORIENT_DICT = {"1": "center", "2": "left", "3": "right"}

# "eventType" column - all event types
EVENT_TYPE_COL = "eventType"
RUN_START = "RunOnset"
BLOCK_START = "TargetScreenOnset"
STIM_START = "Stimulus"
STIM_END = "Fixation"
JITTER_START = "Jitter"
RESPONSE = "Response"
MINIBLOCK_END = "Save"
BASELINE = "Baseline"

# other columns
BLOCK_COL = "block"
MINIBLOCK_COL = "miniBlock"
MINIBLOCK_TYPE_COL = "miniBlockType"
TRIAL_COL = "trial"
DATE_COL = "date"

# new columns
STIM_CODE_COL = "stimCode"
STIM_TYPE_COL = "stimType"
STIM_ORIENT_COL = "stimOrientation"
STIM_ID_COL = "stimID"
STIM_ONSET_COL = "stimOnset"
RUN_ONSET_COL = "runOnset"
TARGET_ONSET_COL = "targetScreenOnset"
FIXATION_COL = "fixationOnset"
JITTER_COL = "jitterOnset"
RESPONSE_TYPE_COL = "responseType"
RESPONSE_TIME_COL = "responseTime"
BASELINE_COL = "baselineOnset"
TASK_RELEVANT = "isTaskRelevant"
EXPECTED_RESP = "dsrdResponse"
SUB_CODE = "subCode"
MODALITY = "modality"
PLND_STIM_DUR = "plndStimulusDur"

# column name maps:
### MRI: in MRI, the random jitter between trials was longer. In addition, we introduced 3 additional baseline periods
# of 12 seconds each between each block within each run. In addition, they have "run onset" and others don't.
fMRI_COL_NAME_MAP = {EVENT_CODE_COL: STIM_CODE_COL, TIME_COL: STIM_ONSET_COL,
                     "time_TargetScreenOnset": TARGET_ONSET_COL, "time_Fixation": FIXATION_COL,
                     "time_Jitter": JITTER_COL, "event_Response": RESPONSE_TYPE_COL,
                     "time_Response": RESPONSE_TIME_COL, "time_Baseline": BASELINE_COL, "time_RunOnset": RUN_ONSET_COL}
# other modalities
COL_NAME_MAP = {EVENT_CODE_COL: STIM_CODE_COL, TIME_COL: STIM_ONSET_COL,
                "time_TargetScreenOnset": TARGET_ONSET_COL, "time_Fixation": FIXATION_COL,
                "time_Jitter": JITTER_COL, "event_Response": RESPONSE_TYPE_COL,
                "time_Response": RESPONSE_TIME_COL}
# column order
fMRI_COL_ORDER = [SUB_CODE, MODALITY, LAB, DATE_COL, BLOCK_COL, MINIBLOCK_COL, MINIBLOCK_TYPE_COL, "targ1", "targ2",
                  TRIAL_COL, STIM_CODE_COL, TASK_RELEVANT, STIM_TYPE_COL, STIM_ORIENT_COL, STIM_ID_COL,
                  PLND_STIM_DUR, "plndJitterDur", EXPECTED_RESP, RUN_ONSET_COL, TARGET_ONSET_COL,
                  STIM_ONSET_COL, FIXATION_COL, JITTER_COL, RESPONSE_TIME_COL, RESPONSE_TYPE_COL, BASELINE_COL]

COL_ORDER = [SUB_CODE, MODALITY, LAB, DATE_COL, BLOCK_COL, MINIBLOCK_COL, MINIBLOCK_TYPE_COL, "targ1", "targ2",
                  TRIAL_COL, STIM_CODE_COL, TASK_RELEVANT, STIM_TYPE_COL, STIM_ORIENT_COL, STIM_ID_COL,
                  PLND_STIM_DUR, "plndJitterDur", EXPECTED_RESP, TARGET_ONSET_COL,
                  STIM_ONSET_COL, FIXATION_COL, JITTER_COL, RESPONSE_TIME_COL, RESPONSE_TYPE_COL]

def parse_event_code(df_row):
    """
    Extract stimulus information from the event code based on the COGITATE mapping
    See: https://twcf-arc.slab.com/posts/behavioral-data-coding-scheme-aa3w6d6w
    """
    event_code = str(int(df_row.loc[EVENT_CODE_COL]))
    stim_type = STIM_TYPE_DICT[event_code[0]]
    stim_orientation = STIM_ORIENT_DICT[event_code[1]]
    stim_id = event_code[2:]
    return pd.Series([stim_type, stim_orientation, stim_id])


def mark_task_relevance(row):
    """
    Mark for a given stimulus row whether the stimulus was task relevant (i.e., from a category subjects were
    supposed to press for) or task irrelevant (i.e., from an irrelevant category)
    """
    task = row[MINIBLOCK_TYPE_COL]
    target = row[STIM_TYPE_COL]
    if target in task:
        return True
    if target == "falseFont" and "false" in task:
        return True
    return False


class Subject:
    """
    A single subjct class, contains all of this subject's session information files and raw data.
    """
    def __init__(self, path, mod, sub_name):
        """
        :param path: the full path to the subject folder
        :param full_run_id: the name that was given to this subject's full run session (if there was one)
        :param prescreen_id: the name that was given to this subject's behavioral sreening session (if there was one)
        """
        self.path = path
        self.lab = sub_name[0:2]
        self.id = sub_name[2:]
        self.sub_name = sub_name
        self.mod = mod
        self.add_data(path)
        self.processed_data = None

    def handle_sa167(self, path):
        """
        Manually fix the behavioral data as per this subject's CRF:
        'CA167_Beh_V1_RawDurR4 contains block 7,
        CA167_Beh_V1_RawDurR4-1 contains block 8 and 9,
        CA167_Beh_V1_RawDurR5 contains block 10'
        'CA167_Beh_V1_SumDur contains block 1 - 7; CA167_Beh_V1_SumDur-1 contains block 8 - 10'
        """
        # raw data files
        data_files = [x for x in os.listdir(path) if x.endswith("csv")]
        data_files = sorted(data_files)  # sort in ascending order
        ordered_beh = list()
        for i in range(0, 3):  # DurR1-4
            data = pd.read_csv(os.path.join(path, data_files[i]))
            ordered_beh.append(data)
        aborted_data = pd.read_csv(os.path.join(path, data_files[3]))
        aborted_data = aborted_data[aborted_data[BLOCK_COL] != 8]  # block 7 was completed but 8 and up are in the next file
        ordered_beh.append(aborted_data)
        # now the files post-restart
        restarted_data = pd.read_csv(os.path.join(path, data_files[4]))
        restarted_data.loc[:, BLOCK_COL] = restarted_data[BLOCK_COL] + 7
        restarted_data.loc[:, MINIBLOCK_COL] = restarted_data[MINIBLOCK_COL] + 28
        ordered_beh.append(restarted_data)
        next_data = pd.read_csv(os.path.join(path, data_files[5]))
        next_data.loc[:, BLOCK_COL] = next_data[BLOCK_COL] + 7
        next_data.loc[:, MINIBLOCK_COL] = next_data[MINIBLOCK_COL] + 28
        ordered_beh.append(next_data)
        # put unto the df
        behavioral_data = pd.concat(ordered_beh)
        behavioral_data.reset_index(inplace=True, drop=True)
        self.raw_data = behavioral_data

        # now sumDur
        sumdur_file = [x for x in os.listdir(path) if x.endswith("xls")]
        sumdur_dict = {f"Sheet{i}": None for i in range(1, SUM_DUR_SHEETS + 1)}
        for i in range(1, SUM_DUR_SHEETS + 1):
            data_1 = pd.read_excel(os.path.join(path, sumdur_file[0]), sheet_name=f"Sheet{i}")
            data_2 = pd.read_excel(os.path.join(path, sumdur_file[1]), sheet_name=f"Sheet{i}")
            sum_data = pd.concat([data_1, data_2])
            sumdur_dict[f"Sheet{i}"] = sum_data
        self.sumdur = sumdur_dict

        return

    def add_data(self, path):
        if self.sub_name == "CA167":  # this subject had a special case, to be treated separately based on the CRF
            self.handle_sa167(path)
        else:
            # beh summary outputted from the experiment
            sumdur_file = [x for x in os.listdir(path) if x.endswith("xls")]
            if len(sumdur_file) == 0:
                self.sumdur = None
            else:
                sumdur_dict = {f"Sheet{i}": None for i in range(1, SUM_DUR_SHEETS+1)}
                for i in range(1, SUM_DUR_SHEETS+1):
                    sumdur_dict[f"Sheet{i}"] = pd.read_excel(os.path.join(path, sumdur_file[0]), sheet_name=f"Sheet{i}")
                self.sumdur = sumdur_dict

            # raw data files
            data_files = [x for x in os.listdir(path) if x.endswith("csv")]
            data_files = sorted(data_files)  # sort in ascending order
            ordered_beh = list()

            # loop over all raw file numbers
            for i in range(1, RAW_FILENUM[self.mod]+1):
                # create a sublist of all files with that run number in their name
                temp_i_list = list()
                for f in data_files:
                    if f"{BEH_RAW_FILE}{i}" in f:
                        temp_i_list.append(f)

                # to handle the case of more than one file:
                aborted_flag = 0
                restarted_flag = 0
                regular_flag = 0
                for data_file in temp_i_list:  # loop over the list of all files with the same raw file number
                    if ABORTED not in data_file and RESTARTED not in data_file:  # a normal file, supposedly
                        if bool(re.match(rf"{self.sub_name}_Beh_V1_RawDurR[1-9].csv", data_file)):  # indeed, a normal file
                            regular_flag += 1
                        else:  # not so normal, and name is manually tagged by the lab
                            raise Exception(f"WARNING: SUBJECT {self.sub_name} RAW DUR {i} : DUPLICATION IN DUR FILE NAMES")
                    elif RESTARTED in data_file:
                        print(f"SUBJECT {self.sub_name} RAW DUR {i} : 1 RESTARTED FILE FOUND")
                        restarted_flag += 1
                    else:
                        print(f"SUBJECT {self.sub_name} RAW DUR {i} : 1 ABORTED FILE FOUND")
                        aborted_flag += 1

                # the normal case - one file per raw dur run, no abort or restart
                if regular_flag == 1 and restarted_flag == 0 and aborted_flag == 0:
                    #print(f"SUBJECT {self.sub_name} RAW DUR {i} : 1 FILE FOUND")
                    data = pd.read_csv(os.path.join(path, temp_i_list[0]))
                    ordered_beh.append(data)

                # we have both a regular file and a restarted / aborted file - take the latest one according to the timestamps
                elif regular_flag == 1 and (restarted_flag != 0 or aborted_flag != 0):
                    max_ts = 0
                    max_data = None
                    max_name = None
                    for f in temp_i_list:
                        data = pd.read_csv(os.path.join(path, f))
                        if not data.empty:
                            ts = max(data.loc[:, TIME_COL].tolist())
                            if ts > max_ts:
                                max_ts = ts
                                max_data = data
                                max_name = f
                    print(f"WARNING: SUBJECT {self.sub_name} MORE THAN ONE RAW DUR {i} FILES FOUND. TAKING THE LATEST: {max_name}")
                    ordered_beh.append(max_data)  # add the LATEST RESTARTED FILE
                    if max_ts == 0:
                        raise Exception(f"ERROR: SUBJECT {self.sub_name} ONE RAW DUR {i} - ALL FILES ARE EMPTY")

                else:  # we do not have exactly one regular file
                    if regular_flag > 1:  # more than one raw dur file
                        raise Exception(f"ERROR: SUBJECT {self.sub_name} DUPLICATION IN DATA FILES FOR RAW DUR {i} - CHECK MANUALLY")
                    if regular_flag == 0:  # no regular raw dur file
                        if aborted_flag > 0 and restarted_flag > 0:  # both an aborted and a restarted file
                            restarted_files = [f for f in temp_i_list if RESTARTED in f]
                            if len(restarted_files) == 1:  # one restarted file
                                restarted_data = pd.read_csv(os.path.join(path, restarted_files[0]))
                                if restarted_data.empty:  # no regular file, empty restarted file, so the most relevant thing to add is the aborted
                                    print(f"WARNING: SUBJECT {self.sub_name} RAW DUR {i}: NO REGULAR FILE, ABORTED W/O RESTARTING")
                                    aborted_file = [f for f in temp_i_list if ABORTED in f][0]
                                    aborted_data = pd.read_csv(os.path.join(path, aborted_file))
                                    ordered_beh.append(aborted_data)
                                else:
                                    ordered_beh.append(restarted_data)
                            else:  # if there is more than one restarted file
                                if aborted_flag < restarted_flag:
                                    raise Exception(f"ERROR: SUBJECT {self.sub_name} RAW DUR {i} - CHECK FILES (ABORTED/RESTARTED MISMATCH)")
                                elif aborted_flag == restarted_flag:  # each aborted was restarted
                                    # NOTE THAT IF WE ENTERED THIS CASE WE HAVE --MORE THAN ONE RESTART WITHIN RAW DUR SESSION--
                                    max_ts = 0
                                    max_data = None
                                    max_name = None
                                    for restarted_file in restarted_files:
                                        rdata = pd.read_csv(os.path.join(path, restarted_file))
                                        r_ts = max(rdata.loc[:, TIME_COL].tolist())
                                        if r_ts > max_ts:
                                            max_ts = r_ts
                                            max_data = rdata
                                            max_name = restarted_file
                                    ordered_beh.append(max_data)  # add the LATEST RESTARTED FILE
                                    print(f"WARNING: SUBJECT {self.sub_name} MORE THAN ONE RESTARTED RAW DUR {i} FILES FOUND. TAKING THE LATEST: {max_name}")
                                    if max_ts == 0:
                                        raise Exception(f"ERROR: SUBJECT {self.sub_name} ONE RAW DUR {i} - ALL RESTARTED FILES ARE EMPTY")

                        elif aborted_flag > 0 and restarted_flag == 0:  # no regular file, an aborted file, no restarted files
                            print(f"WARNING: SUBJECT {self.sub_name} RAW DUR {i}: NO REGULAR FILE, ABORTED W/O RESTARTING")
                            aborted_file = [f for f in temp_i_list if ABORTED in f][0]
                            aborted_data = pd.read_csv(os.path.join(path, aborted_file))
                            ordered_beh.append(aborted_data)

                        elif aborted_flag == 0 and restarted_flag == 0:
                            # no regular file, no aborted / restarted files
                            print(f"WARNING: SUBJECT {self.sub_name} RAW DUR {i} - NO DATA; EXPERIMENT BEH IS INCOMPLETE")

                        elif aborted_flag == 0 and restarted_flag > 0:  # no regular file, no aborted file, but restarted file
                            raise Exception(f"ERROR: SUBJECT {self.sub_name} RAW DUR {i} - RESTARTED W/O ABORTED NOR REGULAR FILE")

            behavioral_data = pd.concat(ordered_beh)
            behavioral_data.reset_index(inplace=True, drop=True)
            self.raw_data = behavioral_data
        return

    def process_data(self):
        """
        This method processes the raw behavioral dataframe of the subject such that the resulting dataframe contains
        one row per trial, with all the important trial information. No summary stats are done, this is just reordering
        and filtering data. Returns the processed dataframe.
        """
        raw = self.raw_data
        if raw.empty:
            print(f"ERROR: Subject {self.sub_name} does not have raw data.")
            return

        """
        NOTE 2023-01-13:
        The PLND_STIM_DUR column denotes the stimulus duration group - either 0.5 / 1 / 1.5 seconds, as per the pre-reg
        (https://osf.io/gm3vd). However, in the ECoG modality, subjects CF102, CF103, and CF104 have different values
        (e.g., 0.508474576 instead of 0.5).
        To unify analyses across all participants, and correctly attribute stimuli to their duration conditions,
        we will now implement a fix for these subjects, that will delete the originally-logged stimulus duration groups
        and replace them with the pre-registered ones.
        We will do so by rounding the stimulus duration to 1 decimal digit. This manipulation will not affect
        other subjects, as this is the format the durations are already saved in.
        """
        raw.loc[:, PLND_STIM_DUR] = raw.loc[:, PLND_STIM_DUR].round(1)

        all_uniques = raw[EVENT_TYPE_COL].unique()  # all unique event types in this file
        all_stims = raw[raw[EVENT_TYPE_COL] == STIM_START]  # all stimulus onsets in this file
        # iterate over event types and re-order the raw data such that there is only 1 ROW PER TRIAL
        for value in all_uniques:
            if value == STIM_START:
                continue
            tmp_df = raw[raw[EVENT_TYPE_COL] == value]
            tmp_df = tmp_df.loc[:, [BLOCK_COL, MINIBLOCK_COL, TRIAL_COL, EVENT_TYPE_COL, EVENT_CODE_COL, TIME_COL]]
            all_stims = all_stims.merge(tmp_df, on=[BLOCK_COL, MINIBLOCK_COL, TRIAL_COL], how="left", suffixes=("", f"_{value}"), copy=False)
        # sort values by block, miniblock and trial numbers
        all_stims.sort_values([BLOCK_COL, MINIBLOCK_COL, TRIAL_COL, "time_Fixation", "time_Jitter"], inplace=True, ascending=True)
        all_stims.drop_duplicates(subset=[BLOCK_COL, MINIBLOCK_COL, TRIAL_COL], keep="last", inplace=True)
        all_stims.reset_index(drop=True, inplace=True)

        # ORGANIZE: reorder the new df, where each row = trial
        # first, parse the "event" column and extract stimulus information columns from it
        all_stims[[STIM_TYPE_COL, STIM_ORIENT_COL, STIM_ID_COL]] = all_stims.apply(lambda row: parse_event_code(row), axis=1)
        all_stims[EVENT_CODE_COL] = all_stims[EVENT_CODE_COL].astype(int)
        # rename some columns
        if self.mod == FMRI:
            all_stims.rename(fMRI_COL_NAME_MAP, axis=1, inplace=True)
        else:
            all_stims.rename(COL_NAME_MAP, axis=1, inplace=True)
        # add a column denoting whether the stimulus in this trial is task relevant or task irrelevant
        all_stims.loc[:, TASK_RELEVANT] = all_stims.apply(lambda row: mark_task_relevance(row), axis=1)
        # forward fill the run onset and target screen onset columns, as they occur once per run / miniblock
        if self.mod == FMRI:  # only fMRI have run onsets
            all_stims.loc[:, RUN_ONSET_COL].fillna(method='ffill', inplace=True)
        if TARGET_ONSET_COL not in all_stims:
            print(f"WARNING: Subject {self.sub_name} does not have {TARGET_ONSET_COL} column in their raw data. Filling with nulls")
            all_stims[TARGET_ONSET_COL] = np.nan
        all_stims.loc[:, TARGET_ONSET_COL].fillna(method='ffill', inplace=True)
        # add subject info: code and modality
        all_stims.loc[:, SUB_CODE] = self.sub_name
        all_stims.loc[:, MODALITY] = self.mod
        all_stims.loc[:, LAB] = self.lab
        # reorder columns and leave only the relevant ones
        if self.mod == FMRI:
            all_stims = all_stims[fMRI_COL_ORDER]
        else:
            all_stims = all_stims[COL_ORDER]
        self.processed_data = all_stims
        return


def read_sub_data(sub_path, mod, sub_name):
    """
    given a single subject folder, create an instance of class Subject and initialize its contents with raw subject data
    :param sub_path: the full path to the subject folder
    :return: a Subject class instance containing the subject's info
    """
    sub_files = [x for x in os.listdir(sub_path) if x.endswith("csv")]  # all the behavioral data files
    # Now, check whether the experiment is complete; does this subject has all the run files?
    valid_cntr = {i: 0 for i in range(1, RAW_FILENUM[mod]+1)}
    for i in range(1, RAW_FILENUM[mod]+1):
        dur_file_name = f"{BEH_RAW_FILE}{i}"
        for file in sub_files:
            if dur_file_name in file:
                valid_cntr[i] += 1

    # do they match (> is because it can be aborted and restarted)
    if sum(valid_cntr.values()) == 0:  # is it empty
        print(f"WARNING: {sub_name} empty session (no data)")
    for raw_filenum in valid_cntr.keys():  # does each rawdur file exist
        if valid_cntr[raw_filenum] < 1:
            print(f"WARNING: {sub_name} DID NOT COMPLETE THE EXPERIMENT")

    sub = Subject(sub_path, mod, sub_name)
    sub.process_data()  # process the loaded raw beh data (now the "processed data" field will have values
    return sub


def is_subject(s):
    """
    given a string s which is a folder name, check if it represents a subject folder or not
    :param s: folder name
    :return: whether this is a subject or not
    """
    if len(s) != SUBLEN:
        return False
    if not (s[2:].isdigit()):
        return False
    if not (s[0:2].isupper()):
        return False
    return True


def data_reader_hpc(root_folder=COGITATE_PATH):
    """
    Iterates throughout the raw data folder structure to reach eac subject across all modalities.
    For each subject, it loads its entire behavioral data by calling "read_sub_data"
    :param root_folder:
    :return: a dictionary where key=subject, value=an instance of Subject class with all the behavioral information of that subject
    """
    subjects_dict = dict()
    for mod in MODALITY_MAPPING.keys():
        print(f"***   EXTRACTING MODALITY: {mod}   ***")
        current_path = os.path.join(root_folder, mod, RAW, PROJ, f"CoG_{mod}_PhaseII")
        for sub_dir in os.listdir(current_path):
            sub_path = os.path.join(current_path, sub_dir)
            sub_name = sub_dir
            if not os.path.isdir(sub_path):
                continue
            sub_v1 = os.path.join(sub_path, f"{sub_name}_{MODALITY_MAPPING[mod]}_V1")
            if os.path.isdir(sub_v1):
                resources_dir = os.path.join(sub_v1, RESOURCES)
                if os.path.isdir(resources_dir):
                    beh_path = os.path.join(resources_dir, BEH)
                    if not os.path.isdir(beh_path):  # the directory does not exist
                        print(f"WARNING: Subject {sub_name} has no behavioral data folder in v1 session. Skipping!")
                        continue
                    subjects_dict[sub_name] = read_sub_data(beh_path, mod, sub_name)
    return subjects_dict
