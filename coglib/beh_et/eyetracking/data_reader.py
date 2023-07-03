import pandas as pd
import numpy as np
import os

"""
This module uploads subject data and organizes it for future checks and analyses.
The "read_data" function expects to receive a path to a directory containing subject folders in the exp.2 structure.
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
RAW = "Raw"
PROJ = "projects"
MODALITY_MAPPING = {FMRI: MR, MEG: MEEG}


class SessDetails:
    """
    A single session's "Details" folder content. Contains details-files that are loaded here to dataframes.
    """
    def __init__(self, path):
        self.LocalizerDetWithFillers = None
        self.LocalizerDetCalculated = None  # This is the final localizer dataframe used for analysis of replay responses : everything that is excluded was removed
        self.LocalizerDetFullCorrected = None  # This is for ET analyses purposes: this includes all the lines originally in localizer data, with corrected responses, but nothing is removed - instead it is annotated in a column
        details_files = [x for x in os.listdir(path) if x.endswith('csv')]
        for f in details_files:
            if FILLERD in f:
                self.FillerDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
                self.FillerDet.rename(columns={FILLER_ONSET_TS: STIM_ONSET_TS, FILLER_ONSET_TS_NP: STIM_ONSET_TS_NP}, inplace=True)
            elif LOCALIZERD in f:
                self.LocalizerDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")  # original localizer details
            elif PROBED in f:
                self.ProbeDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
            elif STIMD in f:
                self.StimulusDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
            elif SUBD in f:
                self.SubjectDet = pd.read_csv(os.path.join(path, f), sep=';', na_values="")
        self.update()

    def update(self):
        """
        Once all "details" files have been loaded, create a version of the localizerDetails that incorporates all the
        localizer (replay) fillers, as they appear in fillerDetails.
        :return:
        """
        additional_cols = list(self.LocalizerDet.columns.values)
        type_ind = additional_cols.index(TYPE_)
        additional_cols = [additional_cols[i] for i in range(type_ind+1, len(additional_cols))]
        self.LocalizerDet[DURING_SAFE_END] = False
        filler_det_copy = self.FillerDet.copy()
        filler_last_col = filler_det_copy[DURING_SAFE_END]
        filler_det_copy.drop(DURING_SAFE_END, axis=1, inplace=True)
        for col_name in additional_cols:
            filler_det_copy[col_name] = np.nan
        filler_det_copy[DURING_SAFE_END] = filler_last_col
        filler_det_copy = filler_det_copy[filler_det_copy[CURR_LEVEL_ID] != CURR_LEVEL_ID]  # header row for some reason
        filler_det_copy[CURR_LEVEL_ID] = filler_det_copy[CURR_LEVEL_ID].astype(int)
        loc_filler_det_copy = filler_det_copy[filler_det_copy[CURR_LEVEL_ID] >= 100]
        loc_filler_det_copy = loc_filler_det_copy[loc_filler_det_copy[TYPE_] != STIM_LOC]  # just fillers
        self.LocalizerDetWithFillers = pd.concat([self.LocalizerDet, loc_filler_det_copy])
        # remove "false" localizers which occur when the level is ending
        self.LocalizerDetWithFillers = self.LocalizerDetWithFillers[self.LocalizerDetWithFillers[DURING_SAFE_END] == False]
        self.LocalizerDetWithFillers.sort_values(by=[STIM_ONSET_TS], inplace=True)
        # All "nan"s in the responseEvaluation column are filler lines (other lines are either stimulus lines OR outsideWindowPresses lines
        # thus, they are all "correct rejections" as there is no stimulus there and no responses (for the moment)
        self.LocalizerDetWithFillers[[EVAL]] = self.LocalizerDetWithFillers[[EVAL]].fillna(value=TRUENEGATIVE)
        return


def helper_stim_name(row):
    stim_type = row["Type"]
    stim_num = row["ID"]
    if not np.isnan(stim_num):  # nan = blank stim
        stimName = STIM_TYPE_MAP[stim_type] + str(int(stim_num)).zfill(2)
    else:
        stimName = STIM_TYPE_MAP[stim_type]
    return stimName


class Session:
    """
    A single session belonging to a single subject.
    A session can be either a pre-screening session, or a full-game session.
    The class contains basic information about the session, and all the relevant data this session has.
    """
    def __init__(self, path, name, stim_analysis_path=""):
        """
        :param path: the full path to the session folder
        :param name: the session name
        :param stim_analysis_path: the path to the "StimulusAnalysis.csv" file which is an Analyzer software output.
        """
        self.path = path
        self.name = name
        self.method = self.get_method(path)
        self.replay_targets = self.get_replay_targets(path)
        self.SessDetails = SessDetails(os.path.join(path, DETAILS))
        self.StimSeq = self.get_stim_seq(path)
        if stim_analysis_path and os.path.isdir(stim_analysis_path):
            self.analyzer_output = self.get_stim_analysis(stim_analysis_path)
        else:
            self.analyzer_output = None

    def get_method(self, path):
        """
        Extract the method/module this session was recorded in : fMRI/MEEG
        :param path: path to the relevant session folder, from which we'll extract the relevant .txt file
        :return: the method
        """
        sub_info = [x for x in os.listdir(path) if x.endswith('_module.txt')]
        sub_method = open(os.path.join(path, sub_info[0]), 'r').read()
        return sub_method

    def get_replay_targets(self, path):
        """
        Extract the replay level target stimuli for this session
        :param path: path to the relevant session folder, from which we'll extract the relevant .txt file
        :return:
        """
        sub_seq = os.path.join(path, SEQ, "Localizers_Corrected.csv")
        if not os.path.exists(sub_seq):  # backward compatibility
            sub_seq = os.path.join(path, SEQ, "Localizers.csv")
        try:
            sub_targets = pd.read_csv(sub_seq, sep=';')
        except pd.errors.ParserError:  # the session was at some point interrupted and the write to this file was err
            sub_targets = pd.read_csv(sub_seq, sep=';', error_bad_lines=False)  # skip the double-header line
            sub_targets.drop_duplicates(keep='last', inplace=True)
            sub_targets.reset_index(drop=True, inplace=True)
        return sub_targets

    def get_stim_seq(self, path):
        """
        Extract from the stimSequence file of the subject all the (unique) stimulus IDs the subject was presented with.
        This will be used for analyses that look at the data per stimulusID.
        :param path: path to the relevant session folder, from which we'll extract the sequence
        :return:
        """
        sub_seq_stim = os.path.join(path, SEQ, "StimSequence.csv")
        sub_stim = pd.read_csv(sub_seq_stim, sep=';')
        sub_stim["stimulusName"] = sub_stim.apply(lambda row: helper_stim_name(row), axis=1)
        return sub_stim

    def get_stim_analysis(self, path):
        stim_analysis_files = [f for f in os.listdir(path) if f.endswith(STIM_ANALYZER_SUFFIX)]
        stim_analysis_csvs = [pd.read_csv(os.path.join(path, f), sep=';') for f in stim_analysis_files]
        stim_analysis_csv = pd.concat(stim_analysis_csvs)
        return stim_analysis_csv


class Subject:
    """
    A single subjct class, contains all of this subject's session information files and raw data.
    """
    def __init__(self, path, full_run_id="", prescreen_id=""):
        """
        :param path: the full path to the subject folder
        :param full_run_id: the name that was given to this subject's full run session (if there was one)
        :param prescreen_id: the name that was given to this subject's behavioral sreening session (if there was one)
        """
        self.path = path
        self.lab = self.sub_code(path)[0:2]
        self.id = self.sub_code(path)[2:]
        self.add_session(full_run_id, prescreen_id)

    def add_session(self, full_run_id="", prescreen_id=""):
        if prescreen_id:
            self.prescreen = Session(path=os.path.join(self.path, prescreen_id), name=prescreen_id,
                                     stim_analysis_path=os.path.join(self.path, prescreen_id, STIMANALYZER))
        if full_run_id:
            self.full = Session(path=os.path.join(self.path, full_run_id), name=full_run_id,
                                stim_analysis_path=os.path.join(self.path, full_run_id, STIMANALYZER))

    def sub_code(self, path):
        """
        extracts the subject code from the full path
        :param path: the full path to the subject folder
        :return: the full subject code (lab id + sub id)
        """
        npath = os.path.normpath(path)
        split = npath.split(os.sep)
        return split[-1]
