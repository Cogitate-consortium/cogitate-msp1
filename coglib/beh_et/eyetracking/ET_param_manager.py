import numpy as np
import math
import AnalysisHelpers

""" Eye-Tracking Parameters and Settings

This module includes all the parameters and settings required for eye-tracking analysis of experiment 2.
It includes both lab-specific and lab-agnostic data, and all the methods required to generate a "params" object
which includes individual subject's ET session params.

It contains the following functions:
* InitParams: initializes the params for a specific session
* UpdateParams: updates parameters for a specific session based on additional calculated data
* GetStimCoords: calculates the coordinates of the stimuli locations with respect to given specifications

@authors: RonyHirsch, AbdoSharaf98
"""

# LAB - DEPENDENT INPUT :
# SCREEN W*H IN CM : taken from Lab_Equip_Setup_Summary doc:
# https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit?usp=sharing
SCREEN_SIZE = {'CA': np.array([78.7, 44.6]), 'CB': np.array([64, 36]), 'CC': np.array([69.8, 39.3]),
               'CD': np.array([58.5, 32.9]), 'CE': np.array([34.5, 19.5]), 'CF': np.array([34.5, 19.5]),
               'SG': np.array([41, 25.8]), 'SX': np.array([53, 30]), 'SZ': np.array([53.5, 29.5])}
# VIEWING DISTANCE IN CM:
# taken from : https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit#gid=0
VIEWING_DIST = {'CA': 119, 'CB': 100, 'CC': 144, 'CD': 123, 'CE': 80, 'CF': 80, 'SG': 80, 'SX': 69.5, 'SZ': 71}


# modules
FMRI = 'fMRI'
MEG = 'MEG'
ECOG = 'ECoG'
SUBJECT_LAB = 'SubjectLab'
# modality per lab : taken from https://twcf-arc.slab.com/posts/institutional-abbreviations-rsi4obcd
MODALITY = {'CA': MEG, 'CB': MEG, 'CC': FMRI, 'CD': FMRI, 'CE': ECOG, 'CF': ECOG, 'SG': ECOG, 'SX': ECOG, 'SZ': MEG}


# blink padding parameter
BLINK_PAD_MS = 200

# PARAMETERS
STIM_VA = 2.3  # EXP.2 stimulus visual angle: see https://osf.io/gm3vd and https://twcf-arc.slab.com/posts/2-exp-2-visual-angle-checks-5kungvut
# THE STIM_VA is relevant for exp.2 only; in exp.1 we only use it for conversion from degrees to pixels. See 'params['DegreesPerPix']'
STIM_DUR = 500  # stimulus duration in ms
STIM_DURATION = "StimDuration"
ANALYZED_EYE = 'R'  # 'L' or 'R' for left or right eye: if subjects have binocular data, this eye will be the arbitrary choice # 2022-09-28 DMT following CONSORTIUM DECITION [Dejan Consult]

FIX_REF_ANGLE_RADIUS = 1.5  # Fixation stability was defined using a DIAMETER of 3° of visual angle; NOTE THIS IS THE RADIUS so 3/2 = 1.5 : decision made by LM in DMT
# TODO: CHANGE FIX_REF_ANGLE_RADIUS TO MATCH EXPERIMENT 1!!!!
CENTER = 'center'
STIMULUS = 'stimulus'
LOCATION = 'Location'
FIX_RF_TYPE = [STIMULUS, CENTER]  # types of reference for fixation analysis: gaze w.r to stimulus / center

# Stimuli
FACE = 'face'
MALE = "male"
FEMALE = "female"
OBJ = 'object'
LETTER = "letter"
FF = "falseFont"
# Relevance
TR = "TaskRelevant"
TR_TARGET = f"{TR}Target"
TR_NON_TARGET = f"{TR}NonTarget"
TI = "TaskIrrelevant"
TASK_RELEVANCE_MAP = {TR_TARGET: True, TR_NON_TARGET: True, TI: False}
# trial number
TRIAL = "Trial"
# response
RESP = "Response"
ORIENTATION = "Orientation"
STIM_OFFSET = "stimOffset"
JITTER_ONSET = "jitterOnset"
EXP_ONSET = "ExpOnset"
REC_ONSET = "RecOnset"
REC_OFFSET = "RecOffset"
MINIBLOCK = "Miniblock"

#-----
# for replay:
FACE_TARGET = "Face_target"
OBJ_TARGET = "Obj_target"
FACE_NONT = "Face_non_target"
OBJ_NONT = "Obj_non_target"
STIM_TYPES = [FACE, OBJ]
FACE_GENDER_TYPES = [MALE, FEMALE]
STIM_LOCS = ['TopRight', 'TopLeft', 'BottomRight', 'BottomLeft']
GAME_WORLDS_LIST = ["World 1", "World 2", "World 3", "World 4"]
REPLAY_WORLDS_LIST = ["World A", "World B"]
FULL_WORLDS_LIST = GAME_WORLDS_LIST + REPLAY_WORLDS_LIST
STIM_PER_WORLD = 50  # how many stimuli are there in each world
FACE_PER_WORLD = 20
OBJ_PER_WORLD = 20
BLANK_PER_WORLD = 10
UNIQUE_FACE_STIM = 10  # GENDERED, there are 10 male and 10 female
UNIQUE_OBJ_STIM = 20
UNIQUE_LETTER_STIM = 20
UNIQUE_FF_STIM = 20


VIS_LIST = ["True Positive", "False Negative", "False Positive", "True Negative"]
EPOCH_START = 500  # time before stimulus onset that defines trial onset in MILLISECONDS as per https://osf.io/gm3vd
EPOCH_END = 3000  # time after stimulus onset that defines trial end in MILLISECONDS as per https://osf.io/gm3vd
FIRST_WINDOW_END = 500  # time POST STIMULUS ONSET when the first window ends in ms
SECOND_WINDOW_END = 1000  # time POST STIMULUS ONSET when the second window ends in ms (starts when first ends)
THIRD_WINDOW_END = 1500  # time POST STIMULUS ONSET when the third window ends in ms (starts when second ends)
BASELINE_START = 250 # time before stimulus onset that defines baseline

# eyelink message types
RECORDING_INFO = "RECCFG"
GAZE_COORDS = "GAZE_COORDS"

# full logs messages
SHOWING_STIMULUS = 'SHOWING_STIMULUS'
# the following mapping is based on column 9 in the full logs and its meaning according to the internal analyzer calculations
# in AnalyzerOutput (subs' BEH). It was verified using 3 full logs types: practice world log, game world log, replay world log.
STIM_LOCATION_MAP = {'BottomLeft': (0.3, 0.3), 'BottomRight': (0.7, 0.3),'TopLeft': (0.3, 0.7), 'TopRight': (0.7, 0.7)}


class Triggers:
    # a class to act as a struct object that will hold the ID information for stimuli, orientation, duration,
    # and task: https://twcf-arc.slab.com/posts/eyetracker-and-meg-coding-scheme-pf78qe5y
    def __init__(self):
        self.Stimuli = [None] * 250
        self.Trials = [None] * 148  # trial numbers
        self.Miniblocks = [None] * 200  # trial numbers
        self.StimCode = {FACE: 0, OBJ: 20, LETTER: 40, FF: 60}
        self.MsgTypes = {}

        # STIMULI
        self.Stimuli[1:10] = [FACE] * UNIQUE_FACE_STIM  # faces TODO: MALE AND FEMALE
        self.Stimuli[11:20] = [FACE] * UNIQUE_FACE_STIM  # faces TODO: MALE AND FEMALE
        self.Stimuli[21:40] = [OBJ] * UNIQUE_OBJ_STIM  # objects
        self.Stimuli[41:60] = [LETTER] * UNIQUE_LETTER_STIM  # letters
        self.Stimuli[61:80] = [FF] * UNIQUE_FF_STIM  # false fonts
        for i in range(1, 81):
            self.MsgTypes[i] = STIMULUS

        # ORIENTATION
        self.Orientation = {"101": "center", "102": "left", "103": "right"}
        for i in range(101, 104):
            self.MsgTypes[i] = ORIENTATION

        # DURATION in MILLISECONDS
        self.DurationMS = {"151": 500, "152": 1000, "153": 1500}
        for i in range(151, 154):
            self.MsgTypes[i] = STIM_DURATION

        # TASK RELEVANCE
        self.TaskRelevance = {"201": TR_TARGET, "202": TR_NON_TARGET, "203": TI}
        for i in range(201, 204):
            self.MsgTypes[i] = TR

        # TRIAL NUMBER
        self.Trials[111:148] = [(x-111 + 1) for x in range(111, 149)]  # trial number
        for i in range(111, 149):
            self.MsgTypes[i] = TRIAL

        # RESPONSE
        self.Response = "255"
        self.MsgTypes[255] = RESP

        # TIMINGS
        self.StimOffset = "96"  # stimulus off, the onset of the blank period
        self.JitterOnset = "97"  # blank period over, the onset of the jitter
        self.MsgTypes[96] = STIM_OFFSET
        self.MsgTypes[97] = JITTER_ONSET

        # EXPERIMENT MANAGEMENT
        self.ExpOnset = "86"
        self.RecOnset = "81"
        self.RecOffset = "83"
        self.MsgTypes[86] = EXP_ONSET
        self.MsgTypes[81] = REC_ONSET
        self.MsgTypes[83] = REC_OFFSET
        # miniblocks
        self.Miniblocks[161:200] = [(x - 161 + 1) for x in range(161, 201)]  # miniblock number
        for i in range(161, 201):
            self.MsgTypes[i] = MINIBLOCK

        self.LPTreset = "0"


def find_tracked_eye(ascii_file_path):
    """
    Given a random ascii file path (it doesn't matter which file it is), find the line with the following information:
    'RECCFG CR 1000 2 0 ???' and extract the tracked eye. Options for tracked eye (in ???) are:
    - L : left eye
    - R: right eye
    - LR: both
    This is the tracked eye for this subject as we assume that we do not change the tracked eye for a single subject
    within-experiment.
    :param ascii_file_path:
    :return: the tracked eye in this ascii file.
    """
    random_ascii_file = open(ascii_file_path, 'r')
    file_content = random_ascii_file.read().splitlines(True)  # split into lines
    random_ascii_file.close()
    for line in file_content:
        if RECORDING_INFO in line:
            msg = line.split(" ")  # the information has both "\t" and " "  in it but to get what we want we use " "
            tracked_eye = msg[-1].replace("\n", "")  # as the L/R info is at the end of the line
            break
    return tracked_eye


def define_analyzed_eye(eye):
    if ANALYZED_EYE in eye:  # if ANALYZED_EYE was tracked
        return ANALYZED_EYE
    else:  # for labs who did not have tracking of the ANALYZED_EYE, we have no choice
        return eye


def InitParams(subject_name, ascii_file_path, is_tobii=False):
    """
    defines and returns the parameters that will be used for analysis
    :return:
    """
    params = dict([])
    sub_lab = subject_name[:2]
    # lab dependent vars
    params['SubjectName'] = subject_name
    params['SubjectLab'] = sub_lab
    params['Modality'] = MODALITY[sub_lab]
    params['ScreenWidth'] = SCREEN_SIZE[sub_lab][0] * 10  # transformation from cm to mm
    params['ScreenHeight'] = SCREEN_SIZE[sub_lab][1] * 10  # transformation from cm to mm
    params['ViewDistance'] = VIEWING_DIST[sub_lab] * 10  # transformation from cm to mm
    params['ScreenResolution'] = None  # this will be retrieved from the ascii-file MESSAGES in UpdateParams
    if not is_tobii:
        params['TrackedEye'] = find_tracked_eye(ascii_file_path)
    else:
        params['TrackedEye'] = "LR"
    params['Eye'] = define_analyzed_eye(params['TrackedEye'])
    params['stimAng'] = STIM_VA  # THIS IS EXPERIMENT **2**'S STIMULUS SIZE IN DEGREES VISUAL ANGLE. THIS IS NOT USED IN EXP.1 other than for calculating DegreesPerPix
    params['Triggers'] = Triggers()
    return params


def UpdateParams(params, msgDF):  # fulllog_path
    """
    This function updates the parameters dictionary based on the eye tracker messages
    :param params: the parameters dict to be updated (should be the output of InitParams)
    :param eyeDFs: dataframe containing all ET messages
    :param fulllog_path: the path to one full log file where the stimulus information is
    :return: an updated parameter dictionary
    """
    # the sampling rate can be extracted from any Eyelink MSG line that contains RECORDING_INFO string
    # the structure of a line in msgDF containing RECORDING_INFO string is that in the text column its value is:
    # "<RECORDING_INFO> <tracking mode> <sample rate> <filter settings for the file data> <filter settings for the link data> <tracked eyes>"
    # (filter settings: 0 is off,1 is standard, and 2 is extra.)
    # for example: "RECCFG CR 1000 2 0 LR"

    recording_info = msgDF[msgDF["text"].str.contains(RECORDING_INFO)].iloc[0]['text'].split(" ")
    params['SamplingFrequency'] = float(recording_info[2])

    # get the screen resolution (assumed to be in the third message)  --> NOT NEEDED AS WE HAVE THAT INFO FROM THE LABS
    # get the screen resolution from the first Eyelink MSG line that contains GAZE_COORDS string
    # the structure of such a line is: "GAZE_COORDS <0 width> <0 height> <1 width> <1 height>"
    # for example: "GAZE_COORDS 0.00 0.00 1919.00 1079.00
    scMsg = msgDF[msgDF["text"].str.contains(GAZE_COORDS)].iloc[0]['text'].split(" ")
    params['ScreenResolution'] = np.array((int(float(scMsg[-2]))+1, int(float(scMsg[-1]))+1))

    # screen conversion factor (how many CM per pixel) : ScreenWidth is the actual width IN MILLIMIETERS (so /10)
    # and screen resolution[0] is the pixels in width
    params['PixelPitch'] = [(params['ScreenWidth']/10) / params['ScreenResolution'][0],
                            (params['ScreenHeight']/10) / params['ScreenResolution'][1]]

    # center location
    params['ScreenCenter'] = params['ScreenResolution'] / 2

    """
    We now define a conversion between the pixels and degrees VA.
    Here is what we know about each lab's setup:
    - SCREEN_SIZE: each lab's screen size (see https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit?usp=sharing)
    - VIEWING_DISTL: each lab's participant viewing distance (see https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit?usp=sharing)
    - screen resolution (taken from the subject's Eyelink log files; but also exist in the above table)
    - EXPERIMENT 2'S STIMULUS SIZE IN DEGREES VISUAL ANGLE: all of these setups were calibrated such that in experiment 2,
    the size of the stimuli in visual angles will be 2.3° (see pre-registration https://osf.io/gm3vd).
    Thus, this process included adjusting parameters based on calculations of HORIZONTAL visual angle, and VERTICAL
    visual angle, for each lab. see https://twcf-arc.slab.com/posts/2-exp-2-visual-angle-checks-5kungvut .

    **NOTABLY** As each lab's setup is IDENTICAL between experiments 1 and 2, we can rely on the abovementioned
    parameters to convert pixels to degrees visual angle IN BOTH EXPERIMENTS.
    Therefore, the 'stimAng' is EXPERIMENT 2'S STIMULUS SIZE IN VISUAL ANGLES,
    and the below calculated 'stimPix' is EXPERIMENT 2'S STIMULSU SIZE IN PIXELS.
    However, in exp.1, this only uses as a transition to easily calculate 'DegreesPerPix', which is true for both
    experiments.
    """

    # view_distance in cm = view_distance / 10 as 'ViewDistance' is in MILLIMETERS
    params['stimPix'] = AnalysisHelpers.deg2pix(params['ViewDistance']/10, params['stimAng'], params['PixelPitch'])
    params['DegreesPerPix'] = params['stimAng'] / params['stimPix']

    return params

