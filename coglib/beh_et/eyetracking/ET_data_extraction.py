import fnmatch
import os
import pickle
import datetime
import math
import numpy as np
import pandas as pd
from based_noise_blinks_detection import based_noise_blinks_detection
from scipy import stats
import DataParser
import ET_param_manager

""" Subject Data Extraction Module

This module manages the preparation of raw eye-tracking data of experiment 2 FOR A SINGLE SUBJECT AT A TIME!!
The main function here is called "extract_data", and it manages all the stages in the process of QC:
- data extraction
- data parsing
- data saving 

The "extract_data" management function creates a folder for each subject under "save_path", in which all the 
resulting plots and data tables of the subject are saved (e.g., "save_path/..."). But the only thing saved at the
extraction stage for each subject is its pickle file - containing all the extracted eyetracking information. 
NOTE: No analysis is performed at the extraction stage!
In the result folder, you'll find one "_EyeTrackingData.pickle" file per each subject who had eye tracking data in the
form of ascii files. 

@authors: RonyHirsch, AbdoSharaf98
"""

EYES = ['L', 'R']

SESS_INFO_FILE = "_SessionInfo.txt"

INVALID_LIST = ["SE118"]  # This is a list of subjects whose CRF make it obvious that their entire ET session is crap and should not be analyzed

# data columns
ONSET = 'StimOnset'
# duration windows
STIM_DUR = 'StimDuration'
PRE_STIM_DUR = 'PreStim'
EPOCH = 'Epoch'
LOC = 'Location'
COORDS = 'StimulusCoords'
BEH_LOC = "LocalizerDetCorrected"
MEDIAN = "Median"
BL_CORRECTED = "_BLcorrected"

"""
The following features are based on Engbert, R., & Mergenthaler, K. (2006) Microsaccades are triggered by low retinal 
image slip. Proceedings of the National Academy of Sciences of the United States of America, 103: 7192-7197.
"""
THRESHOLD = "threshold"
OVERLAP = 'msOverlap'
SACC_FEATURES = {THRESHOLD: 1,  # upper cutoff for AMPLITUDE (in degrees) THIS IS FOR DECIDING IF A SACCADE IS MICROSACCADE!
                OVERLAP: 2,  # number of overlapping points to count as a BINOCULAR saccade
                'vfac': 5,  # will be multiplied by E&K criterion to get velocity threshold: Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. Vision research, 43(9), 1035-1045.
                'mindur': 5,  # minimum duration of a microsaccade (in indices or samples)
                }

#-----
DURR = "DurR"


def calculate_baseline(et_data_prepro, params, is_tobii=False):
    center_coords = params['ScreenCenter']
    eye = params["Eye"]

    baseline = et_data_prepro[DataParser.DF_SAMPLES].groupby(DataParser.BASELINE_START).mean()
    baseline = baseline.iloc[1:, :]

    if not is_tobii:
        baseline[f"{eye}X"] = baseline[f"{eye}X"] - center_coords[0]
        baseline[f"{eye}Y"] = baseline[f"{eye}Y"] - center_coords[1]

        et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X"] - et_data_prepro[DataParser.DF_SAMPLES][DataParser.TRIAL].map(baseline[f"{eye}X"])
        et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y"] - et_data_prepro[DataParser.DF_SAMPLES][DataParser.TRIAL].map(baseline[f"{eye}Y"])
    else:
        baseline[f"{eye}X_p"] = baseline[f"{eye}X_p"] - center_coords[0]
        baseline[f"{eye}Y_p"] = baseline[f"{eye}Y_p"] - center_coords[1]

        et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X_p{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X_p"] - et_data_prepro[DataParser.DF_SAMPLES][DataParser.TRIAL].map(baseline[f"{eye}X_p"])
        et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y_p{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y_p"] - et_data_prepro[DataParser.DF_SAMPLES][DataParser.TRIAL].map(baseline[f"{eye}Y_p"])

    return et_data_prepro


def dist_from_target(et_data_prepro, params):
    """
    Add columns to et_data_prepro[DF_SAMPLES] depicting the gaze's distance from targets:
    - center area (fixation)
    - stimulus (target stimulus of this trial)
    - add a column for "on fixation area" denoting whether the gaze's distance from the fixation is within the radius
    that was defined as "subject fixating" or not.

    :param trial_info: a dataframe containing a line per each trial (=probed stimulus) in the video game
    :param et_data_prepro: dictionary from extract_data, with the 4 ET dataframes: DF_FIX, DF_SACC, DF_SAMPLE, DF_BLINK
    each dataframe contains samples.
    :param params: subject's params so we'll have information about which eye is tracked
    :return:
    """

    # area used to determine fixation bounds for center
    ref_angle = ET_param_manager.FIX_REF_ANGLE_RADIUS  # the RADIUS of ° visual angle which was defined as fixation stability
    fix_area = ref_angle / params['DegreesPerPix']  # the area of fixation stability IN PIXELS (!)
    center_coords = params['ScreenCenter']
    eye = params["Eye"]

    # get the stimulus locations and coordinates for all trials

    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}CenterDistPixels"] = np.sqrt(((et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X"] - center_coords[0]) ** 2) +
                                                                                     ((et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y"] - center_coords[1]) ** 2))
    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}CenterDistDegs"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}CenterDistPixels"] * params['DegreesPerPix']  # convert to degrees
    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}CenterDistPixels{BL_CORRECTED}"] = np.sqrt(((et_data_prepro[DataParser.DF_SAMPLES][f"{eye}X{BL_CORRECTED}"] - center_coords[0]) ** 2) +
                                                                                     ((et_data_prepro[DataParser.DF_SAMPLES][f"{eye}Y{BL_CORRECTED}"] - center_coords[1]) ** 2))
    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}CenterDistDegs{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}CenterDistPixels{BL_CORRECTED}"] * params['DegreesPerPix']  # convert to degrees

    # We are also adding distances from CENTER separate to X coordinate and to Y coordinate (with direction)
    for coordi in ['X', 'Y']:
        center_coord = 0 if coordi == 'X' else 1
        et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistPixelsSigned"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}{coordi}"] - center_coords[center_coord]  # in pixels
        et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistDegsSigned"] = et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistPixelsSigned"] * params['DegreesPerPix']  # in degrees

        et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistPixelsSigned{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}{coordi}{BL_CORRECTED}"] - center_coords[center_coord]  # in pixels
        et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistDegsSigned{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}{coordi}CenterDistPixelsSigned{BL_CORRECTED}"] * params['DegreesPerPix']  # in degrees

    # test whether this distance is within the fixation area
    # is the subject fixating? (within the pre-defined distance from center that is counted as fixation)
    # fixation area in pixels, distance in pixels
    # NOTE: we assume fixArea is defined by RADIUS already so we DON'T divide by 2. If this size is DIAMETER then this needs to be divided by 2
    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}IsInFixationArea"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}CenterDistPixels"] <= fix_area
    et_data_prepro[DataParser.DF_SAMPLES].loc[:, f"{eye}IsInFixationArea{BL_CORRECTED}"] = et_data_prepro[DataParser.DF_SAMPLES][f"{eye}CenterDistPixels{BL_CORRECTED}"] <= fix_area

    return et_data_prepro


def blink_detection(et_data, params):
    """
    Identify real blinks from all samples.
    NOTE :: After running this, we might get a sample that's both a non-blink (e.g., fixation/saccade) AND a "Hershman"
    blink. Hershman blinks are considered the ground truth. Thus, any blink that is not Hershman is removed from dfBlink.
    :param et_data:
    :return:
    """
    et_samples = et_data[DataParser.DF_SAMPLES]
    sampling_freq = params["SamplingFrequency"]

    et_blinks = et_data[DataParser.DF_BLINK]
    et_blinks[DataParser.HERSHMAN] = False

    eye = params[DataParser.EYE]
    et_blinks = et_blinks[et_blinks[DataParser.EYE.lower()] == eye]  # get rid of the non-analyzed eye

    et_samples[f"{eye}{DataParser.HERSHMAN}"] = np.nan
    col = f"{eye}Pupil"
    convert_for_hershman = np.array(et_samples[col], dtype="float32")  # as per their OSF example: https://osf.io/gjt8v/
    blink_data = based_noise_blinks_detection(pupil_size=convert_for_hershman, sampling_freq=sampling_freq)

    # mark samples that are blinks according to the Hershman method
    num_of_blink_samples = blink_data["blink_onset"].shape[0] if isinstance(blink_data['blink_onset'], np.ndarray) else 0
    """
    We also start a blink_counter: each *valid* blink (Hershman) is numbered from the first to last in the entire run. 
    This is used for later preparation of the pre-processed data to analysis. 
    
    NOTE: the "blink_onset" and "blink_offset" returned from based_noise_blinks_detection can be either "real time"
    (a conversion made in the original implementation) or indices (as the author suggested in a comment in line 161, 
    see https://osf.io/qh5sg). ==> I CHOSE FOR THE ALGORITHM TO RETURN **INDICES**, and not try to convert them to time.
    """
    blink_counter = 1
    if num_of_blink_samples > 0:
        # add a column with the duration in MILLISECONDS of each blink (divide in freq for SECONDS, *1000 for MILLISECONDS)
        blink_data['duration'] = 1000 * (blink_data['blink_offset'] - blink_data['blink_onset']) / sampling_freq
        blink_data['tStart'] = list(et_samples.loc[blink_data['blink_onset'], "tSample"])
        blink_data['tEnd'] = list(et_samples.loc[blink_data['blink_offset'], "tSample"])
        blink_data['eye'] = [eye] * len(blink_data['tEnd'])
        blink_data[DataParser.HERSHMAN] = [True] * len(blink_data['tEnd'])
        blink_df = pd.DataFrame.from_dict(blink_data)[['eye', 'tStart', 'tEnd', 'duration', DataParser.HERSHMAN]]
        et_blinks = pd.concat([et_blinks, blink_df])
        for i in range(num_of_blink_samples):
            blink_start = int(blink_data["blink_onset"][i])
            blink_end = int(blink_data["blink_offset"][i])
            et_samples.loc[blink_start: blink_end, f"{eye}{DataParser.HERSHMAN}"] = 1  # the range here is up to and including, i.e., 1:3 is 1, 2, 3
            et_samples.loc[blink_start: blink_end, "blink_number"] = blink_counter
            blink_counter += 1

    """
    throwing away everything that is not Hershman from dfBlinks. In dfSamples, Hershman is a column. 
    """
    et_blinks = et_blinks[et_blinks[f"{DataParser.HERSHMAN}"] == 1]
    et_data[DataParser.DF_BLINK] = et_blinks
    return et_data


def pad_blinks(et_data, params):
    """
    Once Hershman blinks were identified (in blink_detection), this methods pads them with ET_param_manager.BLINK_PAD_MS
    ms BEFORE AND AFTER each blink. The resulting column in dfSamples (DataParser.HERSHMAN_PAD) will be used for
    preprocessing saccades, fixations and pupil data.

    Padding the blinks to create "blink intervals" that are then dropped from analyses of saccades / fixations is
    considered good practice, and done for example in this work:
    Denison, R. N., Yuval-Greenberg, S., & Carrasco, M. (2019).
    Directing voluntary temporal attention increases fixational stability. Journal of Neuroscience, 39(2), 353-363.
    https://doi.org/10.1523/JNEUROSCI.1926-18.2018

    :param et_data:
    :param params:
    :return:  The same dictionary, with updated dfSamples (new column)
    """
    eye = params[DataParser.EYE]
    et_samples = et_data[DataParser.DF_SAMPLES]
    et_samples.loc[:, f"{eye}{DataParser.HERSHMAN_PAD}"] = et_samples.loc[:, f"{eye}{DataParser.HERSHMAN}"]

    bls = et_data[DataParser.DF_BLINK]
    bls = bls[bls[DataParser.HERSHMAN] == True]
    for index, blink in bls.iterrows():
        # Labeling an SSACC...ESACC pair with one or more SBLINK events between them as BLINKS
        et_samples.loc[(et_samples[DataParser.T_SAMPLE] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS) &
                       (et_samples[DataParser.T_SAMPLE] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS), f"{eye}{DataParser.HERSHMAN_PAD}"] = 1
    return et_data


def extract_microsaccades(eye_gaze, velocity, speed, params):
    """

    :param eye_gaze: gaze coordinates data ** FROM dfSamples!!! ** after they were converted with respect to the center
    of the screen
    :param velocity:
    :param speed:
    :param params:
    :return:
    result_df: a dataframe containing a row per each (micro/)saccade identified by the algorithm used in:
    Engbert, R., & Mergenthaler, K. (2006). Microsaccades are triggered by low retinal image slip. Proceedings of the National Academy of Sciences, 103(18), 7192-7197.
    Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. Vision research, 43(9), 1035-1045.

    THE "START" AND "END" OF SACCADE IN THIS DATAFRAME ARE **INDICES** OF **SAMPLES** IN dfSamples in which there is a
    saccade

    radius: the threshold that was calculated to derive that a pupil movement is a saccade. This is a dynamic threshold
    depending on the data given in eye gaze.
    msdx, msdy, std_dev, med_dev
    """
    # Get the relative velocity threshold (exp.1's GetVelocityThreshold)
    for i in range(2):  # For each axis
        """
        STEP 2 in the algorithm https://www.sciencedirect.com/science/article/pii/S0042698903000841
        """
        msd = np.sqrt(np.nanmedian(np.power(velocity[:, i], 2)) - np.power(np.nanmedian(velocity[:, i]), 2))
        if msd < np.finfo('float').tiny:  # if less than the smallest usable float
            # switch to a MEAN estimator instead and see
            msd = np.sqrt(np.nanmean(np.power(velocity[:, i], 2)) - np.power(np.nanmean(velocity[:, i]), 2))
            # raise an error if still smaller
            if msd < np.finfo('float').tiny:
                raise Exception('Calculated velocity threshold (msdx) was smaller than the smallest '
                                'positive representable floating-point number. Did you exclude blinks/'
                                'missing data before saccade detection?')
        if i == 0:
            msdx = msd
        else:
            msdy = msd

    # compute the standard deviation and the median abs deviation for the velocity values in both components
    std_dev = np.nanstd(velocity, axis=0, ddof=1)
    med_dev = stats.median_abs_deviation(velocity, axis=0, nan_policy='omit')

    # BEGIN saccade detection by the E&K algorithm define an ellipsoid : THIS IS STEP 3 in the algorithm
    radiusx = SACC_FEATURES['vfac'] * msdx
    radiusy = SACC_FEATURES['vfac'] * msdy
    radius = np.array([radiusx, radiusy])

    # compute test criterion: ellipse equation
    test = np.power((velocity[:, 0] / radiusx), 2) + np.power((velocity[:, 1] / radiusy), 2)  # this is the criterion
    indices = np.argwhere(test > 1)  # indices where test is above threshold

    # determine saccades
    n_sacs = 0
    dur = 1
    saccade_start = 0
    saccade_starts = []  # a list of indices in dfSamples dataframe during which a saccade started
    saccade_ends = []  # a list of indices in dfSamples dataframe during which a saccade ended
    durations = []
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] == 1:
            dur = dur + 1
        else:
            if dur >= SACC_FEATURES['mindur']:
                n_sacs = n_sacs + 1
                saccade_end = i
                saccade_starts.append(int(indices[saccade_start]))  # this is inclusive!
                saccade_ends.append(int(indices[saccade_end]))  # this is inclusive!
                durations.append(int(indices[saccade_end]) - int(indices[saccade_start]))

            saccade_start = i + 1
            dur = 1

    # check duration criterion for the last microsaccade
    if dur >= SACC_FEATURES['mindur']:
        n_sacs = n_sacs + 1
        saccade_end = i
        saccade_starts.append(int(indices[saccade_start]))
        saccade_ends.append(int(indices[saccade_end]))
        durations.append(int(indices[saccade_end]) - int(indices[saccade_start]))

    result = {DataParser.T_START: saccade_starts, DataParser.T_END: saccade_ends, "duration": durations, DataParser.VPEAK: list(), "saccade_x": list(), "saccade_y": list(), "sacc_dist_dva": list(), "amplitude_x": list(),
              "amplitude_y": list(), DataParser.AMP_DEG: list(), "saccade_dist_to_center": list(), DataParser.SACC_DIRECTION_RAD: list(), "sacc_direction_deg": list()}

    for i in range(n_sacs):
        """
        For each identified saccade, give information about its peak velocity and amplitude
        """
        saccade_start = int(saccade_starts[i])
        saccade_end = int(saccade_ends[i])
        # Find maximal speed between these indices
        result[DataParser.VPEAK].append(float(max(speed[saccade_start:saccade_end])))
        result["saccade_x"].append(eye_gaze[saccade_end, 0] - eye_gaze[saccade_start, 0]) # saccade_x is the X and the end of the saccade - the X at the beginning of the sacc
        result["saccade_y"].append(eye_gaze[saccade_end, 1] - eye_gaze[saccade_start, 1]) # saccade_y is the Y and the end of the saccade - the Y at the beginning of the sacc
        result["sacc_dist_dva"].append(np.sqrt((eye_gaze[saccade_end, 0] - eye_gaze[saccade_start, 0])**2 +
                                                  (eye_gaze[saccade_end, 1] - eye_gaze[saccade_start, 1])**2))  # saccade_y is the Y and the end of the saccade - the Y at the beginning of the sacc

        # amplitude (dX,dY)
        min_ind_x = np.argmin(eye_gaze[saccade_start:saccade_end, 0]) + saccade_start
        max_ind_x = np.argmax(eye_gaze[saccade_start:saccade_end, 0]) + saccade_start
        min_ind_y = np.argmin(eye_gaze[saccade_start:saccade_end, 1]) + saccade_start
        max_ind_y = np.argmax(eye_gaze[saccade_start:saccade_end, 1]) + saccade_start

        min_x = eye_gaze[min_ind_x, 0]
        max_x = eye_gaze[max_ind_x, 0]
        min_y = eye_gaze[min_ind_y, 1]
        max_y = eye_gaze[max_ind_y, 1]

        amp_x = np.sign(max_ind_x - min_ind_x) * (max_x - min_x)
        amp_y = np.sign(max_ind_y - min_ind_y) * (max_y - min_y)
        result["amplitude_x"].append(amp_x)
        result["amplitude_y"].append(amp_y)
        result[DataParser.AMP_DEG].append(np.sqrt(amp_x ** 2 + amp_y ** 2))  # amplitude total

        # saccade distance to fixation (screen center)
        gaze_onset = eye_gaze[saccade_start, :]
        gaze_offset = eye_gaze[saccade_end, :]
        dist_to_fix_onset = np.sqrt((gaze_onset[0] - params['ScreenCenter'][0] * params['DegreesPerPix']) ** 2 +
                                    (gaze_onset[1] - params['ScreenCenter'][1] * params['DegreesPerPix']) ** 2)
        dist_to_fix_offset = np.sqrt((gaze_offset[0] - params['ScreenCenter'][0] * params['DegreesPerPix']) ** 2 +
                                     (gaze_offset[1] - params['ScreenCenter'][1] * params['DegreesPerPix']) ** 2)
        distToFix = (dist_to_fix_offset - dist_to_fix_onset)
        result["saccade_dist_to_center"].append(distToFix)

        # Same calculation as sacc_direction
        """
        there is a (-) before the y parameter of the atan2 because, as said before, in Eyelink - when Y goes down -> 
        that means that the gaze went UP. See http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
        So, we flip this **now** so that the direction will be correct (otherwise, it'll be upside down)
        """
        saccade_direction_rad = math.atan2(-(eye_gaze[saccade_end, 1] - eye_gaze[saccade_start, 1]), eye_gaze[saccade_end, 0] - eye_gaze[saccade_start, 0])
        saccade_direction_deg = saccade_direction_rad * 180 / math.pi
        result[DataParser.SACC_DIRECTION_RAD].append(saccade_direction_rad)
        result["sacc_direction_deg"].append(saccade_direction_deg)
        """
        DO -NOT-(!!!) PERFORM THE CONVERSION TO POSITIVE DEGRESS BETWEEN [0, 360] AT THIS LEVEL. (commented below)
        IF YOU DO -> THEN THE RADIANS AND DEGRESS WILL NOT REPRESENT THE SAME ANGLE
        EXAMPLE: LET'S CONVERT -90 (IN RADS: -1.57) TO 270, AND FOR A DIFFERENT SAMPLE WE'LL HAVE A VALUE OF 90 DEGS.
        The RADIAN average of [-90, 90] is 0. The ANGLE average of [270, 90] is 180, which is 3.14 in RADIANS!!!
        """

    result_df = pd.DataFrame(result)
    return result_df, radius, msdx, msdy, std_dev, med_dev


def saccade_detection(et_data_dict, params):
    """
    For saccade and microsaccade calculation, we need to convert coordinates to degrees visual angle (VA). 
    To convert (X, Y) locations (in PIXELS on the screen) to degrees VA, we will: 
    1. Convert pixels to be relative to the screen center (taking center as (0, 0))
    2. Convert CM to degrees VA (according to the viewing distance) from the center of fixation 
    (assumed to be screen center) based on the relations between degrees and pixels, calculated in params
    
    *** NOTE *** : 
    Gaze coordinates (X, Y) in Eyelink are such that (0, 0) is the TOP LEFT corner of the screen!!!
    This means that when gaze goes DOWN --> Y coordinate goes UP! 
    Source: EL1000 User manual 1.5 chapter 4.4.2.3 GAZE
    http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    
    *** NOTE *** : 
    In fMRI subjects the eye tracking was MONOCULAR, yet, the algorithm for microsaccade detection described in the 
    2006 paper above is to track ONLY **binocular microsaccades**, which they defined as microsaccades detected 
    in both eyes with a temporal overlap of at least one data sample. 
    For clusters consisting of microsaccades with multiple overlapping relations, 
    they selected the largest microsaccades from both eyes within the cluster.
    """
    samples = et_data_dict[DataParser.DF_SAMPLES]
    # convert pixels to relative
    gaze_coords = samples.loc[:, ['LX', 'LY', 'RX', 'RY']]  # dfSamples
    gaze_coords.loc[:, ["LX", "RX"]] = gaze_coords[["LX", "RX"]] - params['ScreenCenter'][0]
    gaze_coords.loc[:, ["LY", "RY"]] = gaze_coords[["LY", "RY"]] - params['ScreenCenter'][1]
    # convert to degrees VA
    gaze_coords.loc[:, :] = gaze_coords * params['DegreesPerPix']

    EK_sacc_dict = {eye: None for eye in EYES}
    for eye in EYES:
        eye_gaze = np.array(gaze_coords[[f"{eye}X", f"{eye}Y"]])
        """ 
        Calculate velocity according to the paper below, with λ = 5 as they suggest 
        Engbert, R., & Mergenthaler, K. (2006). Microsaccades are triggered by low retinal image slip. 
        Proceedings of the National Academy of Sciences, 103(18), 7192-7197.
        https://www.pnas.org/doi/full/10.1073/pnas.0509557103#F1
        
        The velocity calculation itself is an EXACT replica of equation (1) in: 
        Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. 
        Vision research, 43(9), 1035-1045. https://www.sciencedirect.com/science/article/pii/S0042698903000841
        Which represents a moving average of velocities over 5 data **samples** to suppress noise
        """
        # initialize
        velocity = np.zeros(eye_gaze.shape)
        speed = np.zeros((velocity.shape[0], 1))
        # STEP 1: loop through the data points and calculate a moving average of velocities over λ = 5 data samples

        for n in range(2, eye_gaze.shape[0] - 2):  # as per https://www.sciencedirect.com/science/article/pii/S0042698903000841
            velocity[n, :] = (eye_gaze[n + 1, :] + eye_gaze[n + 2, :] - eye_gaze[n - 1, :] - eye_gaze[n - 2, :]) * (params["SamplingFrequency"] / 6)

        # speed: used in the microsaccade detection method: extract_microsaccades
        speed[:, 0] = np.sqrt(np.power(velocity[:, 0], 2) + np.power(velocity[:, 1], 2))

        velocity[velocity == 0] = np.nan  # turn zeros to nans
        # STEPS 2 - 3
        sacc_df_eye, radius, msdx, msdy, std_dev, med_dev = extract_microsaccades(eye_gaze, velocity, speed, params)

        if sacc_df_eye is not None:  # mark microsaccades
            sacc_df_eye.loc[:, "microsaccade"] = sacc_df_eye[DataParser.AMP_DEG] < SACC_FEATURES[THRESHOLD]
            sacc_df_eye.reset_index(drop=True, inplace=True)
            EK_sacc_dict[eye] = sacc_df_eye

        else:
            EK_sacc_dict[eye] = None

    """
    IDEA: identify all the saccades that their L and R intervals overlap for a duration that is 
    >= SACC_FEATURES[OVERLAP] (E&K threshold for a binocular microsaccade). Then, select these saccades based on the L
    eye (ARBITRARY!) and output the binocular saccade information. Meaning, this dataframe will include L eye saccades
    that overlapped with the right eye for at least SACC_FEATURES[OVERLAP]. 
    """
    if params[DataParser.EYE] == 'LR':
        print(f"Only monocular data is considered: 2022-09-12 consortium decision")
    else:  # no binocular data
        EK_sacc_dict['LR'] = None

    EK_sacc_df = pd.DataFrame()
    for EK_sacc_key in EK_sacc_dict.keys():
        if EK_sacc_dict[EK_sacc_key] is None:
            continue
        rel_df = EK_sacc_dict[EK_sacc_key]
        starts = list(rel_df[DataParser.T_START])
        start_times = [samples.loc[i, DataParser.T_SAMPLE] for i in starts]
        ends = list(rel_df[DataParser.T_END])
        end_times = [samples.loc[i, DataParser.T_SAMPLE] for i in ends]
        EK_sacc_dict[EK_sacc_key][DataParser.T_START] = start_times
        EK_sacc_dict[EK_sacc_key][DataParser.T_END] = end_times
        EK_sacc_dict[EK_sacc_key][DataParser.EYE] = EK_sacc_key
        EK_sacc_df = pd.concat([EK_sacc_df, EK_sacc_dict[EK_sacc_key]])

    et_data_dict[DataParser.DF_SACC] = EK_sacc_df

    return et_data_dict


def mark_saccades(et_data_dict, params):
    """
    This marks EK saccs in dfSamples and in dfSacc. dfSacc is a df containing ONLY saccades recognized by the EK algorithm.
    Any saccade that overlaps with the PADDED HERSHMAN BLINK PERIOD is marked in both dfSacc and dfSamples.
    :param et_data_dict:
    :param params:
    :return:
    """
    ek_saccs = et_data_dict[DataParser.DF_SACC]
    samps = et_data_dict[DataParser.DF_SAMPLES]
    eye = params[DataParser.EYE]

    if ek_saccs.empty:
        print("WARNING: no saccades found!")
        samps.loc[:, f"{DataParser.EK}Sacc"] = False
        samps.loc[:, f"{DataParser.AMP_DEG}"] = np.nan
        samps.loc[:, f"microsaccade"] = False
        samps.loc[:, DataParser.REAL_SACC] = False
        samps.loc[:, f"sacc_number"] = np.nan
        return et_data_dict

    ek_saccs.loc[:, f"{DataParser.HERSHMAN_PAD}"] = False
    for index, blink in et_data_dict[DataParser.DF_BLINK].iterrows():
        """
        A saccade is marked as not real (=overlapping with a padded blink period) ONLY IF STARTED during a blink period, 
        OR ENDED during a blink period. 
        The other way around (i.e., blink period starting within a saccade) is not realistic, as saccade last ~50ms
        while the padded blink periods extend for longer than 2 * ET_param_manager.BLINK_PAD_MS . 
        Thus, we do not account for a case where a blink PERIOD is completely submerged within a saccade (i.e., 
        starting after a saccade started AND ending before it ended).
        """
        ek_saccs.loc[(((ek_saccs[DataParser.T_START] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS) & (ek_saccs[DataParser.T_START] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS)) |
                      ((ek_saccs[DataParser.T_END] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS) & (ek_saccs[DataParser.T_END] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS))) & (
                      ek_saccs[DataParser.EYE] == blink['eye']), f"{DataParser.HERSHMAN_PAD}"] = True

    print(f"Mark Saccades {datetime.datetime.now()}")
    ek_saccs = ek_saccs[ek_saccs[DataParser.EYE] == eye]  # get rid of the non-analyzed eye
    samps.loc[:, DataParser.REAL_SACC] = False
    ek_sacc_no_blink = ek_saccs.loc[ek_saccs[f"{DataParser.HERSHMAN_PAD}"] == False, :]
    """
        We also start a saccade_counter: each *valid* saccade (EK + not in blink pad) is numbered from the first to last in the entire run. 
        This is used for later preparation of the pre-processed data to analysis. 
    """
    sacc_counter = 1
    for sacc in ek_saccs.itertuples():
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"{DataParser.EK}Sacc"] = True
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"{DataParser.AMP_DEG}"] = sacc.ampDeg  # amplitude
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"microsaccade"] = sacc.microsaccade  # is microsaccade (EK algorithm)

    for sacc in ek_sacc_no_blink.itertuples():
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), DataParser.REAL_SACC] = True
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"{DataParser.AMP_DEG}"] = sacc.ampDeg  # amplitude
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"microsaccade"] = sacc.microsaccade  # is microsaccade (EK algorithm)
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart, sacc.tEnd), f"sacc_number"] = sacc_counter
        sacc_counter += 1

    et_data_dict[DataParser.DF_SACC] = ek_saccs
    return et_data_dict


def mark_fixations(et_data_dict, params):
    """
    Similarly to saccades, we consider real fixations as ones that do not overlap with a blink PADDED PERIOD.
    :param et_data_dict:
    :param params:
    :return:
    """
    fixations = et_data_dict[DataParser.DF_FIXAT]
    samps = et_data_dict[DataParser.DF_SAMPLES]
    eye = params[DataParser.EYE]

    """
    A fixation is marked as not real if it is overlapping with a padded blink period)
    """
    fixations.loc[:, f"{DataParser.HERSHMAN_PAD}"] = False
    for index, blink in et_data_dict[DataParser.DF_BLINK].iterrows():
        """
        ONE WAY: a fixation is nullified if it started during a blink period, OR ENDED during a blink period. 
        """
        fixations.loc[(((fixations[DataParser.T_START] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS) & (fixations[DataParser.T_START] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS)) |
                      ((fixations[DataParser.T_END] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS) & (fixations[DataParser.T_END] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS))) & (
                      fixations[DataParser.EYE.lower()] == blink['eye']), f"{DataParser.HERSHMAN_PAD}"] = True

    for index, blink in et_data_dict[DataParser.DF_BLINK].iterrows():
        """
        SECOND WAY: a fixation is nullfied if it contains a whole blink period inside it (i.e., a blink PADDED PERIOD
        both started AND ENDED during the fixation)
        """
        fixations.loc[(((fixations[DataParser.T_START] <= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS) & (fixations[DataParser.T_END] >= blink[DataParser.T_START] - ET_param_manager.BLINK_PAD_MS)) |
                      ((fixations[DataParser.T_START] <= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS) & (fixations[DataParser.T_END] >= blink[DataParser.T_END] + ET_param_manager.BLINK_PAD_MS))) & (
                      fixations[DataParser.EYE.lower()] == blink['eye']), f"{DataParser.HERSHMAN_PAD}"] = True

    print(f"Mark fixations {datetime.datetime.now()}")
    fixations = fixations[fixations[DataParser.EYE.lower()] == eye]  # get rid of the non-analyzed eye
    samps.loc[:, DataParser.REAL_FIX] = False
    fix_no_blink = fixations.loc[fixations[f"{DataParser.HERSHMAN_PAD}"] == False, :]

    for fix in fix_no_blink.itertuples():
        samps.loc[samps[DataParser.T_SAMPLE].between(fix.tStart, fix.tEnd), DataParser.REAL_FIX] = True

    et_data_dict[DataParser.DF_FIXAT] = fixations
    return et_data_dict


def mark_pupils(et_data_dict, params):
    """

    :param et_data_dict:
    :param params:
    :return:
    """
    samps = et_data_dict[DataParser.DF_SAMPLES]
    eye = params[DataParser.EYE]
    samps[DataParser.REAL_PUPIL] = samps[f"{eye}Pupil"]
    samps.loc[samps[f"{eye}{DataParser.HERSHMAN_PAD}"] == True, DataParser.REAL_PUPIL] = np.nan
    et_data_dict[DataParser.DF_SAMPLES] = samps
    return et_data_dict


def analysis_windows(trial_info, et_data_dict, sub_params, is_tobii=False):
    """
    This method prepares the pre-processed eye-tracking data to be analyzed as per the pre-regsitered linear models.
    Pre-registration 4.0: https://osf.io/gm3vd
    We need for each time window (first, second, third), per trial (=row), the following information:
    - the average fixation distance from screen center (in that window)
    - the maximal saccade amplitude (in that window)
    - the number of blinks (in that window)
    Extreme cases: if a blink/saccade are partially in one window and partially in another, we will attribute the
    blink/saccade to the window containing the majority of that event.
    If a fixation is between two windows, it will be attributed to both of them considering its duration in each
    window.
    :param sub_data: the dictionary containing all the subjects' eye-tracking data, i.e., their trial information df,
    their lab-specific parameters, the tracked eye and the et_data dictionary with dfs per eye data type.
    :return:
    """
    eye = sub_params["Eye"]
    samps = et_data_dict[DataParser.DF_SAMPLES]

    # FIXATION ANALYSIS PREPARATION
    real_fix_samps = samps.loc[samps[DataParser.REAL_FIX] == True]  # take only samples that are real fixations
    for window in [DataParser.TRIAL, DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
        # The Current implementation IS WEIGHTED
        mean_fix = real_fix_samps.groupby([window]).mean().reset_index()
        for index, trial in mean_fix.iterrows():
            if trial[window] != -1:  # if -1, this means it is not part of the current window
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}CenterDistDegs"] = trial[f"{eye}CenterDistDegs"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"X{window}CenterDistDegsSigned"] = trial[f"{eye}XCenterDistDegsSigned"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"Y{window}CenterDistDegsSigned"] = trial[f"{eye}YCenterDistDegsSigned"]

                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}CenterDistDegs{BL_CORRECTED}"] = trial[f"{eye}CenterDistDegs{BL_CORRECTED}"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"X{window}CenterDistDegsSigned{BL_CORRECTED}"] = trial[f"{eye}XCenterDistDegsSigned{BL_CORRECTED}"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"Y{window}CenterDistDegsSigned{BL_CORRECTED}"] = trial[f"{eye}YCenterDistDegsSigned{BL_CORRECTED}"]
        median_fix = real_fix_samps.groupby([window]).median().reset_index()
        for index, trial in median_fix.iterrows():
            if trial[window] != -1:  # if -1, this means it is not part of the current window
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}{MEDIAN}CenterDistDegs"] = trial[f"{eye}CenterDistDegs"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"X{window}{MEDIAN}CenterDistDegsSigned"] = trial[f"{eye}XCenterDistDegsSigned"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"Y{window}{MEDIAN}CenterDistDegsSigned"] = trial[f"{eye}YCenterDistDegsSigned"]

                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}{MEDIAN}CenterDistDegs{BL_CORRECTED}"] = trial[f"{eye}CenterDistDegs{BL_CORRECTED}"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"X{window}{MEDIAN}CenterDistDegsSigned{BL_CORRECTED}"] = trial[f"{eye}XCenterDistDegsSigned{BL_CORRECTED}"]
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"Y{window}{MEDIAN}CenterDistDegsSigned{BL_CORRECTED}"] = trial[f"{eye}YCenterDistDegsSigned{BL_CORRECTED}"]

    if not is_tobii:  # no blinks in tobii eye data
        # BLINK ANALYSIS PREPARATION
        samp_blinks = samps[~samps[f"{eye}Hershman"].isna()]  # take only samples that are real blinks (Hershman)
        blink_numbers = list(samp_blinks["blink_number"].unique())  # retrieve their blink numbers, as numbered in the parsing step
        # initialize a dictionary where key=blink, val= a dictionary counting how many samples out of that blink are in each window
        blink_dict_counter = {elem: {DataParser.FIRST_WINDOW: 0, DataParser.SECOND_WINDOW: 0, DataParser.THIRD_WINDOW: 0} for elem in blink_numbers}
        for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
            samps_blink_window = samp_blinks.loc[samp_blinks[window] != -1, :]  # all blink samples that are in that window
            samps_in_window_count = samps_blink_window.groupby(["blink_number"]).count().reset_index()
            for index, blink_num in samps_in_window_count.iterrows():
                blink_dict_counter[blink_num["blink_number"]][window] = blink_num[DataParser.T_SAMPLE]

        """
        Once each numbered blink's samples within each window were counted, the following calculates where does the majority
        of the blink occured - i.e., what is the window in which the number of samples belonging to that blink is maximal.
        The window with the largest amount of samples belonging to that blink is added to the MaxWindow column.
        """
        for key in blink_dict_counter.keys():
            max = 0
            max_val = None
            for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
                if blink_dict_counter[key][window] > max:
                    max = blink_dict_counter[key][window]
                    max_val = window
            if max > 0:
                samp_blinks.loc[samp_blinks["blink_number"] == key, "MaxWindow"] = max_val

        for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
            trial_info.loc[:, f"{window}NumBlinks"] = 0  # nans to zeros, as these are blink counts
            samps_blink_window = samp_blinks[(samp_blinks[window] != -1) & (samp_blinks["MaxWindow"] == window)]  # all blink samples that are in that window
            unique_blinks_in_window = samps_blink_window.groupby([window]).blink_number.nunique()
            for index, count in unique_blinks_in_window.items():
                trial_info.loc[trial_info["trialNumber"] == index, f"{window}NumBlinks"] = count

    # SACCADE ANALYSIS PREPARATION
    if et_data_dict[DataParser.DF_SACC].empty:
        print(f"WARNING: zero saccades; skipping saccade analysis preparation")
        return trial_info

    samp_saccs = samps[samps[DataParser.REAL_SACC] == True]  # take only samples that are real blinks (Hershman)
    sacc_numbers = list(samp_saccs["sacc_number"].unique())  # retrieve their blink numbers, as numbered in the parsing step
    # initialize a dictionary where key=blink, val= a dictionary counting how many samples out of that blink are in each window
    sacc_dict_counter = {elem: {DataParser.FIRST_WINDOW: 0, DataParser.SECOND_WINDOW: 0, DataParser.THIRD_WINDOW: 0}
                          for elem in sacc_numbers}
    for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
        samps_sacc_window = samp_saccs.loc[samp_saccs[window] != -1, :]  # all blink samples that are in that window
        samps_in_window_count = samps_sacc_window.groupby(["sacc_number"]).count().reset_index()
        for index, sacc_num in samps_in_window_count.iterrows():
            sacc_dict_counter[sacc_num["sacc_number"]][window] = sacc_num[DataParser.T_SAMPLE]

    """
    Once each numbered saccade samples within each window were counted, the following calculates where does the majority
    of the saccade occured - i.e., what is the window in which the number of samples belonging to that saccade is maximal.
    The window with the largest amount of samples belonging to that saccade is added to the MaxWindow column.
    """
    for key in sacc_dict_counter.keys():
        max = 0
        max_val = None
        for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
            if sacc_dict_counter[key][window] > max:
                max = sacc_dict_counter[key][window]
                max_val = window
        if max > 0:
            samp_saccs.loc[samp_saccs["sacc_number"] == key, "MaxWindow"] = max_val

    for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
        samps_sacc_window = samp_saccs[(samp_saccs[window] != -1) & (samp_saccs["MaxWindow"] == window)]  # all sacc samples that are in that window
        max_saccs_in_window = samps_sacc_window.groupby([window]).max().reset_index()
        for index, max_sacc in max_saccs_in_window.iterrows():
            trial_info.loc[trial_info["trialNumber"] == max_sacc[window], f"{window}MaxAmp"] = max_sacc[f"ampDeg"]

        unique_saccs_in_window = samps_sacc_window.groupby([window]).sacc_number.nunique()
        for index, count in unique_saccs_in_window.items():
            trial_info.loc[trial_info["trialNumber"] == index, f"{window}NumSaccs"] = count

    return trial_info


def extract_data(sub_code, et_path, asc_files, save_path, sub_beh_data):  # example_log_path
    """
    This is the main manager function of the quality checks. It receives its input from the ET_manager module. Then,
    it extracts and parses all the ET data, to create structures which will then be summarized and saved as the subject's
    QC.
    :param sub_code: subject code for exp.2
    :param et_path: the path to subject's "ET" folder, which assumed to be organized in accordance with the XNAT
    convention.
    :param asc_files: path to the folder of the relevant ascii files (ET raw data).
    :param save_path: the path to the dir where all data will be saved.
    :return: the pickle file containing all the subject's extracted and parsed data, which can be used for analysis.
    """
    print(f'-------- Extracting data for subject: {sub_code} ----------')

    # choose the first file in the Ascii file-name list and give the full path to this file to the Init params function
    ascii_file_path = os.path.join(et_path, asc_files[0])
    params = ET_param_manager.InitParams(sub_code, ascii_file_path)

    # if data was already extracted
    pick_file = [f for f in os.listdir(save_path) if fnmatch.fnmatch(f, '%sEyeTrackingData.pickle' % sub_code)]
    if len(pick_file) > 0:
        print(f"Subject {sub_code} already has a pickle file, QC will use the existing file and not re-extract data")
        return

    if sub_code in INVALID_LIST:
        print(f"Subject {sub_code} does not have analyzable ET data; ignored")
        return

    """
    et_data : the dictionary which contains information parsed based on EYELINK logging, the parsing matches the
    documentation in eyelink's programmer's manual version 3.0:
    -dfRec contains information about recording periods (often trials)
    -dfMsg contains information about messages (usually sent from stimulus software)
    -dfFix contains information about fixations
    -dfSacc contains information about saccades
    -dfBlink contains information about blinks
    """
    et_data = dict.fromkeys([DataParser.DF_MSG, DataParser.DF_FIXAT, DataParser.DF_SACC, DataParser.DF_BLINK, DataParser.DF_SAMPLES]) #DataParser.DF_REC,

    # loop through the asc files and read the data
    print('Parsing subject eye tracking data')
    next_time_delta = 0
    total_problem_time = 0
    for fl in asc_files:
        is_valid = 0
        header = fl.split('.')[0]
        header_split = header.split('_')
        for chunk in header_split:
            if DURR in chunk:
                blockNo = chunk.split(DURR)[1]
                is_valid = 1
        if is_valid == 0:  # else, this is the MEG's "RestinEO" file (or, an invalid file, which is not supposed to happen)
            print(f'-- SKIPPING {fl}: This is a non-experiment file!-- ')
            continue
        print(f'-- Block {blockNo} --')
        # this is the Eyelink data information for the specific ascii file that is being processed.
        flEyeData, is_problematic, temp = DataParser.ParseEyeLinkAsc(os.path.join(et_path, fl), next_time_delta, total_problem_time)
        # If the subject has a wrong timing, add the last timestamp to the newer files from now on.
        if is_problematic:
            total_problem_time += next_time_delta
        next_time_delta = temp
        # update the messages with the world number
        flEyeData[DataParser.DF_MSG]['BlockID'] = [f"Block {blockNo}"] * flEyeData[DataParser.DF_MSG].shape[0]
        # now, add this file's parsed information to the general dictionary (et_data) that contains all files' data
        for key in [DataParser.DF_MSG, DataParser.DF_FIXAT, DataParser.DF_SACC, DataParser.DF_BLINK, DataParser.DF_SAMPLES]:  # DataParser.DF_REC,
            if flEyeData[key] is not None:  # as SC130 for example had no fixations in one of the ASCII files (130103)
                et_data[key] = pd.concat([et_data[key], flEyeData[key]])

    print("-- All blocks processed --")
    """
    Example: SD161
    Go over all the keys in the subject et_data dictionary (fixations, saccades, blinks) and check they all have data.
    If one dictionary is completely empty (i.e., no parsed blinks/saccades/fixations..) - this subject is invalid as 
    it doesn't make sense to have 0 fixations in 1:30 hrs of experiment.
    """
    for key in [DataParser.DF_MSG, DataParser.DF_FIXAT, DataParser.DF_SACC, DataParser.DF_BLINK, DataParser.DF_SAMPLES]:
        if et_data[key].empty:  # if the dataframe has nothing in it after parsing ALL files
            print(f"Issue with subject's {sub_code}: no {key} at all in the entire experiment. Inspect manually")
            return
    """
    Example: SB999
    This subject had dfSacc but it was all empty: no tracking in the R eye, and in the left, all zeros. 
    Unclear why it was tagged by Eyelink as a saccade.
    It doesn't make sense to have 0 saccades in 1:30 hrs of experiment.
    """
    if len(et_data[DataParser.DF_SACC].vPeak.unique()) == 1 and et_data[DataParser.DF_SACC].shape[0] > 1:
        if et_data[DataParser.DF_SACC].vPeak.unique()[0] == 0:
            print(f"Issue with subject's {sub_code}: no {DataParser.DF_SACC} at all in the entire experiment. Inspect manually")
            return

    params = ET_param_manager.UpdateParams(params, et_data[DataParser.DF_MSG])

    print('Pre-processing ET data: recognizing fixations and saccades (remove artifacts) ')
    """
    FIXATIONS = any sample with 'Lfix'/'Rfix' == 1
    SACCADES = any sample with 'Lsacc'/'Rsacc' == 1
    BLINKS/MISING DATA = ALL the other samples! (both samples where 'Lblink'/'Rblink' == 1 AND samples in which
    the sample is not fixation and not saccade).
    """

    print("\n--- Data Pre-Processing: Blinks ---")
    print("Mark eye tracking data according to Eyelink messages")
    # this marks dfSamples such that every sample is a fixaion / saccade / blink according to EYELINK
    et_data = DataParser.et_data_mark_Eyelink(et_data)

    print(f"Running Hershman's blink detection algorithm to mark real blinks (as opposed to missing pupil) {datetime.datetime.now()}")
    # this marks dfSamples AND dfBlink with a HERSHMAN column denoting this is a real blink according to Hershman
    et_data = blink_detection(et_data, params)

    # Pad blinks with et_param_manager.BLINK_PAD_MS window before and after. This is used for saccade/fixation/pupil filtering
    et_data = pad_blinks(et_data, params)

    print("\n--- Data Pre-Processing: Saccades ---")

    print(f"Running E&K's saccade and microsaccade detection algorithm {datetime.datetime.now()}")
    et_data = saccade_detection(et_data, params)  # this is the RAW EK OUTPUT

    print(f"mark samples with EK data")
    if et_data[DataParser.DF_SACC].empty:  # subject SB999, who does have fixations and blinks - dfSacc is empty!
        print(f"WARNING: subject {sub_code} has 0 saccades; skipping this step")
    # if empty, mark_saccades just adds null columns
    et_data = mark_saccades(et_data, params)  # this marks EK saccades IFF they don't overlap with padded blink periods

    print(f"mark samples with fixation data")
    et_data = mark_fixations(et_data, params)  # this marks Eyelink fixations IFF they don't overlap with padded blink periods

    """
    Then, we pre-process pupil data to prepare it for pupil-size analysis. 
    The pre-processing is limited, because Eyelink pupil size units are ARBITRARY, and the labs did not calibrate 
    the real pupil size with an artificial pupil (the Eyelink units - to mms conversion). 
    Therefore, we will take the artibtrary sizes as they are for future analyses (across subjects); However, 
    they still need to be filtered such that blinks are not included. 
    """
    et_data = mark_pupils(et_data, params)  # this nullifies pupil size IFF it overlaps with padded blink periods

    print('Comparing to behavioral trigger messages in eye-tracking data')
    # this gets a df similar to sess_beh_data but derived from the triggers the videogame sent to Eyelink
    et_trial_info = DataParser.get_trial_info(et_data[DataParser.DF_MSG], params)

    """
    Because we might have read an aborted/restarted file, we select as trials only the LATEST trial in each Block/Miniblock.
    This allows us to mark the same trials as the BEH data, hence use only timestamps that are relevant for those trials when markering trials.
    **NOTE**: this relies on the fact that when restarting a run, it starts from the top. 
    """
    et_trial_info_unsorted = et_trial_info.groupby([DataParser.BLOCK_COL, DataParser.MINIBLOCK_COL, DataParser.TRIAL_COL]).tail(1)

    """
    SB035 - in this subject's ET trigger data, 2 trials from block 7 are somehow sent AFTER the entire block 8 is done.
    Meaning, block 7 starts from mibiblock 25 **trial 3**, and trials 1, 2 in miniblock 25 (block 7) appear between
    blocks 8 and 9!
    As the CRF of this subject did not include any information about anything wrong, and as all triggers are technically
    there, we will continue by sorting the et_trial_info dataframe by block.
    """
    et_trial_info = et_trial_info_unsorted.sort_values(by=[DataParser.BLOCK_COL, DataParser.MINIBLOCK_COL, DataParser.TRIAL_COL], ascending=True)
    if not(et_trial_info.equals(et_trial_info_unsorted)):
        print(f"WARNING: {sub_code} ET TRIGGER TIMINGS WERE OUT OF PLACE!")

    # match the ET response events (derived from triggers) with the behavioral log response events (what actually happened + DMT corrections)
    trial_info = DataParser.set_trial_info(et_trial_info, sub_beh_data.processed_data, sub_code, save_path)

    print('Sequencing trials')
    et_data, trial_info = DataParser.et_data_to_trials(et_data, trial_info, params)

    print("--- DATA PARSING DONE ---")

    print("--- Data Pre-Processing: Fixations ---")
    print('Calculate fixation baseline from target')
    et_data = calculate_baseline(et_data, params)

    print('Calculating gaze (fixation) distances from targets')
    # After dist_from_target is done, et_data_per_trial[DF_SAMPLES] now contains additional columns, denoting for each
    # sample (doesn't matter what it is) its distance from the target stimulus, its distance from the fixation area,
    # and whether the distance from fixation is counted as gazing at the fixation area (as per the parameter that
    # defines the radius within which we consider a gaze to be "within the are of fixation").
    et_data = dist_from_target(et_data, params)

    print('Calculating fixation, saccade, and blink stats in windows')
    """
    In this step, we take the parsed data and pre-process it for future analyses and descriptives. 
    As per the pre-registration, we are interested in 3 time windows (onset-500ms, 500ms-1000ms, 1000ms-1500ms).
    Within each time window, we are intersted in fixations (average distance from screen center), saccades (maximum
    amplitude), and blinks (number) during that window. The analysis_windows method handles these summaries, 
    and adds them as columns in the trial_info dataframe
    """
    trial_info = analysis_windows(trial_info, et_data, params)

    # pickle the data
    print("\n--- Done Data Pre-Processing ---")
    print('Saving subject eye tracking data...')
    fl = open(os.path.join(save_path, f"{sub_code}EyeTrackingData.pickle"), 'wb')
    sub_data = {"trial_info": trial_info, "et_data_dict": et_data, "params": params}
    pickle.dump(sub_data, fl)
    fl.close()

    return sub_data
