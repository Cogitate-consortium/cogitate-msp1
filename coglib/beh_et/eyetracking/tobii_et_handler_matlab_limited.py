import os
import pandas as pd
import numpy as np
import pickle
import datetime
import math

import DataParser
import ET_data_processing
import ET_param_manager
import AnalysisHelpers
import ET_qc_manager
import ET_data_extraction

"""
This script takes all the tobii csv files from their HPC locations, and (subject by subject) calls the matlab script "tobii_parsrer.m" to parse them.
tobii_parser prepares the data s.t. the ClusterFix package could be run on the data, to extract fixations and saccades based on the coordinate data in the csv files. 
Notably, ClusterFix crashes on NaN samples. Therefore, we use pchip to interpolate missing values, BUT as some sections contain prolonged durations of missing values, 
every section above 500ms that is missing is later ignored. The end result is a pickle file per subject with dfSamples- which includes all the raw samples as well as their
ClusterFix taggings and interpolation results, and estimated fixations (dfFix) and saccades (dfSacc) that contain ClusterFix results on all data that was not missing for 
longer than the predefined threshold. 

Author: Rony Hirschhorn
"""


TOBII_NAME = "_ET_V1_DurR"
BLOCK = "block_number"
SYSTEM_TS = "system_time_stamp"  # system ts = experimental computer; device ts = tobii clock. so we take the system ts.
eye = "R"
y_pix = "yPix"
x_pix = "xPix"
ABORTED = "ABORTED"
ECOG_SUBFOLDER = "ECOG_V1"
TRIGGERS = "Triggers"
MISSING_PAD = DataParser.HERSHMAN_PAD # Because we want to be consistent with Eyelink data, we call hershman although that is missing
DF_MISSING = "dfMissing"

SCREEN_WIDTH_PIX = 1920.0   # This is float for matlab computation needs
SCREEN_HEIGHT_PIX = 1080.0  # This is float for matlab computation needs

MICRO_TO_MILI = 1000
MISSING_DATA_THRESH = 500  # in MILLISECONDS, threshold abovewhich there's no point in interpolating as there's too much - Guidelines from Simon Henin, 2022-11-21
# missing data; thus, the ClusterFix interpolation result will be ignored

TOBII_SAMPLING_RATE = 90  # Hz


def parse_tobii(tobii_raw_path, beh_data_path, root_folder, save_path):
    """
    This method manages the parsing and pre-processing of all subjects whose gaze data was collected with tobii.
    It runs the pre-processing, and saves pickle files with the processed data.

    The files saved gaze coordinates, but no pupil data. This means that identifying fixations and saacades is not
    trivial, and identifying (real) blinks and pupil size - is impossible.
    Therefore, to be able to extract fixations and saccades the pre-processing pipeline calls ClusterFix
    https://buffalomemorylab.com/clusterfix
    This is a Matlab package for extracting fixations and saccades from X, Y coordinates.
    * NOTE*: for piping Matlab from Python, MATLAB.ENGINE VERSION HAS TO BE 9.11.19!

    :param tobii_raw_path: the folder where all the subjects who have tobii (SF) are (ECoG)
    :param beh_data_path: the folder where all the behavioral data resides (DMT's QC folder)
    :param save_path: where to save the processed data to
    """
    qc_path = os.path.join(root_folder, ET_qc_manager.DMT_FOLDER, ET_qc_manager.V1, ET_qc_manager.ET_RES_FOLD)
    subs = [f for f in os.listdir(tobii_raw_path)]
    beh_data_all = ET_qc_manager.load_beh_data(beh_data_path)
    tobii_parsed_subs = []
    for sub in subs:
        sub_code = sub
        if not sub_code.startswith("SF"):
            continue
        tobii_sub_path = os.path.join(tobii_raw_path, sub_code, f"{sub_code}_{ECOG_SUBFOLDER}", "RESOURCES", "ET")
        sub_save_path = os.path.join(save_path, f"{ET_data_processing.SUBJECT}-{sub_code}", ET_qc_manager.SES_V1, ET_qc_manager.ET_RES_FOLD)
        if not os.path.exists(sub_save_path):
            os.mkdir(sub_save_path)

        print(f"--- Parsing subject {sub_code} ---")
        if sub_code not in beh_data_all:
            print(f"sub {sub_code} does not have BEH data")
            continue
        # get subject's behavioral data
        sub_beh_data = beh_data_all[sub_code]
        if parse_tobii_sub(sub_code, tobii_sub_path, sub_save_path, sub_beh_data):
            tobii_parsed_subs.append(os.path.join(sub_save_path, f"{sub_code}EyeTrackingData.pickle"))

    max_val = 0
    minimal_dims_va = (24.670391061452513, 13.877094972067038)
    ET_data_processing.fix_hist_mod(tobii_parsed_subs, ET_qc_manager.ECoG, DataParser.FIRST_WINDOW, qc_path, minimal_dims_va, max_val,
                                    "phase3_tobii", plot=True, square=True, in_va=True, filter_fix=False)

    return


def UpdateParamsTobii(params):  # fulllog_path
    """
    This function updates the parameters dictionary based on the eye tracker messages
    :param params: the parameters dict to be updated (should be the output of InitParams)
    :param eyeDFs: dataframe containing all ET messages
    :param fulllog_path: the path to one full log file where the stimulus information is
    :return: an updated parameter dictionary
    """

    params['SamplingFrequency'] = float(100)
    params['ScreenResolution'] = np.array((1920, 1080))

    # screen conversion factor (how many CM per pixel) : ScreenWidth is the actual width IN MILLIMIETERS (so /10)
    # and screen resolution[0] is the pixels in width
    params['PixelPitch'] = [(params['ScreenWidth'] / 10) / params['ScreenResolution'][0],
                            (params['ScreenHeight'] / 10) / params['ScreenResolution'][1]]

    # center location
    params['ScreenCenter'] = params['ScreenResolution'] / 2

    # view_distance in cm = view_distance / 10 as 'ViewDistance' is in MILLIMETERS
    params['stimPix'] = AnalysisHelpers.deg2pix(params['ViewDistance'] / 10, params['stimAng'], params['PixelPitch'])
    params['DegreesPerPix'] = params['stimAng'] / params['stimPix']

    return params


def trial_info_tobii(df_samples, params):
    """
    A parallel of trial info. It prepares trigger information to look like Eyelink info.
    """
    triggers_df = df_samples[~df_samples[TRIGGERS].isna()]
    triggers_df.rename(columns={TRIGGERS: "text", BLOCK: "BlockID", DataParser.T_SAMPLE: "time"}, inplace=True)
    triggers_df['text'] = triggers_df['text'].astype(int)
    triggers_df['text'] = triggers_df['text'].astype(str)
    triggers_df['BlockID'] = triggers_df['BlockID'].astype(str)
    triggers_df["BlockID"] = "Block " + triggers_df["BlockID"]
    # now we can call get_trial_info. With tobii, there are additional weird triggers (millions), so we ignore them (with Eyelink we don't ignore anything)
    trial_info = DataParser.get_trial_info(triggers_df, params, is_tobii=True)
    if not trial_info[trial_info[ET_param_manager.STIM_OFFSET].isna()].empty:
        print("subject has LIMITED trial information, please look manually as well") # TODO: IS THIS PRINT OK?
        trial_info = trial_info[~trial_info[ET_param_manager.STIM_OFFSET].isna()]
    return trial_info


def pad_missing_data(et_data):
    """
    With Eyelink data, we pad blinks with a window of ms before and after each real blink, such that saccades and
    fixations do not contain this data. With tobii, we do not have blink information (or pupil information),
    so we now pad the periods of missing data with the same padding we do for blinks in the Eyelink case.
    """
    et_samples = et_data[DataParser.DF_SAMPLES]
    et_samples.loc[:, f"{MISSING_PAD}"] = False

    if "is_missing_pad" not in et_samples.columns:
        return et_data

    missing_data = et_samples[et_samples["is_missing_pad"] == 1]
    for index, missing in missing_data.iterrows():
        # Labeling an SSACC...ESACC pair with one or more SBLINK events between them as BLINKS
        et_samples.loc[(et_samples[DataParser.T_SAMPLE] <= missing[DataParser.T_SAMPLE] + ET_param_manager.BLINK_PAD_MS) &
                       (et_samples[DataParser.T_SAMPLE] >= missing[DataParser.T_SAMPLE] - ET_param_manager.BLINK_PAD_MS), f"{MISSING_PAD}"] = True
    return et_data


def parse_tobii_sub(sub_code, tobii_raw_path, save_path, sub_beh_data):
    """
    This is the method that manages parsing of a tobii data file - the file was saved in csv format, and contains
    for each sample the gaze X, Y coordinates (in pixels, in proportion of the screen relative to its size, and in cm).
    The following prepares the raw data for the Clustefix method, and then parses the output.

    *NOTE*: the Clustefix method assumes NO MISSING DATA SAMPLES. Meaning, the raw data needs to be filtered s.t. it
    does not contain periods of missing data -> then the missing samples need to be inserted to the Clustefix results
    once we get them. The reason for that is that Clustefix crashes on missing data.

    :param sub_code: subject code
    :param tobii_raw_path: path to a single subject's tobii output
    :param save_path: path to save the processed data to
    :param sub_beh_data: the subject's behavioral data
    """

    # step 1: concatenate all files into a single data session
    try:
        tobii_files = [f for f in os.listdir(tobii_raw_path)]
        tobii_files = [f for f in tobii_files if TOBII_NAME in f]
    except Exception:
        print(f"subject {sub_code} has no tobii files, cannot parse")
        return False

    tobii_files.sort()
    tobii_file_generic = f"{sub_code}{TOBII_NAME}"
    file_list = list()
    df_fix_list = list()
    df_sacc_list = list()
    df_missing_list = list()
    params = ET_param_manager.InitParams(sub_code, None, is_tobii=True)
    params = UpdateParamsTobii(params)
    for f in tobii_files:
        data = pd.read_csv(os.path.join(tobii_raw_path, f))
        """
        Change the X, Y coordinates from the proportion on the screen --> to degrees visual angle
        NOTE: Clustefix ASSUMES the coordinate input is in DVA!
        """
        data["R_x"] = data["R_x"] * params['DegreesPerPix'] * SCREEN_WIDTH_PIX
        data["L_x"] = data["L_x"] * params['DegreesPerPix'] * SCREEN_WIDTH_PIX
        data["R_y"] = data["R_y"] * params['DegreesPerPix'] * SCREEN_HEIGHT_PIX
        data["L_y"] = data["L_y"] * params['DegreesPerPix'] * SCREEN_HEIGHT_PIX

        data = mark_missing_data(data)  # interpolate over nans according to a threshold: RIGHT NOW DOES NOTHING BUT MARKING MISSING SAMPLES
        # mark the block and append the file
        block_num = f.replace(tobii_file_generic, "")[0]
        data.loc[:, BLOCK] = block_num
        file_list.append(data)

    # step : unify
    df_samples = pd.concat(file_list)
    df_samples[SYSTEM_TS] = df_samples[SYSTEM_TS] / MICRO_TO_MILI  # convert the timestamps

    df_samples.rename(columns={SYSTEM_TS: DataParser.T_SAMPLE, f"L_{x_pix}": "LX_p", f"L_{y_pix}": "LY_p", f"R_{x_pix}": "RX_p", f"R_{y_pix}": "RY_p", "clusterfix_R_x": "RX", "clusterfix_R_y": "RY"}, inplace=True)

    # Move tSample to first column (important for later)
    tSample_vals = df_samples.pop(DataParser.T_SAMPLE)
    df_samples.insert(0, DataParser.T_SAMPLE, tSample_vals)
    df_samples.reset_index(drop=True, inplace=True)

    # now we finally have the eye-tracking data from tobii subjects; let's continue like we did for the Eyelink case
    et_trial_info = trial_info_tobii(df_samples, params)
    """
    Because we might have read an aborted/restarted file, we select as trials only the LATEST trial in each Block/Miniblock.
    This allows us to mark the same trials as the BEH data, hence use only timestamps that are relevant for those trials when markering trials.
    **NOTE**: this relies on the fact that when restarting a run, it starts from the top. 
    """
    et_trial_info = et_trial_info.groupby([DataParser.BLOCK_COL, DataParser.MINIBLOCK_COL, DataParser.TRIAL_COL]).tail(1)

    # a parallel of Eyelink's subjects' et_data_dict
    et_data_dict = {DataParser.DF_SAMPLES: df_samples}
    et_data_dict = pad_missing_data(et_data_dict)

    # match the ET response events (derived from triggers) with the behavioral log response events (what actually happened + DMT corrections)
    trial_info = DataParser.set_trial_info(et_trial_info, sub_beh_data.processed_data, sub_code, save_path)
    print('Sequencing trials')
    et_data, trial_info = DataParser.et_data_to_trials(et_data_dict, trial_info, params, is_tobii=True)

    et_data = ET_data_extraction.calculate_baseline(et_data, params, is_tobii=True)
    et_data[DataParser.DF_SAMPLES][f"RX{ET_data_extraction.BL_CORRECTED}"] = et_data[DataParser.DF_SAMPLES][f"RX_p{ET_data_extraction.BL_CORRECTED}"] * params['DegreesPerPix']
    et_data[DataParser.DF_SAMPLES][f"RY{ET_data_extraction.BL_CORRECTED}"] = et_data[DataParser.DF_SAMPLES][f"RY_p{ET_data_extraction.BL_CORRECTED}"] * params['DegreesPerPix']

    et_data = ET_data_extraction.dist_from_target(et_data, params)

    # pickle the data
    print("\n--- Done Data Pre-Processing ---")
    print('Saving subject eye tracking data...')
    fl = open(os.path.join(save_path, f"{sub_code}EyeTrackingData.pickle"), 'wb')
    sub_data = {"trial_info": trial_info, "et_data_dict": et_data, "params": params}
    pickle.dump(sub_data, fl)
    fl.close()

    return True


def mark_missing_data(orig):
    """
    For each missing data period, this method decides whether to take the Clusterfix interpolation result, or declare
    the period as a missing data period (if too long).
    :param cluster_fix: The output of the Matlab Clusterfix method https://buffalomemorylab.com/clusterfix
    :param orig: The subject's raw csv data
    :return: the updated csv data, a dataframe of missing data
    """
    nan_series = orig["R_x"].isnull()
    nan_indices = list(nan_series[nan_series].index)
    orig.loc[orig["R_x"].isnull(), "is_missing"] = 1  # data is a missing period

    start = None
    end = None
    just_ended = True
    prev = None
    sample_s = (1/TOBII_SAMPLING_RATE)  # 1/Hz = seconds
    sample_ms = sample_s * 1000
    """
    This goes over the nan indices: if the next nan index is not the current + 1, look at the current range.
    If range's length exceeds the pre-defined threshold, set "is_missing" to 1. Otherwise, set it to 0.
    Set the clusterfix coordinates to be the original coordinates in non-NaN indices, and to the interpolated 
    coordinates if "is_missing" == 0 (if 1, the cell is NaN).
    """
    # This helps create a missing DF, which helps to mark padded missing data (similar to padded Hershman)
    for ind in nan_indices:
        if prev is None or just_ended:
            start = ind
            prev = ind
            just_ended = False
        elif prev + 1 != ind:
            end = prev
            prev = ind
            if (end - start) > int(MISSING_DATA_THRESH/sample_ms):  # if missing data for more than the threshold
                orig.loc[start:end, "is_missing_pad"] = 1  # data is a missing period
            start = ind
        else:
            prev = ind

    orig.loc[:, "clusterfix_R_x"] = orig.loc[:, "R_x"]
    orig.loc[:, "clusterfix_R_y"] = orig.loc[:, "R_y"]
    return orig


if __name__ == '__main__':
    parse_tobii(tobii_raw_path=r"/mnt/beegfs/XNAT/COGITATE/ECoG/Raw/projects/CoG_ECoG_PhaseII",
                beh_data_path=r"/mnt/beegfs/XNAT/COGITATE/QC/v1/beh",
                root_folder=r"/mnt/beegfs/XNAT/COGITATE",
                save_path=r"/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/qcs")

