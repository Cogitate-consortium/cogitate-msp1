import os
import ET_data_extraction
import fnmatch
import pickle
import pandas as pd

import ET_data_processing
import QualityChecker
from multiprocessing import Process

""" Quality Checks Management Module

This module manages everything related to the quality checks of the eye-tracking output data of experiment 2.
NOTE: The checks assumes XNAT-COMPATIBLE STRUCTURE: meaning, that "sub_dir" is a folder which includes one or more 
subject folders (e.g. "SZ104"). Each subject folder contains 2 sub-folders: /BEH, /ET (e.g., "SZ104/BEH"). 
Each data type (BEH/ET) contains the structure of the resource as it should've been uploaded to XNAT 
(e.g., "SZ104/BEH/SZ104/1"). 
See more here: 
https://twcf-arc.slab.com/posts/2-subject-visit-data-structure-and-naming-conventions-gkibcour

The QC creates a folder under "sub_dir" which is named "QC_summary", in which all the resulting plots and data tables 
are saved, per each subject separately (e.g., "sub_dir/QC_summary/SZ104"). In the result folder, you'll find:
- "QC_results_Epoch" / "QC_results_PreStim" / "QC_results_StimDuration" : 3 files with the same structure. All of them
contain summary information about the % of fixation on the stimulus/center, and the distance of gaze from stimulus/center, 
both across all trials and per trial type (Location, World etc), in a specific time-frame 
(pre-stimulus, stimulus duration, or the entire epoch - all of these windows are defined in "ET_param_manager.py" file). 
- A folder per condition type (stimulus category, game world, replay world, location, visibility), containing plots and 
their corresponding data tables. The plots are:
    - Heatmaps: of gaze distribution during each time-window type (Epoch, Stimulus duration, Pre-stim)
    - Line plots: of mean (and sem) of gaze distance from stimulus/center, across each time-window type.

@authors: RonyHirsch, AbdoSharaf98
"""

BEH = "BEH"
ET = "ET"
ET_RES_FOLD = "et"
FULL_LOGS = 'FullLogs'
FULLLOGNAME = "FullLogLevel"
CSV = 'csv'
ASC = 'asc'
COMPLETED = 'COMPLETED'
SUBLEN = 5
TEST_FOLDER = "test_inputs"

COGITATE_PATH = "/mnt/beegfs/XNAT/COGITATE"
DMT_FOLDER = "QC"
V1 = "v1"
RESOURCES = "RESOURCES"
FMRI = 'fMRI'
MR = "MR"
MEEG = 'MEEG'
MEG = 'MEG'
ECOG = 'ECOG'
ECoG = 'ECoG'
RAW = "Raw"
PHASE = "phase_2"
PROCESSED = "processed"
BIDS = "bids"
DERIV = "derivatives"
QCS = "qcs"
SUB_PREFIX = "sub-"
SES_V1 = "ses-v1"
PROJ = "projects"
MODALITY_MAPPING = {FMRI: MR, MEG: MEEG, ECoG: ECOG}  # xnat-name: raw-folder-name-under-subject on the HPC
SESS_INFO_FILE = "_SessionInfo.txt"
BEH_QC_FILE_NAME = "quality_checks"
BEH_QC_SUB_COL = "subCode"
BEH_QC_VALID_COL = "Is_Valid?"


def is_sub(sub_code: str):
    """
    Checks whether a given string is a subject code
    :param sub_code: string of some name
    :return: True if input string is subcode, False if not
    """
    if len(sub_code) != SUBLEN:
        return False
    if not (sub_code[2:].isdigit()):
        return False
    if not (sub_code[0:2].isupper()):
        return False
    return True


def load_beh_data(beh_data_path):
    file_name = "subject_beh.pickle"
    beh_data_file = os.path.join(beh_data_path, file_name)
    if not os.path.exists(beh_data_file):
        print("ERROR: no behavioral data summary found. Run behavioral QC")
        return None
    fl = open(beh_data_file, 'rb')
    beh_data = pickle.load(fl)
    fl.close()
    print(f"Found behavioral data")
    return beh_data


def qc_ET(beh_data_path, phase_name, straight_to_qc=False, root_folder=COGITATE_PATH):
    """
    Given a path to a directory where one or more subject folders are, perform QC on each subject separately,
    Save the results and create a pickle to hold subject data for future analysis.
    :param sub_dir: path to where subject directories are.
    NOTE: this code assumes that under each subject folder the BEH and ET data are saved saprately, in accordance with
    the naming and structure conventions:
    https://twcf-arc.slab.com/posts/2-subject-visit-data-structure-and-naming-conventions-gkibcour#eye-tracking-files-experiment-2
    :param save_path: where to save the QC resulting data and figures
    """

    qc_path = os.path.join(root_folder, DMT_FOLDER, V1, ET_RES_FOLD)

    # load the behavioral data from the relevant phase file
    beh_data_all = load_beh_data(beh_data_path)
    beh_data_qc_path = os.path.join(beh_data_path, f"{BEH_QC_FILE_NAME}_{phase_name}.csv")
    beh_data_qc_file = pd.read_csv(beh_data_qc_path)
    # take only valid subjects from this file (those who passed the behavioral qc)
    sub_beh_data_qc = beh_data_qc_file.loc[beh_data_qc_file[BEH_QC_VALID_COL] == True, :]

    """
    As ECoG patients are valuable, it was decided in Paris to *NOT* exclude patients EVEN IF THEIR BEHAVIOR IS INVALID.
    Therefore, if we are in phase 3, we need to RE-INCLUDE the ECoG subjects that were just now excluded. 
    """
    if phase_name == "phase3":
        excluded_ecog_subs = beh_data_qc_file[(beh_data_qc_file[BEH_QC_VALID_COL] != True) & (beh_data_qc_file["modality"] == ECoG)]
        # re-include them - take them as valid (but first print their names)
        print("WARNING: these iEEG patients' behavior is INVALID, but they will be included in the analysis anyway")
        print(excluded_ecog_subs.loc[:, BEH_QC_SUB_COL].tolist())
        excluded_ecog_subs.to_csv(os.path.join(qc_path, "quality_checks_phase3_included_invalid_ecog.csv"), index=False)
        sub_beh_data_qc = sub_beh_data_qc.append(excluded_ecog_subs)

    # modality dictionary
    mod_valid_dict = {mod: sorted(list(sub_beh_data_qc.loc[sub_beh_data_qc["modality"] == mod, BEH_QC_SUB_COL])) for mod in MODALITY_MAPPING.keys()}
    # counts all the excluded subjects
    mod_exclusion = {mod: len(list(beh_data_qc_file.loc[beh_data_qc_file["modality"] == mod, BEH_QC_SUB_COL])) - len( mod_valid_dict[mod]) for mod in MODALITY_MAPPING.keys()}

    # Now, let's either QC or process for analysis
    if not straight_to_qc:
        # we are doing ET QC from the top:
        for mod in MODALITY_MAPPING.keys():
            print(f"-------- {mod} --------")
            mod_result_path = os.path.join(root_folder, mod, PHASE, PROCESSED, BIDS, DERIV, QCS)
            current_path = os.path.join(root_folder, mod, RAW, PROJ, f"CoG_{mod}_PhaseII")
            if not os.path.exists(current_path):
                print(f"No {mod} data found; moving on to next modality")
                continue
            for sub_dir in mod_valid_dict[mod]:
                sub_path = os.path.join(current_path, sub_dir)
                sub_name = sub_dir
                if not os.path.isdir(sub_path):
                    print(f"ERROR: Subject {sub_name} has no data in {current_path}")
                    continue
                sub_v1 = os.path.join(sub_path, f"{sub_name}_{MODALITY_MAPPING[mod]}_V1")
                if os.path.isdir(sub_v1):
                    resources_dir = os.path.join(sub_v1, RESOURCES)
                    if os.path.isdir(resources_dir):
                        et_path = os.path.join(resources_dir, ET)
                        if not os.path.exists(et_path):
                            print(f'Sub {sub_name}: No ET data! Skipped')
                            continue
                        # get a list of asc files
                        ascFiles = [fl for fl in os.listdir(et_path) if os.path.isfile(os.path.join(et_path, fl)) and fl.endswith(ASC)]
                        if len(ascFiles) == 0:
                            print(f'-------- Sub {sub_name}: No ascii files found! Skipped -------- Sub ')
                            continue
                        ascFiles.sort()  # sort the list in place such that the block order is increasing
                        # subject exists and has all data; create a result folder for the ET qc
                        sub_result_path = os.path.join(mod_result_path, f"{SUB_PREFIX}{sub_name}", SES_V1, ET_RES_FOLD)
                        if not os.path.exists(sub_result_path):
                            os.mkdir(sub_result_path)
                        # get subject's behavioral data
                        sub_beh_data = beh_data_all[sub_name]
                        p = Process(target=ET_data_extraction.extract_data,
                                    args=(sub_name, et_path, ascFiles, sub_result_path, sub_beh_data))
                        p.start()  # start the process of QC for this subject
                        p.join()  # this blocks until the process terminates, no new sub will run until the previous sub ended
                        if p.exitcode > 0:  # if the process exited with an error
                            exit(1)  # exit with an error as well

            print(f"{mod_exclusion[mod]} subjects were skipped as they were previously excluded in behavioral quality-checks")

    print(f'\n-------- Data Extraction Complete ----------\n')

    subs = {FMRI: list(), MEG: list(), ECoG: list()}
    unprocessed_subs = list()  # This is a list of subjects whose ET data IS INVALID so a pickle wasn't even generated for analysis
    processed_subs = list()
    for mod in MODALITY_MAPPING.keys():
        mod_result_path = os.path.join(root_folder, mod, PHASE, PROCESSED, BIDS, DERIV, QCS)
        if not os.path.exists(mod_result_path):
            print(f"No {mod} data found; moving on to next modality")
            continue
        for sub_dir in mod_valid_dict[mod]:  # iterate over PROCESSED data
            sub_result_path = os.path.join(mod_result_path, f"sub-{sub_dir}", SES_V1, ET_RES_FOLD)
            sub_name = sub_dir.replace(SUB_PREFIX, '')
            if not os.path.exists(sub_result_path):
                print(f"{sub_name} has no et folder")
                unprocessed_subs.append(sub_name)
                continue
            pick_file = [f for f in os.listdir(sub_result_path) if fnmatch.fnmatch(f, f"{sub_name}EyeTrackingData.pickle")]
            if len(pick_file) == 0:
                print(f"{sub_name} has no ET pickle file")
                continue
            processed_subs.append(sub_name)
            print(f"Found subject {sub_name} saved data. Loading now...")
            pick_path = os.path.join(sub_result_path, pick_file[0])
            subs[mod].append(pick_path)

    # save into a file all the subjects that were NOT PROCESSED AT ALL as their ET was too problematic
    unprocessed_subs_df = pd.DataFrame({"subCode": unprocessed_subs})
    unprocessed_subs_df.to_csv(os.path.join(qc_path, "et_invalid_subs.csv"), index=False)

    processed_subs_df = pd.DataFrame({"subCode": processed_subs})
    processed_subs_df.to_csv(os.path.join(qc_path, f"et_valid_subs_{phase_name}.csv"), index=False)

    # now we have all the existing subjects in all modalities
    p = Process(target=ET_data_processing.process_data, args=(subs, beh_data_path, qc_path, phase_name, False))
    p.start()
    p.join()
    if p.exitcode > 0:  # if the process exited with an error
        exit(1)  # exit with an error as well

    return


if __name__ == '__main__':
    qc_ET(beh_data_path=r"/mnt/beegfs/XNAT/COGITATE/QC/v1/beh",
          straight_to_qc=True, root_folder=r"/mnt/beegfs/XNAT/COGITATE", phase_name="phase3")


