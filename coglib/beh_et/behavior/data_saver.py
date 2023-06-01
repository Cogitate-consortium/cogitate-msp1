import os
import data_reader

"""
This module includes all the data-saving related methods. It creates the directories where the data will be saved 
and has methods for saving files into the appropriate folders. Every module which saves analysis / 
QC data uses this module. 

@authors: AbdoSharaf98, RonyHirsch
"""

QC_NAME = "BEH_QC_summary"
ANALYSIS_NAME = "BEH_analysis"
SPECIFIC_ANALYSIS_NAME = "BEH_specific_analysis"

HPC_QC_FOLDER = "QC"
V1 = "v1"
PHASE2 = "phase_2"
QCS = "qcs"
SESS_V1 = f"ses-{V1}"
BEH = "beh"
QCS_PREFIX = "processed/bids/derivatives/qcs"



def create_dir(d):
    """
    This function creates a directory, given a specific path.
    If the directory didn't exist, the function creates it. If it cannot be created,an error message is printed.
    If it already exists, the function does nothing.
    :param d: directory path
    """
    if not (os.path.isdir(d)):
        try:
            os.mkdir(d)
        except Exception as e:
            warning_msg = ' \nCould not create directory ' + str(d)
            print(warning_msg)
            raise e
        else:
            print('Created directory ' + str(d))


def build_file_hierarchy(*dirs):
    """
    Creates directories specified in input, if they do not exist.
    :param dirs: specify an unlimited number of directories to create as strings. Do not pass a list of strings. Pass
    each directory you wish to create as a separate parameter. The order of directories specified must move down the
    file hierarchy - for example, if you want to create a folder named 'folderA', and another named 'folderB' within
    'folderA', you must pass the directories in the following order: build_file_hierarchy('/folderA','/folderA/folderB')
    :return: Nothing. The function creates directories in the order specified. If the directory cannot be made,
    a warning message is printed and the next directory is attempted. If the directory is created successfully,
    a message is printing noting that the directory was created. If the directory already exists, nothing is done.
    """
    for d in dirs:
        create_dir(d)


def create_quality_checks(path_to_data_folder):
    qc_dir = os.path.join(path_to_data_folder, QC_NAME)
    if not os.path.isdir(qc_dir):
        create_dir(qc_dir)
    return qc_dir


def create_analysis(path_to_data_folder):
    analysis_dir = os.path.join(path_to_data_folder, ANALYSIS_NAME)
    if not os.path.isdir(analysis_dir):
        create_dir(analysis_dir)
    difficulty_dir = os.path.join(analysis_dir, "difficulty")  # difficulty analysis will go here
    performance_dir = os.path.join(analysis_dir, "performance")  # performance analysis will go here
    diff_per_resp_dir = os.path.join(difficulty_dir, "per_response_type")  # difficulty per response type will go here
    diff_across_game_dir = os.path.join(difficulty_dir, "across_game")  # difficulty across the game will go here
    perf_per_resp_dir = os.path.join(performance_dir, "per_response_type")  # performance per response will go here
    perf_across_game_dir = os.path.join(performance_dir, "across_game")  # performance across the game will go here
    perf_v_diff_dir = os.path.join(analysis_dir, "performance_vs_difficulty")  # diff vs. perf analysis will go here
    resp_rate_dir = os.path.join(analysis_dir, "response_type_rates")  # diff vs. perf analysis will go here
    game_features_dir = os.path.join(analysis_dir, "vg_features")
    try:
        build_file_hierarchy(analysis_dir, difficulty_dir, performance_dir, diff_per_resp_dir, diff_across_game_dir
                             , perf_per_resp_dir, perf_across_game_dir, perf_v_diff_dir, resp_rate_dir, game_features_dir)
    except Exception as e:
        err_msg = 'termination because of inability to create file hierarchy necessary for analysis.'
        raise Exception(err_msg) from e
    return analysis_dir


def create_specific_analysis(path_to_data_folder):
    if not os.path.isdir(path_to_data_folder):
        create_dir(path_to_data_folder)
    sa_dir = os.path.join(path_to_data_folder, SPECIFIC_ANALYSIS_NAME)
    world_dir = os.path.join(sa_dir, "per_world")
    level_dir = os.path.join(sa_dir, "per_level")
    loc_dir = os.path.join(sa_dir, "per_location")
    stim_dir = os.path.join(sa_dir, "per_stim")
    try:
        build_file_hierarchy(sa_dir, world_dir, level_dir, loc_dir, stim_dir)
    except Exception as e:
        err_msg = 'termination because of inability to create file hierarchy necessary for analysis.'
        raise Exception(err_msg) from e
    return sa_dir


def safe_save(folder_path, file_name):
    """
    Checks whether the folder path + file name combo already exist in folder. This function is called right before
    saving data to an IDENTICAl combo, so if a file like that is found it's DELETED, to be replaced by the new file
    that we are trying to save.
    Meaning, if the folder path + file name exist - the file is DELETED and then replaced by a file with an identical
    name.
    :param folder_path: path where the file is saved
    :param file_name: file name
    """
    existing_files = [f for f in os.listdir(folder_path)]
    is_duplicate = 0
    if len(existing_files) == 0:
        print(f"No files in directory {folder_path}")
        return
    for f in existing_files:
        if f == file_name:
            is_duplicate = 1
    if is_duplicate == 1:
        print(f"File {f} already exists in {folder_path}. Save action is replacing the old file with a new one.")
        # remove the old file
        os.remove(os.path.join(folder_path, file_name))
    else:
        print(f"File {f} was not found in {folder_path}. Save action will not override anything.")
    return


def create_sub_qcs_hpc(sub_name, modality, root_folder=data_reader.COGITATE_PATH):
    qc_path = f"{root_folder}/{modality}/{PHASE2}/{QCS_PREFIX}"
    if not os.path.exists(qc_path):
        print(f"no qcs directories in {root_folder}/{modality}/{PHASE2}")
        return
    sub_folder = os.path.join(qc_path, f"sub-{sub_name}")
    create_dir(sub_folder)
    ses_folder = os.path.join(sub_folder, f"{SESS_V1}")
    create_dir(ses_folder)
    beh_folder = os.path.join(f"{ses_folder}/{BEH}")
    create_dir(beh_folder)
    return beh_folder


def create_hpc_quality_checks(root_folder):
    qc_dir = os.path.join(root_folder, HPC_QC_FOLDER, V1, BEH)
    if not os.path.isdir(qc_dir):
        create_dir(qc_dir)
    return qc_dir
