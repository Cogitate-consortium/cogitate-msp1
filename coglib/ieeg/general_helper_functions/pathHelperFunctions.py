""" This script contains several functions to handle path and directory easily
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Simon Henin
    Simon.Henin@nyulangone.org
"""
import os
import glob
import numpy as np
from pathlib import Path
import pandas as pd


def find_files(root, naming_pattern=None, extension=None):
    """
    This function finds files matching a specific naming pattern recursively in directory from root
    :param root: root of the directory among which to search. Must be a string
    :param naming_pattern: string the files must match to be returned
    :param extension: Extension of the file to search for
    :return:
    list of files
    """
    if extension is None:
        extension = '.*'
    if naming_pattern is None:
        naming_pattern = '*'

    matches = []
    for sub_folder, dirnames, filenames in os.walk(root):
        for filename in glob.glob(sub_folder + os.sep + '*' + naming_pattern + '*' + extension):
            matches.append(os.path.join(sub_folder, filename))
    # XNAT will mess up the order of files in case there was an abortion, because it will put things in folders called
    # ABORTED. Therefore, the files need to be sorted based on the file names:
    # Getting the file names:
    matches_file_names = [file.split(os.sep)[-1] for file in matches]
    files_order = np.argsort(matches_file_names)
    matches = [matches[ind] for ind in files_order]
    [print(match) for match in matches]

    return matches


def path_generator(root, analysis=None, preprocessing_steps=None, fig=False, stats=False, data=False):
    """
    Generate the path to where the data should be saved
    :param root: (string or pathlib path object) root of where the data should be saved
    :param analysis: (string) name of the analysis. The highest level folder for the saving will be called accordingly
    :param preprocessing_steps: (string) description of the preprocessing steps used to generate the used data to keep
    track of things
    :param fig: (boolean) whether or not the path is for saving figures. If set to false, the stats should be set to
    true
    :param stats: (boolean) whether or not the path is for saving statistics. If set to false, the fig should be set
    to true
    :param data: (boolean) whether or not the path is for saving data. If set to false, the fig should be set
    to true
    :return: save_path (Pathlib path object) path to where the data should be saved
    """

    if fig is True and stats is False and data is False:
        save_path = Path(root, "figure", analysis, preprocessing_steps)
    elif fig is False and stats is True and data is False:
        save_path = Path(root, "results", analysis, preprocessing_steps)
    elif fig is False and stats is False and data is True:
        save_path = Path(root, "data", analysis, preprocessing_steps)
    else:
        raise Exception("You attempted to generate a path to save the analysis specifying that it'S neither stats nor "
                        "figure. that doesn't work. Make sure that only one of the two is true")
    # Generating the directory if that doesn't exist
    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    return save_path


def get_subjects_list(bids_path, analysis_name):
    """
    This function parses the bids participants tsv to find the list of subjects for a given analysis
    :param bids_path: (string or Pathlib path object) path to the bids directory root
    :param analysis_name: (string) name of the analysis. Options:
        "visual_responsiveness"
        "category_selectivity"
        "activation_analysis"
        "synchrony"
        "decoding"
        "rsa"
    :return: (list of strings) list of the subjects to use for the given analysis.
    NOTE: this scripts relies on what is in the participants.tsv. In our project, the participants.tsv is updated
    as we get new subjects.
    """
    # Load the participants_tsv:
    participants_tsv = pd.read_csv(Path(bids_path, "participants.tsv"), sep="\t")
    # Make sure that the analysis is one of the column
    assert analysis_name in participants_tsv, "The analysis {} is not found in the participants.tsv".format(
        analysis_name)
    # Get the list of subjects:
    subjects_list = participants_tsv.loc[participants_tsv[analysis_name], "participant_id"].to_list()
    # Remove the sub-:
    subjects_list = [subject.replace("sub-", "") for subject in subjects_list]
    # Return the subjects list:
    return subjects_list
