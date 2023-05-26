""" This script generates the laplacing mapping in a semi automated way. The remapping is determined based on
    channels names and then the user is prompted to adjust the generated file manually if needed
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Simon Henin
    Simon.Henin@nyulangone.org
"""

from general_helper_functions.data_general_utilities import list_subjects
from general_helper_functions.pathHelperFunctions import find_files
from pathlib import Path
from mne_bids import BIDSPath
import pandas as pd
import re
import json


def semi_automated_laplace_mapping(channels):
    """
    This function generates a dictionary with one key per channel. And for each of these keys, there are two further
    keys: ref_1 and ref_2. This refers to the two references to use for the laplace mapping, which takes the two
    neighboring channels and compute their average to subtract from the one channel. Specifically, this function
    determines neighborhood by looking at the last number of the electrode name and doing +-1 to find the one
    prior and after the given one. This works well, as in ieeg, the convention is in most cases that the electrodes
    on a shaft are ordered by number. Note however that if you have electrode grids, this function won't handle
    things properly. It will indeed ignore the fact that you have corners and sides, and consider all the electrodes
    follow each other in space. So if you have an 8x8 grids, with names G1, G2, ... G64, it will consider that all of
    them are on one line. You will have to go an manually alter the mapping in the files, as there are no straight
    forward ways to deal with this programmatically!
    In other words, if you have an 8 by 8 grid, it would look something like this:
    G1  G2  G3  G4  G5  G6  G7  G8
    G9  G10 G11 G12 G13 G14 G15 G16
    G17  G18 G19 G20 G21 G22 G23 G24
    ...
    So you will want your mapping to look something like this:
    {"G1": {"ref_1": None, "ref_2": "G2"},
    "G2": {"ref_1": "G1", "ref_2": "G3"},
    "G3": {"ref_1": "G2", "ref_2": "G4"},
    ...
    "G7": {"ref_2": "G6", "ref_2": "G8"},
    "G8": {"ref_2": "G7", "ref_2": None},
    "G9": {"ref_2": None, "ref_2": "G10"},
    ....
    }
    Because the grid is a square of 8 by 8 electrodes, meaning that when you reached the 8th (or any multiple thereof)
    electrode in a row, you have reached the end of a grid. So electrode 9 won't be right after 8. The function
    will by default ignore this and do the mapping like so:
        {"G1": {"ref_1": None, "ref_2": "G2"},
    "G2": {"ref_1": "G1", "ref_2": "G3"},
    "G3": {"ref_1": "G2", "ref_2": "G4"},
    ...
    "G7": {"ref_2": "G6", "ref_2": "G8"},
    "G8": {"ref_2": "G7", "ref_2": "G9"},
    "G9": {"ref_2": "G8", "ref_2": "G10"},
    ....
    }
    So if you have grids, you MUST go to the file and adjust it manually.
    :param channels: (list of strings) list of channels names
    :return: laplace_mapping: dict(channel_n: {ref_1_: channel_n-1, ref_2: channel_n+1})
    """
    # First, create a dict of dict, for each channel:
    laplace_mapping = {ch: {"ref_1": None, "ref_2": None} for ch in channels}
    # Now, looping through each channel to figure out the ref 1 and 2:
    for ch in laplace_mapping.keys():
        # Parsing the numbers out of the channel name:
        ch_num = re.findall(r'\d+', ch)[-1]
        ch_string = ch.replace(ch_num, "")
        # Look for the first reference:
        if ch_string + str(int(ch_num) - 1) not in channels:
            laplace_mapping[ch]["ref_1"] = None
        else:
            laplace_mapping[ch]["ref_1"] = ch_string + str(int(ch_num) - 1)
        # Look for the second reference:
        if ch_string + str(int(ch_num) + 1) not in channels:
            laplace_mapping[ch]["ref_2"] = None
        else:
            laplace_mapping[ch]["ref_2"] = ch_string + str(int(ch_num) + 1)

    return laplace_mapping


def run_laplace_mapping(bids_root=None, session="V1", data_type="ieeg", task_name="Dur"):
    """
    This script enables to do automated laplace mapping for existing participants. This function loops
    through the existing participants to generate the laplace mapping if it doesn't exist already.
    :param bids_root: (Path object or string) root to the bids raw data set
    :param session: (string) session of the data for which the laplace mapping should be generated
    :param data_type: (string) type of bids data
    :param task_name: (string) name of the task to consider
    :return: None
    """
    # Find all existing subjects:
    subjects_list = list_subjects(bids_root)
    # Pregenerate the full path to the laplace mapping:
    laplace_mapping_file = "sub-{0}_ses-" + session + "_laplace_mapping_" + data_type + ".json"
    # Loop through each subject:
    for sub in subjects_list:
        # Creating the bids path object:
        bids_path = BIDSPath(root=bids_root, subject=sub,
                             session=session,
                             datatype=data_type,
                             task=task_name)
        # Create the file name:
        sub_laplace_mapping = Path(bids_path.directory, laplace_mapping_file.format(sub))
        if sub_laplace_mapping.is_file():
            print("The laplace mapping for sub-{0} already exists here {1}".format(sub, sub_laplace_mapping))
        else:
            # Find the channels.tsv file:
            channels_file = find_files(bids_path.directory, naming_pattern="*channels", extension=".tsv")[0]
            # Load the file:
            channels_tsv = pd.read_csv(channels_file, sep="\t")
            # Generate laplace mapping:
            try:
                laplace_mapping = semi_automated_laplace_mapping(channels_tsv.loc[channels_tsv["type"].isin(
                    ["ECOG", "SEEG"]), "name"].to_list())
            except:
                print("ERROR: The laplace semi automated mapping failed for sub-" + sub)
                continue
            # Save to file:
            with open(sub_laplace_mapping, "w") as fp:
                json.dump(laplace_mapping, fp, indent=4)
            input(
                "CAREFUL: you have used the automated channels mapping! This is a semi automated way! You should go and"
                "\nmanually check that the mapping was set correctly in the file {0}."
                "\nPress enter to continue".format(laplace_mapping_file))
    return None


if __name__ == "__main__":
    run_laplace_mapping(bids_root="/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids",
                        session="V1", data_type="ieeg", task_name="Dur")
