""" This script annotates the bids channels tsv file
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import pandas as pd
import math
from mne_bids import BIDSPath
from general_helper_functions.pathHelperFunctions import find_files


def channel_annot_to_bids(channel_annot_file, bids_root, subjects=None, session="V1", data_type="ieeg", task="task"):
    """
    This function reads in an excel file in which some electrodes of the data set are described if they have something
    particular such as epileptic onset, noisy... The idea is that one can manually create an excel file to take notes
    about specific channels. This function then populates the bids channels.tsv file under status_description to
    enable parsing of the bids in the preprocessing.
    :param channel_annot_file: (path) path to the csv table containing the channels descriptions
    :param bids_root: (path) path to the bids root
    :param session: (string) name of the session in which to document
    :param data_type: (string) data type as described by bids
    :param task: (string) name of the task or interest
    :param subjects: (string or list) if all, run on all subjects. Otherwise, run only on subjects passed!
    :return:
    None: saves the status_description to channels.tsv files
    """
    # Loading the channels annotation:
    channel_annotation = pd.read_csv(channel_annot_file)

    # Looping through each participant in the channel annotation file:
    for subject in channel_annotation["subject"].unique():
        if subject is not None:
            if subject not in subjects:
                continue
        # Extracting only that one subject channel annotation:
        subject_channel_annot = channel_annotation.loc[channel_annotation["subject"] == subject]
        if len(subject_channel_annot) == 0:
            print("WARNING: There were no channels annotation for sub-" + subject)
            continue
        # Generating the bids path:
        bids_path = BIDSPath(root=bids_root, subject=subject,
                             session=session,
                             datatype=data_type,
                             task=task)
        # Looking for the channel file:
        channel_tsv_file = find_files(bids_path.directory, naming_pattern="*_channels", extension=".tsv")
        if len(channel_tsv_file) == 0:
            print("WARNING: No channel tsv found for sub-{}!".format(subject))
            continue
        assert len(channel_tsv_file) == 1, "The number of *_channels.tsv does not match what is expected!"
        # Loading the channels tsv:
        channel_tsv = pd.read_csv(channel_tsv_file[0], sep="\t")
        # Making sure that all the annotated channels exist in the bids file:
        non_existing_channels = [channel
                                 for channel in subject_channel_annot["channel"].to_list()
                                 if channel not in channel_tsv["name"].to_list()]
        if len(non_existing_channels) > 0:
            raise Exception("The following channels do not exist in sub-{} channels.tsv:"
                            "\n{}"
                            "\nCheck spelling!".format(subject, non_existing_channels))

        # We can now loop through the annotated channels to extend the annotations:
        for channel in subject_channel_annot["channel"]:
            # Extract the new annotation:
            ch_new_annots = subject_channel_annot.loc[subject_channel_annot["channel"] == channel]
            for ind, new_annot in ch_new_annots.iterrows():
                # Find whether there is already an existing annotation for this channel:
                tsv_annotation = channel_tsv.loc[channel_tsv["name"] == channel, "status_description"].item()
                # Checking whether the annotation from the one file is already present in the tsv annotation:
                if isinstance(tsv_annotation, float) and math.isnan(tsv_annotation):
                    ch_annot = new_annot["description"]
                elif new_annot["description"] in tsv_annotation:
                    print("Channel {} was already annotated as {} in the bids channels tsv. It will be skipped".format(
                        channel, new_annot["description"]))
                    continue
                else:
                    # Otherwise, appending the new annotation to the existing one:
                    ch_annot = "/".join([tsv_annotation, new_annot["description"]])
                # And editing the channels tsv:
                channel_tsv.loc[channel_tsv["name"] == channel, "status_description"] = ch_annot

        # The channels tsv can now be saved again:
        channel_tsv.to_csv(channel_tsv_file[0], index=False, sep="\t")


if __name__ == "__main__":
    channel_annot_to_bids("/hpc/users/alexander.lepauvre/sw/github/ECoG/general_helper_functions/bad_channels_annotation.csv", "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids",
                          subjects="all", session="V1", data_type="ieeg", task="task")