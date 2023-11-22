"""Convert to BIDS using mne."""
import math
from pathlib import Path
import numpy as np
from collections import OrderedDict
import pandas as pd
import mne
from mne.datasets import fetch_fsaverage
from nibabel.freesurfer.io import read_geometry
from mne_bids import (write_raw_bids, BIDSPath)
from general_helper_functions.pathHelperFunctions import find_files


def channels_desc_validator(channels_desc_file, raw):
    """
    This function checks a couple of things about the channel descriptions file
    :param channels_desc_file: (pathlib path object or string) path to the channels_description file
    :param raw: (mne raw object) contains info about the data, importantly the channels names
    :return:
    """
    mand_col = ["name", "status", "status_description"]
    if channels_desc_file.endswith(".tsv"):
        # Loading the file:
        channel_desc = pd.read_csv(channels_desc_file, sep="\t")
        # Checking that the required columns are present
        if all([True if col in channel_desc.columns else False for col in mand_col]):
            # Check that the channels in the channel description file are present in the data being considered:
            if all([True if channel in raw.ch_names else False for channel in channel_desc["name"].to_list()]):
                return channel_desc
            else:
                raise Exception("Some of the channels in the channels_description_file do not match the channels found "
                                "in the data")
        else:
            raise Exception(
                "The channel description file must contain the columns {0}!".format(mand_col))
    else:
        raise Exception("The file you have passed as channel_description_file is of format {0} instead of {1}".format(
            Path(channels_desc_file).suffix, ".tsv"))


def create_channels_descriptions(raw, channels_description_file=None):
    """
    This function enables to manually define descriptions for single channels or load them from existing file.
    For the manual description, for each channel, you can add a status (good or bad) and a description (string
    describing the channel in whatever way you see fit). Not that if  for a given channel, you set more than 1
    description in different iterations, each description will be separated by a forward slash. CAREFUL:
    if you add descriptions several times for the same channel, the last stated status (good or bad) will be the one
    retained for that channel!
    :param raw: (mne raw object) contains the data to annotate. Used to make sure that the channels for which you want
    to set a description do exist!
    :param channels_description_file: (str or Pathlib path object) path to the file describing the electrodes:
    :return: channels_descriptions (dict) for each channel you want to give a description, contains a key for status
    and a key for description. So the dict format is:
    {"G1":
        "status": "good",
        "description": "epileptic_onset"
    ...
    }
    """
    print("Existing channels")
    print(raw.ch_names)
    channels_descriptions = {}
    if channels_description_file is None:
        add_descriptions = True
        # While loop to make several iterations for different annotations
        while add_descriptions:
            # Ask for the channels to add. Can be either several or only one:
            channels_to_describe = input(
                "Enter a single channel or a list of channels (comma separated) you would like "
                "to describe or leave empty if you want to proceed:")
            # If a channel was passed:
            if len(channels_to_describe) > 0:
                # While loop to set the channel type to something that is supported (good or bad)
                supported_status = False
                while not supported_status:
                    channels_status = input(
                        "Select status to give a channel good, bad:").lower()
                    if channels_status != "good" and channels_status != "bad":
                        print(
                            "You have passed a channel status that is not supported. You must give either good or bad")
                        supported_status = False
                    else:
                        supported_status = True
                # Ask channel description
                channels_status_description = input(
                    "What description would you like to give to the channels you have just "
                    "marked?")
                # Parse the passed channels:
                parsed_channels = channels_to_describe.replace(
                    " ", "").split(",")
                # Checking the passed channels against the existing channels:
                wrong_channels = any(
                    [ch not in [channel.replace(" ", "") for channel in raw.ch_names] for ch in parsed_channels])
                if wrong_channels:
                    print(
                        "You have passed channel names that do not exist for this participant! \n "
                        "Please give new ones!")
                    continue
                else:
                    # Create the dict of channels description for this current iteration:
                    current_desc = {ch: {"status": channels_status, "status_description": channels_status_description}
                                    for ch in parsed_channels}
                    # Checking if description for each channel already exists from a previous iteration:
                    existing_keys = [channel for channel in current_desc.keys() if
                                     channel in channels_descriptions.keys()]
                    # Looping through each channel that was passed in this iteration:
                    for channel in channels_descriptions:
                        # If this channel was not described in previous iteration:
                        if channel not in existing_keys:
                            # Add the channel description to the dictionary of channel description
                            channels_descriptions[channel] = channels_descriptions[channel]
                        else:
                            # If this channel was already described in previous iterations AND that the same description
                            # was already given, skip it
                            if current_desc[channel]["status_description"] in \
                                    channels_descriptions[channel]["status_description"]:
                                print(
                                    "{0} was already described as {1} and will be ignored".format(
                                        channel, current_desc[channel]["status_description"])
                                )
                            else:
                                # If this is a new description for a channel that already had another description,
                                # append it to the existing  description with forward slash
                                channels_descriptions[channel]["status_description"] = \
                                    "/".join([channels_descriptions[channel]["status_description"],
                                              current_desc[channel]["status_description"]])
                                # Update channel status:
                                if channels_descriptions[channel]["status"] == "bad":
                                    print(
                                        "{0} status was set as bad in a previous iteration and will therefore remain "
                                        "as such!".format(channel))
                                else:
                                    channels_descriptions[channel]["status"] = channels_status
                                    print("{0} status was set as good in a previous iteration and will now be set to "
                                          "{1}".format(channel, channels_status))
                    # Removing these keys from the current desc:
                    [current_desc.pop(key) for key in existing_keys]
                    # And now appending:
                    channels_descriptions.update(current_desc)
                # Continue while loop for new iteration
                add_descriptions = True
            else:  # if no channels were passed, terminate iteration:
                add_descriptions = False
    else:
        # Loading and validating the channels description file:
        channels_description_from_file = channels_desc_validator(
            channels_description_file, raw)
        # Convert the channels descriptions to a dictionary to keep format:
        channels_descriptions = {}
        for ind, row in channels_description_from_file.iterrows():
            channels_descriptions[row["name"]] = {"status": row["status"],
                                                  "status_description": row["status_description"]}

    return channels_descriptions


def write_channels_description_bids(channels_descriptions, bids_path):
    """
    This function adds channels description to the bids _channels.tsv
    :param channels_descriptions: (dict) contains information about the channels for which a description should be
    added. The format is as follows: {"channel_name": {"status": "good", "status_description": "epileptic_onset"}}. The
    function create_channels_descriptions enables to programmatically generate the dictionary with user input
    :param bids_path: (mne_bids path object) contains path to the bids channels info file
    :return:
    """
    # We can now update the bids data to add the description:
    # Load the channel info bids file:
    channel_info_file = find_files(
        bids_path.directory, naming_pattern="*_channels", extension=".tsv")[0]
    channel_info = pd.read_csv(channel_info_file, sep="\t")
    # Now looping through each channel that was marked:
    for channel in channels_descriptions.keys():
        channel_info.loc[channel_info["name"].str.replace(" ", "") == channel, "status"] = \
            channels_descriptions[channel]["status"]
        channel_info.loc[channel_info["name"].str.replace(" ", "") == channel, "status_description"] = \
            channels_descriptions[channel]["status_description"]
    # Saving the channel info again:
    channel_info.to_csv(channel_info_file, sep='\t')

    return None


def remove_elecname_leading_zero(electrodes_list):
    """
    This function takes the electrodes names as found in the electrode reconstruction files and if the number of the
    electrodes has a leading 0, it gets removed, as they are not found in teh raw data!
    :param electrodes_list: the ordered dict of the functions below
    :return: electrode_tsv: with changed name!
    """

    # Setting the counter
    ctr = 0
    # declare variable to store electrodes without leading 0
    elec_name_no_zero = []
    # Looping through each electrodes
    for elec_names in electrodes_list:
        # Parsing the electrode names in digits and letters
        elec_digit = [int(i) for i in list(elec_names) if i.isdigit()]
        elec_letter = [i for i in list(elec_names) if i.isalpha()]

        # Then joining the letters of the electrodes names!
        elec_letter = ''.join(elec_letter)
        # Now, removing the leading 0 from electrodes numbers:
        elec_digit = str(int(''.join([str(x) for x in elec_digit])))

        # Recreating the channel names by putting things back together:
        elec_name_no_zero.append(elec_letter + elec_digit)

        # Updating the counter:
        ctr = ctr + 1

    return elec_name_no_zero


def project_elec_to_surf(raw, subjects_dir, subject, montage_space="T1"):
    """
    This function project surface electrodes onto the brain surface to avoid having them floating a little.
    :param raw: (mne raw object)
    :param subjects_dir: (path or string) path to the freesurfer subject directory
    :param subject: (string) name of the subject
    :param montage_space: (string) montage space. If T1, then the projection will be done to the T1 scan of the subject
    If MNI, will be done to fsaverage surface.
    :return:
    """
    # Loading the left and right pial surfaces:
    if montage_space == "T1":
        left_surf = read_geometry(Path(subjects_dir, subject, "surf", "lh.pial"))
        right_surf = read_geometry(Path(subjects_dir, subject, "surf", "rh.pial"))
    elif montage_space == "MNI":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # Downloading the data if needed
        subject = "fsaverage"
        left_surf = read_geometry(Path(subjects_dir, subject, "surf", "lh.pial"))
        right_surf = read_geometry(Path(subjects_dir, subject, "surf", "rh.pial"))
    else:
        raise Exception("You have passed a montage space that is not supported! Either MNI or T1!")

    # Getting the surface electrodes:
    ecog_picks = mne.pick_types(raw.info, ecog=True)
    ecog_channels = [raw.ch_names[pick] for pick in ecog_picks]
    # Get the montage:
    montage = raw.get_montage()
    # Looping through each of these channels:
    for channel in ecog_channels:
        # Get the channel index
        ch_ind = montage.ch_names.index(channel)
        # Get the channel coordinates:
        ch_coord = montage.dig[ch_ind]["r"]
        # Checking if the coordinates are NAN:
        if math.isnan(ch_coord[0]):
            continue
        if ch_coord[0] < 0:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(left_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(left_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(left_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [left_surf[0][min_vert_ind, 0] * 0.001, left_surf[0][min_vert_ind, 1] * 0.001,
                 left_surf[0][min_vert_ind, 2] * 0.001]))
        else:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(right_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(right_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(right_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [right_surf[0][min_vert_ind, 0] * 0.001, right_surf[0][min_vert_ind, 1] * 0.001,
                 right_surf[0][min_vert_ind, 2] * 0.001]))
    # Adding the montage back to the raw object:
    raw.set_montage(montage, on_missing="warn")

    return raw


def save_to_BIDs(raw, elec_recon_file=None, bids_root=None, subject_id=None, session=None,
                 task=None, data_type=None, line_freq=None, site="CF", channels_description_file=None,
                 project_ecog_to_surf=True, raw_root=None):
    """
    This function creates a montage for the ECoG electrodes and save the data to BIDS (according to the tutorial found
    here: https://mne.tools/stable/auto_tutorials/misc/plot_ecog.html)
    :param raw: (mne raw object) MNE raw object to save. Must not be preloaded!!!
    :param elec_recon_file: (string or windows path) path and file name of the table containing info about the
    electrodes localization.
    :param bids_root: (string or windows path) root to the BIDS folder
    :param subject_id: (string) subject name or identifier
    :param session: (string) name of the session the data are from
    :param task: (string) name of the task that was ran
    :param data_type: (string) type of data: ecog, meeg, fmri and so on
    :param line_freq: frequency of the line noise (60 in the US, 50 in EU...)
    :param site: the format of the electrodes reconstruction file is slightly different in different sites and
    needs to be handled accordingly
    :param channels_description_file: (string) file containing the channels descriptions
    :param project_ecog_to_surf: (boolean) whether or not to project the ecog electrodes onto the surface. The
    electrodes are sometimes floating a little, in which case they can be projected to the nearest pial surface vertex
    :param raw_root: (string or path) root to the raw data
    :return: None: only save data
    """

    # ------------------------------------------------------------------------------------------------------------------
    if elec_recon_file is not None:
        montage = {}
        for coordinates_space in elec_recon_file.keys():
            # Formatting the electrodes reconstruction file:
            # Loading the electrodes coordinates in a data frame:
            if site == "CF":
                elec_coord_raw = np.genfromtxt(elec_recon_file[coordinates_space], dtype=str, delimiter=' ',
                                               comments=None, encoding='utf-8')
            elif site == "CE":
                elec_coord_raw = np.genfromtxt(elec_recon_file[coordinates_space], dtype=str, delimiter=',',
                                               comments=None, encoding='utf-8')
                # Keeping only the columns of interest
                elec_coord_raw = elec_coord_raw[1:, [0, 3, 4, 5]]
            elif site == "SG":
                elec_coord_raw = np.genfromtxt(elec_recon_file[coordinates_space], dtype=str, delimiter='\t',
                                               comments=None, encoding='utf-8')
                # Keeping only the columns of interest
                elec_coord_raw = elec_coord_raw[1:, [0, 1, 2, 3]]
            # Declaring the dict to store the electrodes location:
            electrode_tsv = OrderedDict()
            # Converting each column to a list:
            for i, name in enumerate(['name', 'x', 'y', 'z']):
                electrode_tsv[name] = elec_coord_raw[:, i].tolist()

            # At NYU, the output of the electrodes reconstruction adds a 0 before some of the numbers,
            # which don't exist in the edfs. They therefore need to be removed:
            if site == "CF":
                electrode_tsv['name'] = remove_elecname_leading_zero(
                    electrode_tsv['name'])
            # Get the channels name
            ch_names = electrode_tsv['name']
            # load in the xyz coordinates as a float
            elec = np.empty(shape=(len(ch_names), 3))
            for ind, axis in enumerate(['x', 'y', 'z']):
                elec[:, ind] = list(map(float, electrode_tsv[axis]))
            # Converting the elec position to meters as required by the coordinate system
            elec = elec / 1000.
            # --------------------------------------------------------------------------------------------------------------
            # Preparing the raw file for saving:
            # Making the montage:
            if coordinates_space == "T1":
                # Making the montage:
                montage[coordinates_space] = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                                                           coord_frame='mri')
            elif coordinates_space == "MNI":
                montage[coordinates_space] = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                                                           coord_frame='mni_tal')
            print('Created %s channel positions' % len(ch_names))
            print(dict(zip(ch_names, elec)))

    # Declaring the line noise, as required by BIDS:
    raw.info['line_freq'] = line_freq

    # If there were electrode recon files, saving the data with the montage:
    if elec_recon_file is not None:
        # If there were more than one type of elec recon coordinate systems (typically MNI and T1 spaces), saving the
        # data twice to save both coordinate systems. This is not super elegant, but this ensures that we have both
        # coordinates at hands
        for coordinates_space in montage.keys():
            raw.set_montage(montage[coordinates_space], on_missing='warn')

            # ----------------------------------------------------------------------------------------------------------
            # Separating the data based on the channel types. If there is EEG and iEEG in the signal, we need to save
            # them into separate files to be BIDS compliant, otherwise it will fail:
            channel_types = raw.get_channel_types()
            # Then getting the eeg channels specifically
            if 'seeg' in channel_types or 'ecog' in channel_types:
                # Creating an mne raw object with only the ieeg channels
                ieeg_raw = raw.copy().pick_types(ecog=True, seeg=True, ecg=True, emg=True)

                if project_ecog_to_surf:
                    # Generate the path to the data:
                    fs_root = Path(raw_root, subject_id, "{}_ECOG_V1".format(subject_id), "RESOURCES", "ElecRecon",
                                   "fs_recon")
                    if len(mne.pick_types(ieeg_raw.info, ecog=True)) > 0:
                        ieeg_raw = project_elec_to_surf(ieeg_raw, fs_root, subject_id, montage_space=coordinates_space)

                # Creating the BIDS path:
                bids_path = BIDSPath(subject=subject_id, session=session,
                                     task=task, datatype=data_type, root=bids_root)
                write_raw_bids(ieeg_raw, bids_path,
                               overwrite=True, format='auto')
                # Enable user to add channels descriptions:
                channels_description = create_channels_descriptions(ieeg_raw,
                                                                    channels_description_file=channels_description_file)
                if len(channels_description) > 0:
                    write_channels_description_bids(
                        channels_description, bids_path)

            # Saving the eeg channels separately:
            if 'eeg' in channel_types:
                # Getting the EEG channels:
                eeg_raw = raw.copy().pick_types(eeg=True)
                # Creating the BIDSPath for the eeg data:
                bids_path_eeg = BIDSPath(subject=subject_id, session=session,
                                         task=task, datatype='eeg', root=bids_root)
                write_raw_bids(eeg_raw, bids_path_eeg,
                               overwrite=True, format='auto')
                # Enable user to add channels descriptions:
                channels_description = create_channels_descriptions(eeg_raw)
                if len(channels_description) > 0:
                    write_channels_description_bids(
                        channels_description, bids_path_eeg)
    else:
        # ------------------------------------------------------------------------------------------------------------------
        # Separating the data based on the channel types. If there is EEG and iEEG in the signal, we need to save them
        # into separate files to be BIDS compliant, otherwise it will fail:
        channel_types = raw.get_channel_types()
        # Then getting the eeg channels specifically
        if 'seeg' in channel_types or 'ecog' in channel_types:
            # Creating an mne raw object with only the ieeg channels
            ieeg_raw = raw.copy().pick_types(ecog=True, seeg=True, ecg=True, emg=True)

            # Creating the BIDS path:
            bids_path = BIDSPath(subject=subject_id, session=session,
                                 task=task, datatype=data_type, root=bids_root)
            write_raw_bids(ieeg_raw, bids_path, overwrite=True, format='auto')
            # Enable user to add channels descriptions:
            channels_description = create_channels_descriptions(ieeg_raw,
                                                                channels_description_file=channels_description_file)
            if len(channels_description) > 0:
                write_channels_description_bids(
                    channels_description, bids_path)

        # ------------------------------------------------------------------------------------------------------------------
        # Saving the eeg channels separately:
        if 'eeg' in channel_types:
            # Getting the EEG channels:
            eeg_raw = raw.copy().pick_types(eeg=True)
            # Creating the BIDSPath for the eeg data:
            bids_path_eeg = BIDSPath(subject=subject_id, session=session,
                                     task=task, datatype='eeg', root=bids_root)
            write_raw_bids(eeg_raw, bids_path_eeg,
                           overwrite=True, format='auto')
            # Enable user to add channels descriptions:
            channels_description = create_channels_descriptions(eeg_raw)
            if len(channels_description) > 0:
                write_channels_description_bids(
                    channels_description, bids_path_eeg)

    return None
