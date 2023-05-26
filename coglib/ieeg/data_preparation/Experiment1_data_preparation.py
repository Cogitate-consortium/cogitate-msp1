"""Data preparation for experiment 1."""
import os
import re
import glob
import tempfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

from Preprocessing.SubjectInfo import SubjectInfo
from general_helper_functions.pathHelperFunctions import find_files

from data_preparation.DataPreparationParameters import DataPreparationParameters
from data_preparation import trigger_alignment
from data_preparation.mne_bids_converter import save_to_BIDs
from data_preparation.trigger_alignment import select_start_and_end
from data_preparation.mne_bids_converter import remove_elecname_leading_zero


def find_correct_file(files_list):
    """
    This function reads the name of files and returns only those that do not have rh (right hemisphere) or lh (left
    hemisphere) in them, as in cases of bilaterly implant, there will be one file for electrodes of the one hemisphere
    another for the other hemisphere and a final one with both combined. we want only the latter
    :param files_list: list of files containing a date in them
    :return:
    """
    # Removing any file that contain the rh and lh names as we are interested in only the ones that have both:
    relevant_files_list = [
        file for file in files_list if "rh" not in file and "lh" not in file and "backup" not in file]

    return relevant_files_list[0]


def load_signal(root_path, file_naming_pattern, subject_info, file_extension=None, debug=False):
    """
    This function lists the EDF files in the path and loads them with MNE.
    NOTE: if you have several files in the directory, they will be loaded and concantenated to another!!!
    :param root_path: (string or path) Path to the root data. This function is recursive and will look for the data
    across the root directory down the structure. It will load all the files that have matching naming conventions to
    what is passed. Careful therefore not to have different files matching the naming convention in the directory that
    you don't want to load. In that case, you should give the root directory at a lower level to avoid loading spurious
    stuff.
    :param file_naming_pattern: (string) naming pattern of the file to load. String with wildcard accepted
    :param subject_info: (SubjectInfo custom object) object containing info about the specific participant running
    :param file_extension: (string) extension of the file, with the dot first!
    :param debug: true or false flag to be in debug mode or not. In debug mode, you will only load a few channels
    :return: raw: mne raw object containing the signal
    """

    # Priting info to let the user know it might take a while:
    print('-' * 40)
    print('The edf file(s) is/are being loaded, it might take a little while')

    data_files_list = find_files(
        root_path, naming_pattern=file_naming_pattern, extension=file_extension)
    if len(data_files_list) == 0:
        raise ReferenceError("No files with " + file_naming_pattern + " naming pattern were found in the root:"
                             + str(root_path))

    # Creating a dictionary to store the raws in:
    raws = {}

    # looping through the files and loading them:
    for ind, file in enumerate(data_files_list):
        full_file = file
        print('Now loading this file ', full_file)
        if debug:
            raws[ind] = mne.io.read_raw_edf(full_file,
                                            verbose='error', preload=False)
            # Dropping all channels except the trigger channel and the trigger reference channel
            channels_to_drop = [ch for ch in raws[ind].info['ch_names'] if
                                ch != subject_info.TRIGGER_CHANNEL and ch != subject_info.TRIGGER_REF_CHANNEL]
            raws[ind].drop_channels(channels_to_drop)
            # Loading only the few channels to cut down loading time
            raws[ind].load_data()

        else:
            raws[ind] = mne.io.read_raw_edf(full_file,
                                            verbose='error', preload=True)

    # Concatenating the files:
    # Instantiating the raw object
    raw = raws[0]

    # Looping through the dict to raws to append to the raw:
    for key in raws:
        # If we are at key 0 of the dict, then this is the first and it has already been add to raw
        if key == 0:
            pass
        else:
            # But if we are at keys other than 0, we concatenate
            mne.concatenate_raws([raw, raws[key]])

    return raw


def extract_pd_signal(raw, subject_info):
    """
    This function extracts the photodiode signal in an easy to manage format:
    :param subject_info: (SubjectInfo object) custom made object containing info about the participant
    :param raw: mne raw object
    :return: pd_signal: (dict) dictionary of np arrays with "time" key being the photodiode time in seconds, "amp" being
    the amplitude of the photodiode signal
    """
    # If there is a reference trigger channel, using it to correct PD amplitude:
    if subject_info.TRIGGER_REF_CHANNEL != '':
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
                   - np.array([raw.get_data(picks=subject_info.TRIGGER_REF_CHANNEL)[0]])
        })
    else:  # Otherwise, just extract the photodiode channel:
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
        })

    return pd_signal


def select_bad_channels(raw, debug, subject_info):
    """
    This function asks the user whether they want to select manually the bad channels and if yes, the signal is plotted
    using mne. The user should hand pick the badly channel (dead ones and so on, but leave DC)
    :param raw: mne raw structure containing the raw data
    :param debug: one or zero flag, if you are in debug mode you can't do that
    :param subject_info: (SubjectInfo object)
    :return: subject_info: updated participant info
    """
    exclude_hu_standard_channels = \
        input(
            "Do you want to exclude OSAT, Pleth, PR, TRIG and SpO2 if they exist? [Yes or No]")
    if exclude_hu_standard_channels == "Yes" or exclude_hu_standard_channels == "yes":
        if not debug:
            to_exclude_theory = ['OSAT', 'Pleth', 'PR', 'TRIG', 'SpO2']
            to_exclude_practice = [
                ch for ch in to_exclude_theory if ch in raw.ch_names and ch not in raw.info['bads']]
            raw.drop_channels(to_exclude_practice)

    return subject_info


def manual_adjust_threshold(subject_info, pd_signal):
    """
    This function enables the user to adjust the photodiode threshold manually, if it turns out to not be fitting this
    particular data set:
    :param pd_signal: photodiode signal
    :param subject_info: subject information object
    :return: subject_info updated
    """
    # Adjusting photodiode threshold:
    # Asking the user whether there are things they wish to change:
    pd_yes_no = input("The photodiode threshold is set to "
                      + str(subject_info.PD_THRESHOLD) + ". Would you like to modify it? [Yes or No]")

    # If the user wished to adjust the pd threshold, plotting the signal:
    if pd_yes_no == "Yes" or pd_yes_no == "yes":
        print("Close the figure to be able to enter the threshold.")
        fig = plt.figure(figsize=(8, 6))
        plt.plot(pd_signal["Amp"][0], 'g')
        plt.show()
        new_pd_threshold = input(
            "Type the photodiode threshold you wish to apply: ")
        subject_info.PD_THRESHOLD = float(new_pd_threshold)

    return subject_info


def identify_eyelink_grey_noise(subject_info, pd_signal):
    """
        The eyelink tracker has a calibration with grey screen, that is overlayed on the whole screen, polluting the
        photodiode flashes detection. The user therefore here has a chance to detect such events and clean the signal!
        :param subject_info: object containing the participant specific infos
        :param pd_signal: photodiode signal!
        :return:
        """

    choose_intervals = input("You have not set any trigger channel noise intervals. Do you wish to do so? [Yes or No]"
                             "\n If your experiment was restarted, it is a good idea to remove the following triggers"
                             "\n which are send in the restarting process: "
                             "\n 1. two experiment start/end triggers (4 consecutive triggers each)"
                             "\n 2. block start (4 consecutive triggers) \n")

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    plt.plot(pd_signal['Amp'][0], 'k')
    plt.xlabel('Sample nr')
    plt.ylabel('Amplitude')

    while choose_intervals == 'Yes' or choose_intervals == 'yes':

        print('Before')
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        plt.plot(pd_signal['Amp'][0], 'b')
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')
        plt.title('Raw photodiode signal with thresholds and onset. '
                  + '\n 1. Zoom in on the interval you want to zero.'
                  + '\n 2. Deselect all buttons from the lower left button bar.'
                  + '\n 3. You can now click and drag a rectacle around the x axis interval enclosing the specific interval'
                  + '\n 4. When you are happy with the interval, close this plot to choose more intervals or proceed.')

        ind_start_noise, ind_end_noise = select_start_and_end(fig, ax, 'box')
        print('You have chosen the start index to be ', ind_start_noise)
        print('and the end index to be ', ind_end_noise)
        print('All values in this interval will be set to zero.')
        print('If you are happy with these values, close the figure and you can enter the next interval.')

        if ind_start_noise < 0:
            ind_start_noise = 0
        subject_info.start_inds_trigger_noise.append(ind_start_noise)
        subject_info.end_inds_trigger_noise.append(ind_end_noise)

        choose_intervals = input("More to remove? Answer [Yes or No]")

    # Finally, updating the json with that info:
    print("Updating the subject info json file")
    subject_info.update_json()

    return subject_info


def handle_duplicates(full_logs):
    """
    This function reads in the log files and identify where there were repetitions and flags it
    :param full_logs: (pandas data frame) full logs from the first experiment
    :return: full_logs: (pandas data frame) data frame with duplication flag appended
    """
    # Finding the duplicates. What should be unique is the combination of block, miniblock and trial within those.
    # However, a trial consists most of the time of three events: stimulus, fixation and jitter, which will have the
    # same combination of the above. Hence the need for this specific subset
    full_logs['duplicate'] = full_logs.duplicated(
        subset=['block', 'miniBlock', 'trial', 'eventType'], keep='last')
    full_logs = full_logs.reset_index(drop=True)
    # To that, need to add the responses if there were any inbetween the repetition, because those won't be caught by
    # the above:
    for row_n, row in full_logs.iterrows():
        if row['eventType'] == "Response":
            if full_logs.loc[row_n - 1, 'duplicate'] == 1 and full_logs.loc[row_n + 1, 'duplicate'] == 1:
                full_logs.loc[row_n, 'duplicate'] = True

    return full_logs


def load_logs(root_path, file_naming_pattern, file_extension=None):
    """
    This function loads the log files and prepares them for the alignment with the photodiode flashes. This function
    will load recursively all the files with the matching naming pattern and extension along the root directory. So make
    sure you don't have multiple different files in the directory you will be searching to avoid loading spurious files
    :param root_path: (string or Pathlib path) root path to look for the files
    :param file_naming_pattern: (string) naming pattern of the files to load. You can use wildcards
    :param file_extension: (string) extension of the file to load. With the dot!
    :return: full_logs: (pandas data frame) log files of the experiment with duplicates removed
    """
    import pandas as pd

    # Getting the list of the log files:
    filesList = find_files(
        root_path, naming_pattern=file_naming_pattern, extension=file_extension)

    # Make sure they are in order:
    filesList.sort()

    # Preparing to load the logs:
    full_logs = pd.DataFrame()
    # Looping through file list to load the logs:
    for files in filesList:
        full_logs = full_logs.append(pd.read_csv(files))

    full_logs = handle_duplicates(full_logs)

    return full_logs


def data_preparation():
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument('--SubjectID', type=str, default=None,
                        help="Subject ID, for instance SE110")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Analysis parameters file (file name + path)")
    args = parser.parse_args()
    print(args.SubjectID)

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiating parameters jsons
    # Creating the analysis parameters:
    print("Reading analysis info from analysis parameter config file")
    data_preparation_parameters = DataPreparationParameters(
        args.AnalysisParametersFile)

    # Creating the subject info based on the json file created:
    print("Reading subject info from subject info config file or preparing to create subject info file")
    subject_info = SubjectInfo(args.SubjectID, data_preparation_parameters)
    # Get the site as some steps will differ a little based on it
    site = re.split('(\d+)', subject_info.SUBJ_ID)[0]
    # ------------------------------------------------------------------------------------------------------------------
    # Loading the data for inspection:
    print(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID)
    raw = load_signal(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID,
                      data_preparation_parameters.ecog_files_naming, subject_info,
                      file_extension=data_preparation_parameters.ecog_files_extension,
                      debug=data_preparation_parameters.debug)
    print(raw.info)
    # Get the sampling frequency of the file
    sr = raw.info["sfreq"]
    if site == "SE":
        change_sr = input(
            "The sampling rate in your file is {0}. Do you want to change it? (Yes/No) ".format(sr))
        if change_sr.lower() == "yes":
            new_sr = input("What was the sampling rate of your file? (Float) ")
            raw.info["sfreq"] = float(new_sr)
    # Extracting the photodiode signal from the raw:
    pd_signal = extract_pd_signal(raw, subject_info)

    # Selecting bad channels (EKG, DC...) manually:
    subject_info = select_bad_channels(
        raw, data_preparation_parameters.debug, subject_info)

    # Adjusting the photodiode threshold if needed:
    subject_info = manual_adjust_threshold(subject_info, pd_signal)

    # Finally, the subject info in the json needs to be updated:
    subject_info.update_json()

    # ------------------------------------------------------------------------------------------------------------------
    # Photodiode trigger alignment:
    # In ECoG there are no LPT triggers or standardized way to align signal to experiment computer. We relied on
    # photodiode to detect events onsets. The information from the computer log files is then used to get the trial
    # information after alignment. Based on this information, we align the photodiode signal to the log files to then
    # create annotations and events in our signal:

    # First, loading the log files:
    full_logs = load_logs(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID,
                          data_preparation_parameters.beh_files_naming,
                          file_extension=data_preparation_parameters.beh_files_extension)

    # Computing triggers onsets:
    pd_onsets, pd_onsets_clean, pd_signal_raw, edf_srate, raw = trigger_alignment.detect_triggers(
        data_preparation_parameters,
        subject_info, raw)

    # Checking alignment with logs:
    full_logs_clean = trigger_alignment.check_alignment(pd_onsets_clean, full_logs,
                                                        data_preparation_parameters, subject_info,
                                                        raw.info['sfreq'])

    # As a sanity check, comparing the stimuli expected vs observed duration:
    trigger_alignment.check_stim_duration(
        full_logs_clean, data_preparation_parameters, subject_info)

    # Now that we made sure things are aligned, we can use the information from the log files to create the events
    # in our signal:
    raw = trigger_alignment.create_annotations(full_logs_clean,
                                               raw)

    # ------------------------------------------------------------------------------------------------------------------
    # Setting channels type:
    # If the electrodes reconstruction, it is possible that the electrodes type (seeg, ecog or eeg) is found in the
    # electrodes loc table. But if not, do so based on namiong conventions. Note that based on naming conventions
    # is way less reliable. Not also that any channel that wasn't reconstructed will be considered EEG:
    try:
        elec_recon_files = find_files(data_preparation_parameters.elec_loc_root + os.sep + subject_info.SUBJ_ID,
                                      naming_pattern=data_preparation_parameters.elec_loc[
                                          list(data_preparation_parameters.elec_loc.keys())[0]],
                                      extension=data_preparation_parameters.elec_loc_extension)
        # If there were more than 1 elec recon file, use whichever was the latest:
        if len(elec_recon_files) > 1:
            elec_recon_file = find_correct_file(elec_recon_files)
        else:
            elec_recon_file = elec_recon_files[0]
        print(elec_recon_file)
        # There are slight differences in format of elec recon between the different sites, which is adressed here:
        if site == "SF":
            # Loading the file:
            elec_coord_raw = np.genfromtxt(elec_recon_file, dtype=str, delimiter=' ',
                                           comments=None, encoding='utf-8')
            elec_coord_raw[:, 0] = remove_elecname_leading_zero(
                elec_coord_raw[:, 0])
            # Convert to a dataframe for ease of use:
            elec_coord = pd.DataFrame({
                "name": elec_coord_raw[:, 0],
                "x": elec_coord_raw[:, 1],
                "y": elec_coord_raw[:, 2],
                "z": elec_coord_raw[:, 3],
                "type": elec_coord_raw[:, 4],
            })
        elif site == "SE":
            # Loading the file:
            elec_coord_raw = np.genfromtxt(elec_recon_file, dtype=str, delimiter=',',
                                           comments=None, encoding='utf-8')
            # Keeping only the columns of interest
            elec_coord_raw = elec_coord_raw[1:, [0, 1, 3, 4, 5]]
            # Convert to a dataframe for ease of use:
            elec_coord = pd.DataFrame({
                "name": elec_coord_raw[:, 0],
                "x": elec_coord_raw[:, 2],
                "y": elec_coord_raw[:, 3],
                "z": elec_coord_raw[:, 4],
                "type": elec_coord_raw[:, 1],
            })
        elif site == "SG":
            # Loading the file:
            elec_coord_raw = np.genfromtxt(elec_recon_file, dtype=str, delimiter='\t',
                                           comments=None, encoding='utf-8')
            # Keeping only the columns of interest
            elec_coord_raw = elec_coord_raw[1:, [0, 1, 2, 3]]
            # Convert to a dataframe for ease of use:
            elec_coord = pd.DataFrame({
                "name": elec_coord_raw[:, 0],
                "x": elec_coord_raw[:, 1],
                "y": elec_coord_raw[:, 2],
                "z": elec_coord_raw[:, 3],
                "type": "D",
            })

        # Looping through all channels in the signal, to set their types:
        for ind, ch in enumerate(raw.ch_names):
            # Finding the channel in the table:
            if ch.replace(" ", "") in elec_coord["name"].to_list():
                # Set the channel type accordingly:
                if elec_coord.loc[elec_coord['name'] == ch.replace(" ", ""), "type"].item() == "D":
                    raw.set_channel_types({ch: "seeg"})
                    print("{0}: {1}".format(ch, "seeg"))
                else:
                    raw.set_channel_types({ch: "ecog"})
                    print("{0}: {1}".format(ch, "ecog"))
            # If the electrode is not in the table but the letter match one of the naming convention, setting the:
            # channel type accordingly
            elif re.findall("[a-zA-Z]+", ch)[0] in list(data_preparation_parameters.additional_channel_conv.keys()):
                raw.set_channel_types({
                    ch: data_preparation_parameters.additional_channel_conv[re.findall("[a-zA-Z]+", ch)[0]]
                })
                print("{0}: {1}".format(ch, data_preparation_parameters.additional_channel_conv[re.findall("[a-zA-Z]+",
                                                                                                           ch)[0]]))
            else:  # Otherwise, set the channel as bad, we don't know what that is:
                raw.set_channel_types({ch: "seeg"})
                raw.info['bads'].append(ch)
                mne.utils.warn("Channel {0} was not found in the electrode localization file nor did it match known "
                               "\nnaming conventions! It was therefore set to seeg channel type but set as bad. "
                               "\nGo an edit the bids channels information if this is not a correct "
                               "handling".format(ch), RuntimeWarning)
                print("{0}: {1}".format(ch, "seeg"))
    except IndexError:
        # If the site is SE, then most of the electrodes are depth, therefore, setting all the channels to seeg. The
        # other type are handled below:
        if site == "SE":
            raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
        for elec_name_conv in data_preparation_parameters.additional_channel_conv.keys():
            raw.set_channel_types({ch_name: data_preparation_parameters.additional_channel_conv[elec_name_conv]
                                   for ch_name in raw.ch_names if elec_name_conv == ch_name[0:len(elec_name_conv)]})

    # ------------------------------------------------------------------------------------------------------------------
    # BIDS conversion:
    # Getting the reconstructed electrodes file. The [0] in the end is to avoid returning a list. There should only be
    # one anyways:
    try:
        electrode_recon = {}
        for coordinates_space in data_preparation_parameters.elec_loc.keys():
            elec_recon_files = find_files(data_preparation_parameters.elec_loc_root + os.sep + subject_info.SUBJ_ID,
                                          naming_pattern=data_preparation_parameters.elec_loc[
                                              coordinates_space],
                                          extension=data_preparation_parameters.elec_loc_extension)
            # If there were more than 1 elec recon file, use whichever was the latest:
            if len(elec_recon_files) > 1:
                elec_recon_file = find_correct_file(elec_recon_files)
            else:
                elec_recon_file = elec_recon_files[0]
            electrode_recon[coordinates_space] = elec_recon_file

    except IndexError:
        electrode_recon = None

    try:
        # Creating a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Generate a file name
        fname = temp_dir + os.sep + "_raw.fif"
        # Saving the data in that temp dir:
        raw.save(fname)
        # Adjust the channels_description_file:
        if data_preparation_parameters.channels_description_file is not None:
            data_preparation_parameters.channels_description_file = \
                data_preparation_parameters.channels_description_file.format(subject_info.SUBJ_ID,
                                                                             subject_info.session,
                                                                             data_preparation_parameters.task_name)
        # Saving the data to BIDS:
        print(raw.info)
        save_to_BIDs(mne.io.read_raw(fname, preload=False),
                     elec_recon_file=electrode_recon,
                     bids_root=data_preparation_parameters.BIDS_root,
                     subject_id=subject_info.SUBJ_ID, session=data_preparation_parameters.session,
                     task=data_preparation_parameters.task_name, data_type="ieeg",
                     line_freq=data_preparation_parameters.line_freq, site=site,
                     channels_description_file=data_preparation_parameters.channels_description_file,
                     project_ecog_to_surf=True, raw_root=data_preparation_parameters.raw_root)

    finally:  # Removing the temporary directory:
        files = glob.glob(temp_dir + os.sep + '*')
        for f in files:
            os.remove(f)
        os.rmdir(temp_dir)

    # Finally, the subject info in the json needs to be updated:
    subject_info.update_json()
    print("Done. Ready to preprocessing EDFPreprocessing.py!")


if __name__ == "__main__":
    data_preparation()
