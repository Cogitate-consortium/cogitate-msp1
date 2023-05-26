""" This script contains all helper function for the preprocessing pipeline
    authors: Alex Lepauvre and Katarina Bendtz
    alex.lepauvre@ae.mpg.de
    katarina.bendtz@tch.harvard.edu
    Dec 2020
"""

import os
from pathlib import Path
import numpy as np
import math

import mne
from mne.viz import plot_alignment, snapshot_brain_montage
from mne.datasets import fetch_fsaverage
from nibabel.freesurfer.io import read_geometry

import matplotlib.pyplot as plt

from scipy.stats import median_abs_deviation, linregress
import pandas as pd

import shutil
import subprocess

from general_helper_functions.pathHelperFunctions import find_files


def path_generator(root, step, signal=None, previous_steps_list=None, figure=False):
    """
    This function generates path according to the cogitate folders structure convention.
    :param root: (pathlib path object OR string path) root to where the data should be saved
    :param step: (string) name of the step we are trying to save
    :param signal: (string) name of the signal we are trying to save
    :param previous_steps_list: (list of strings) list of all the previous steps that were performed
    :param figure: (boolean) whether or not the file we are trying to save is a figure
    :return:
    """
    # The previous steps list has strings too long for path generation. Concatenating the first three letters of the
    # step description:
    if previous_steps_list is not None:
        previous_steps_string = "_".join(["".join([st[0:3] for st in steps.split("_")]) for steps in
                                          previous_steps_list])
        if len(previous_steps_string) == 0:
            previous_steps_string = "raw"

    # First, generating the path:
    if figure is False:
        if previous_steps_list is not None and signal is not None:
            save_path = Path(root, step, signal, previous_steps_string)
        elif signal is not None:
            save_path = Path(root, step, signal)
        else:
            save_path = Path(root, step)
    else:
        if previous_steps_list is not None and signal is not None:
            save_path = Path(root, step, "figure", signal,
                             previous_steps_string)
        elif signal is not None:
            save_path = Path(root, step, "figure", signal)
        else:
            save_path = Path(root, step, "figure")

    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    return save_path


def file_name_generator(save_path, file_prefix, description, file_extension, data_type="ieeg"):
    """
    This function generates full file names according to the cogitate naming conventions:
    :param save_path: (pathlib path object or path string) root path to where the data should be saved
    :param file_prefix: (string) prfix of the file name
    :param description: (string) what some after teh prefix in the file name
    :param file_extension: (string) what comes after the description, if anything
    :param data_type: (string) data type the data are from
    :return:
    """
    full_file_name = os.path.join(
        save_path, file_prefix + description + "_" + data_type + file_extension)

    return full_file_name


def mne_data_saver(data, preprocessing_parameters, subject_info, step_name, data_type="ieeg", signal="broadband",
                   mne_file_extension="-raw.fif"):
    """
    This function saves the different instances of mne objects
    :param data: (mne object: epochs, evoked, raw...) data to be saved
    :param preprocessing_parameters: (custom object) contains info about the preprocessing
    :param subject_info: (custom objct) contains info about the subject
    :param step_name: (string) name of the preprocessing step in which we are trying to save, to ensure that we save it
    in the right place
    :param data_type: (string) type of the data being saved
    :param signal: (string) name of the signal we are saving
    :param mne_file_extension: (string) file ending. Try to respect the mne conventions
    :return:
    """
    print("=" * 40)
    print("Saving mne object")

    # First, generating the root path to save the data:
    save_path = path_generator(subject_info.participant_save_root, step_name, signal,
                               preprocessing_parameters.analyses_performed)
    # Generating the full file name:
    full_file_name = file_name_generator(save_path, subject_info.files_prefix, step_name, mne_file_extension,
                                         data_type=data_type)

    # Saving the data:
    data.save(full_file_name, overwrite=True)
    # Saving the parameters as well:
    preprocessing_parameters.save(save_path, subject_info.files_prefix)

    return None


def plot_channels_psd(raw, preprocessing_parameters, subject_info, step_name, data_type="ieeg", signal="broadband",
                      file_extension=".png", plot_single_channels=False, channels_type=None):
    """
    This function plots and saved the psd of the chosen electrodes. There is also the option to plot each channel
    separately
    :param raw: (mne raw object)
    :param preprocessing_parameters: (custom made object) contains info about the preprocessing
    :param subject_info: (custom object) contains info about the subject' data
    :param step_name: (string) name of the preprocessing step this function was called in
    :param data_type: (string) type of data being preprocessed
    :param signal: (string) name of the signal being investigated
    :param file_extension: (string) pic extension
    :param plot_single_channels: (boolean) whether or not to plot single channels or only all of them superimposed
    :param channels_type: (dict) list of the channels of interest
    :return:
    """
    # Getting  the relevant channels:
    if channels_type is None:
        channels_type = {"ecog": True, "seeg": True}
    picks = mne.pick_types(raw.info, **channels_type)
    # Handle path save and so on:
    save_path = path_generator(subject_info.participant_save_root, step_name, signal,
                               preprocessing_parameters.analyses_performed, figure=True)
    full_file_name = file_name_generator(save_path, subject_info.files_prefix, "PSD", file_extension,
                                         data_type=data_type)

    # Plotting the psd from all the channels:
    raw.plot_psd(picks=picks, show=False)
    # Saving the figure:
    plt.savefig(full_file_name, dpi=300, transparent=True)
    plt.close()

    # For all channels separately:
    if plot_single_channels:
        # Compute the PSD for all the picks:
        psd, freqs = mne.time_frequency.psd_welch(raw, picks=picks, average="mean")
        for ind, pick in enumerate(picks):
            fig, ax = plt.subplots(figsize=[15, 6])
            ax.plot(freqs, np.log(psd[np.arange(psd.shape[0]) != ind, :].T), linewidth=0.5, color="k", alpha=0.8)
            ax.plot(freqs, np.log(psd[ind, :].T).T, linewidth=2, color="r")
            ax.set_ylabel("\u03BCV\u00B2/Hz (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim([freqs[0], freqs[-1]])
            ax.grid(linestyle=":")
            ax.set_title("{} PSD".format(raw.ch_names[pick]))
            full_file_name = file_name_generator(save_path, subject_info.files_prefix, "PSD" + raw.ch_names[pick],
                                                 file_extension, data_type=data_type)
            plt.savefig(full_file_name, dpi=300, transparent=True)
            plt.close()

    return None


def plot_bad_channels(raw, preprocessing_parameters, subject_info, step_name, data_type="ieeg", signal="broadband",
                      file_extension=".png", plot_single_channels=False, picks="bads"):
    """
    This function plots the bad channels psd and raw signal to show what it being disarded:
    :param raw: (mne raw object)
    :param preprocessing_parameters: (custom made object) contains info about the preprocessing
    :param subject_info: (custom object) contains info about the subject' data
    :param step_name: (string) name of the preprocessing step this function was called in
    :param data_type: (string) type of data being preprocessed
    :param signal: (string) name of the signal being investigated
    :param file_extension: (string) pic extension
    :param plot_single_channels: (boolean) whether or not to plot single channels or only all of them superimposed
    :param picks: (list) list of the channels of interest
    :return:
    """
    # Handling picks input:
    if picks == "bads":
        picks = raw.info["bads"]
    # Handling the saving path and so on:
    save_path = path_generator(subject_info.participant_save_root, step_name, signal,
                               preprocessing_parameters.analyses_performed, figure=True)
    full_file_name = file_name_generator(save_path, subject_info.files_prefix, "badchannels", file_extension,
                                         data_type=data_type)

    if len(picks) > 0:
        # Plotting the psd from all the channels:
        fig, axs = plt.subplots(1, 1)
        plt.suptitle("Bad channels: N= " + str(len(picks))
                     + " out of " + str(len(raw.info["ch_names"])))
        # Plotting the average of the good channels with standard error for reference. Downsampling, otherwise too many
        # data points:
        good_channels_data, times = \
            raw.copy().resample(200).get_data(picks=mne.pick_types(
                raw.info, ecog=True, seeg=True), return_times=True)
        mean_good_data = np.mean(good_channels_data.T, axis=1)
        ste_good_data = np.std(good_channels_data.T, axis=1) / \
                        np.sqrt(good_channels_data.T.shape[1])
        axs.plot(times, mean_good_data, alpha=0.5, color="black")
        axs.fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data, alpha=0.2,
                         color="black")
        # Plotting the time series of all the bad channels:
        axs.plot(times, raw.copy().resample(200).get_data(
            picks=picks).T, alpha=0.8, label=picks)
        # Adding labels and legend:
        axs.set_xlabel("time (s)")
        axs.set_ylabel("amplitude")
        axs.legend()
        # Plotting the psd
        # raw.plot_psd(picks=picks, show=False, ax=axs[1:2])
        # Saving the figure:
        plt.savefig(full_file_name, transparent=True)
        plt.close()
        # For all channels separately:
        if plot_single_channels:
            # Looping through each channels:
            for pick in picks:
                # Plotting the psd from this channel:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(times, mean_good_data, alpha=0.5, color="black")
                axs[0].fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data,
                                    alpha=0.2,
                                    color="black")
                axs[0].plot(times, raw.copy().resample(
                    200).get_data(picks=pick).T, label=pick)
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                # Adding labels and legend:
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                axs[0].legend()
                # Plotting the psd
                raw.plot_psd(picks=pick, show=False, ax=axs[1])
                full_file_name = file_name_generator(save_path, subject_info.files_prefix, "bad" + pick, file_extension,
                                                     data_type=data_type)
                plt.savefig(full_file_name, transparent=True)
                plt.close()

    return None


def plot_rejected_trials(epochs, ind_trials_to_drop, preprocessing_parameters, subject_info, step_name,
                         data_type="ieeg", signal="broadband",
                         file_extension=".png", plot_single_channels=False, picks=None):
    """
    This function plots the bad channels psd and raw signal to show what it being disarded:
    :param epochs: (mne epochs object)
    :param ind_trials_to_drop: list of index of the trials to drop
    :param preprocessing_parameters: (custom made object) contains info about the preprocessing
    :param subject_info: (custom object) contains info about the subject' data
    :param step_name: (string) name of the preprocessing step this function was called in
    :param data_type: (string) type of data being preprocessed
    :param signal: (string) name of the signal being investigated
    :param file_extension: (string) pic extension
    :param plot_single_channels: (boolean) whether or not to plot single channels or only all of them superimposed
    :param picks: (list) list of the channels of interest
    :return:
    """
    # Handling picks input:
    if picks is None:
        picks = epochs.ch_names

    # Handling the saving path and so on:
    save_path = path_generator(subject_info.participant_save_root, step_name, signal,
                               preprocessing_parameters.analyses_performed, figure=True)

    # For all channels separately:
    if plot_single_channels:
        # Looping through each channels:
        for pick in picks:
            # Getting the data of this specific channel:
            data = np.squeeze(epochs.get_data(picks=pick))
            # Getting the bad trials:
            bad_trials = data[ind_trials_to_drop, :]
            # Getting the good trials:
            ind_trials_not_to_drop = [i for i in range(
                0, len(epochs)) if i not in ind_trials_to_drop]
            good_trials = data[ind_trials_not_to_drop, :]
            # Plotting the psd from this channel:
            if bad_trials.size != 0:
                plt.plot(bad_trials.T, linewidth=0.8, color="red")
            if good_trials.size != 0:
                plt.plot(good_trials.T, linewidth=0.2,
                         color="black", alpha=0.2)
            plt.title("Rejected and accepted trials for channel " + pick)
            plt.ylabel("Amplitude")
            plt.xlabel("sample")
            full_file_name = file_name_generator(save_path, subject_info.files_prefix, "badepochs" + pick,
                                                 file_extension, data_type=data_type)
            plt.savefig(full_file_name, dpi=300, transparent=True)
            plt.close()
    # dropping the bad trials:
    epochs.drop(ind_trials_to_drop, reason="AUTOMATED_REJECTION")
    # Plotting the drop log:
    full_file_name = file_name_generator(save_path, subject_info.files_prefix, "badepochs", file_extension,
                                         data_type=data_type)

    # Plotting the drop logs:
    epochs.plot_drop_log(show=False)
    # Saving the figure:
    plt.savefig(full_file_name, dpi=300, transparent=True)
    plt.close()

    return None


def manual_signal_inspection(raw, subject_info, instructions=""):
    """
    This function plots the raw data alongside the events. The user should scroll through the signal to isolate bad
    channels:
    :param raw: (mne raw object)
    :param subject_info: (SubjectInfo Class object) custom made participant info object
    :param instructions: (String) instructions to be displayed to the user to know what to do with interactive plot
    :return:
    """
    # Extracting which channels were already marked as bad:
    bad_channels = raw.info["bads"]
    # You should manually inspect the channels to find the bad ones:
    raw.plot(show=True, block=True, title=instructions)
    # Now extracting which are the newly marked bad channels:
    new_bad_channels = [ch for ch in raw.info['bads'] if ch not in bad_channels]
    # The channels that are marked as bad on the plot are saved as bad channels in the raw object
    subject_info.manu_bad_channels.append(new_bad_channels)

    return raw


def automated_bad_channel_detection(raw, method="psd_based", epoch_length=1.0, mad_thresh=4,
                                    segment_proportion_cutoff=0.1, channel_types=None, reject_bad_channels=False):
    """
    This function detects bad channels automatically according to criterion detailed here:
    https://doi.org/10.1016/j.celrep.2021.109585 with slight adaptation
    The raw signal is segmented in 1 sec bin. The range within each segment is computed and compared to the thresholds
    max and min range. The slope (i.e. micro V / sec) is also computed between each pair of time points in each segment.
    For each segment, the max slope is compared to max_slope. A channel is detected as bad if for more than
    segment_proportion_cutoff proportion of the trial were found to violate the previously described criterion.
    There is further the option to reject or not directly those channels (i.e. add the channel in the list of bads in
    the raw object).
    :param raw: (mne raw object) raw object for which to detect the bad channels
    :param method: (string) method to use: either psd_based or activation_based. In psd based, the power spectrum
    density is computed and the
    :param epoch_length: (float) duration of the epoch to segment the data  in sec (default 1 sec)
    :param mad_thresh: (float) z score threshold that the observed data must not exceed to be kept
    :param segment_proportion_cutoff: (float) how many segments must be messed up for the channel to be considered bad
    Only considered for activation_based method
    :param channel_types: (dict) channels type to include in this
    :param reject_bad_channels: (bool) whether or not to add the information about the channels being bad to the raw
    object to be ignored from here on
    :return:
    raw: mne raw object
    detected_bad_channels: list of channels names considered as bad
    """
    # Handling parameters:
    # Name of the preprocessing step to name the file and create folder:
    print("-" * 40)
    print("Performing automated bad channels detection")
    if method not in ["psd_based", "activation_based", "both"]:
        raise Exception("You have passed an unsupported method! Either psd_based or activation_based or both!")
    if channel_types is None:
        channel_types = {"ecog": True, "seeg": True}
    bad_channels = []
    for channel_type in channel_types.keys():
        # First getting the channels:
        ch_names = [raw.ch_names[ind] for ind in mne.pick_types(raw.info, **{channel_type: True, "exclude": "bads"})]
        if len(ch_names) == 0:
            mne.utils.warn("You have attempted to plot {0} channels, but none where found in your signal".
                           format(channel_type),
                           RuntimeWarning)
            continue
        if method.lower() == "psd_based" or method.lower() == "both":
            # Compute average psd for each channel:
            psd, freqs = mne.time_frequency.psd_welch(raw, picks=ch_names, average="mean")
            # Log transforming the psd:
            log_psd = np.log(psd)
            # Mean center the PSD:
            mean_cent_psd = np.array(log_psd.T - np.mean(log_psd, axis=-1)).T
            # Compute the average of the mean centered PSD:
            avg_mean_cent_psd = np.mean(mean_cent_psd, axis=-1)
            # Compute the median absolute deviation of the mean centered psd:
            mean_psd_mad = np.array([np.abs(avg_mean_cent_psd[i] - np.median(avg_mean_cent_psd)) /
                                     median_abs_deviation(avg_mean_cent_psd, axis=0)
                                     for i in range(avg_mean_cent_psd.shape[0])])
            # Compute the psd slope for each channel from 10 to 100Hz:
            ind_1, ind_2 = np.where(freqs >= 10)[0][0], np.where(freqs <= 100)[0][-1]
            log_psd_slopes = np.array([linregress(freqs[ind_1:ind_2], log_psd[i, ind_1:ind_2]).slope
                                       for i in range(log_psd.shape[0])])
            # Compute the MAD of the slope:
            log_slope_mad = np.array([np.abs(log_psd_slopes[i] - np.median(log_psd_slopes)) /
                                      median_abs_deviation(log_psd_slopes, axis=None)
                                      for i in range(log_psd_slopes.shape[0])])
            # Binarize both MAD:
            slopes_bin = log_slope_mad > mad_thresh
            mean_bin = mean_psd_mad > mad_thresh
            # Find the channels that have either weird slopes or weird mean centered mean
            bad_channels.extend([ch_names[i] for i in np.where(np.logical_or(slopes_bin, mean_bin))[0]])
        if method.lower() == "activation_based" or method.lower() == "both":
            # Generate epochs of fixed length:
            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True,
                                                  reject_by_annotation=True, overlap=0.0,
                                                  verbose=False).pick(ch_names)
            # Extract the data:
            data = epochs.get_data() * 10 ** 6
            # Compute segments range:
            segments_range = np.ptp(data, axis=-1)
            # Compute the max slope (microV / millisec) in each segment (i.e. diff between every successive samples
            # x the interval between two samples):
            segments_max_slope = np.max(np.abs(np.diff(data, axis=-1) * (1000 / epochs.info["sfreq"])), axis=-1)
            # Compute zscore of each segment:
            range_mad = np.abs(median_abs_deviation(segments_range, axis=None))
            slope_mad = np.abs(median_abs_deviation(segments_max_slope, axis=None))
            # Compare channels against the z score threshold:
            range_bin = range_mad > mad_thresh
            slope_bin = slope_mad > mad_thresh
            # Combine both into one array:
            rej_array = np.logical_or(range_bin, slope_bin)
            # Compute the proprotion of rejected trials for each channel:
            rej_prop = np.mean(rej_array, axis=0)
            # Get the indices of the bad channels:
            bad_channels.extend([ch_names[i] for i in np.where(rej_prop > segment_proportion_cutoff)[0]])

    # Remove duplicates:
    bad_channels = list(set(bad_channels))
    print('')
    print("The following channels were marked as bad in the mne raw object and in the subject info file:")
    print(bad_channels)

    # Set the bad channels as bad in the raw object:
    if reject_bad_channels:
        raw.info['bads'].extend(bad_channels)

    return raw, bad_channels


def custom_car(raw, reference_channel_types=None, target_channel_types=None):
    """
    This function takes specific channel types as reference and averages the amplitude across them along time. This mean
    time series is then substract from, all the channels at each time points.
    :param raw: (mne raw object) contains the data to be rereferenced
    :param reference_channel_types: (dict) dictionary specifying the channels types to be take as reference, as well as
    which to exclude. See mne documentation for mne.pick_types to know what you can pass here
    :param target_channel_types: (dict) dictionary specifying the channels types to apply reference to, as well as
    which to exclude. See mne documentation for mne.pick_types to know what you can pass here
    :return: raw: (mne raw object) modified instance of the mne object. Note that the data are modified in place, so
    use copy upon calling this function to avoid overwriting things
    """
    # Handling empty input:
    if reference_channel_types is None:
        reference_channel_types = {'ecog': True}
    if target_channel_types is None:
        target_channel_types = {'ecog': True}

    # Setting the electrodes to apply from and to
    ref_from = mne.pick_types(raw.info, **reference_channel_types)
    ref_to = mne.pick_types(raw.info, **target_channel_types)

    # Fetching the data. Using _data instead of get_data, because that enables modifying data in place:
    data = raw._data

    # Compute and apply ref:
    ref_data = data[..., ref_from, :].mean(-2, keepdims=True)
    data[..., ref_to, :] -= ref_data

    # Logging the custom ref:
    raw.info['custom_ref_applied'] = 1

    return raw


def notch_filtering(raw, njobs=1, frequency=60, remove_harmonics=True, filter_type="fir",
                    cutoff_lowpass_bw=None, cutoff_highpass_bw=None, channel_types=None):
    """
    This function filters the raw data according to the set parameters
    :param raw: (mne raw object) continuous data
    :param njobs: (int) number of jobs to preprocessing the filtering in parallel threads
    :param frequency: (int or float) frequency to notch out.
    :param remove_harmonics: (boolean) whether or not to remove all the harmonics of the declared freq (up until the
    sampling freq)
    :param filter_type: (string) what type of filter to use, iir or fir
    :param cutoff_lowpass_bw: (float) frequency for low pass (only used if type is iir)
    :param cutoff_highpass_bw: (float) frequency for high pass (only used if type is iir)
    :param channel_types: (dict) type of channels to notch filter, boolean for the channel types
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Filtering the data:
    # Getting the harmonics frequencies if any:
    if channel_types is None:
        channel_types = {"ecog": True, "seeg": True, "exclude": "bads"}
    if remove_harmonics:
        freq_incl_harmonics = [
            frequency * i for i in range(1, int((raw.info['sfreq'] / 2) // frequency) + 1)]
        frequency = freq_incl_harmonics

    # Selecting the channels to filter:
    picks = mne.pick_types(raw.info, **channel_types)
    if filter_type.lower() == "fir":
        # Applying FIR if FIR in parameters
        raw.notch_filter(frequency, filter_length='auto',
                         phase='zero', n_jobs=njobs, picks=picks)  # Default method is FIR
    elif filter_type.lower() == "butterworth_4o":  # Applying butterworth 4th order
        # For the iir methods, mne notch_filter does not support to pass several frequencies at the same time.
        # It also does not support having customized cutoff frequencies.
        # We therefore
        # 1. call the filter function to perform a custom butterworth 4th order
        # (default if method is set to "iir" and no iir_params parameter is given)
        # bandpass filter (notch_cutoff_hp Hz - notch_cutoff_lp Hz)
        if cutoff_lowpass_bw != 0:
            raw.filter(cutoff_lowpass_bw, cutoff_highpass_bw,
                       phase='zero', method='iir', n_jobs=njobs, picks=picks)
        else:
            raw.notch_filter(frequency, method='iir', n_jobs=njobs)
        # If there are harmonics to filter out as well:
        if remove_harmonics:
            # Now drop the first frequency from the frequencies:
            frequencies = frequency[1:]
            # 2. call the notch_filter function for each of the harmonics to perform the filtering of the harmonics.
            # Note that we here use the standard bandwidth freq/200.
            for freq in frequencies:
                raw.notch_filter(freq, method='iir', n_jobs=njobs, picks=picks)

    return raw


def create_metadata_from_events(epochs, metadata_column_names):
    """
    This function parses the events found in the epochs descriptions to create the meta data. The column of the meta
    data are generated based on the metadata column names. The column name must be a list in the same order as the
    strings describing the events. The name of the column must be the name of the overall condition, so say the
    specific column describes the category of the presented stim (faces, objects...), then the column should be called
    category. This will become obsolete here at some point, when the preprocessing is changed to generate the meta data
    directly
    :param epochs: (mne epochs object) epochs for which the meta data will be generated
    :param metadata_column_names: (list of strings) name of the column of the meta data. Must be in the same order
    as the events description + must be of the same length as the number of word in the events description
    :return: epochs (mne epochs object)
    """

    # Getting the event description of each single trial
    trials_descriptions = [[key for key in epochs.event_id.keys() if epochs.event_id[key] == event]
                           for event in epochs.events[:, 2]]
    trial_descriptions_parsed = [description[0].split(
        "/") for description in trials_descriptions]
    # Making sure that the dimensions of the trials description is consistent across all trials:
    if len(set([len(vals) for vals in trial_descriptions_parsed])) > 1:
        raise ValueError('dimension mismatch in event description!\nThe forward slash separated list found in the '
                         'epochs description has inconsistent length when parsed. Having different number of '
                         'descriptors for different trials is not yet supported. Please make sure that your events '
                         'description are set accordingly')
    if len(metadata_column_names) != len(trial_descriptions_parsed[0]):
        raise ValueError("The number of meta data columns you have passed doesn't match the number of descriptors for\n"
                         "each trials. Make sure you have matching numbers. In doubt, go and check the events file in\n"
                         "the BIDS directory")
    if len(trial_descriptions_parsed) != len(epochs):
        raise ValueError("Somehow, the number of trials descriptions found in the epochs object doesn't match the "
                         "number of trials in the same epochs. I have no idea how you managed that one champion, so I "
                         "can't really help here")

    # Convert the trials description to a pandas dataframe:
    epochs.metadata = pd.DataFrame.from_records(
        trial_descriptions_parsed, columns=metadata_column_names)

    return epochs


def epoching(raw, events, events_dict, picks="all", tmin=-0.5, tmax=2.0, events_not_to_epoch=None,
             baseline=(None, 0.0), reject_by_annotation=True, meta_data_column=None):
    """
    This function performs the epoching according to a few parameters
    :param raw: (mne raw object) data to epochs
    :param events: (mne events numpy array) three cols, one for event time stamp, the other for the event ID. This
    is what you get out from the mne functions to extract the events
    :param events_dict: (dict) mapping between the events ID and their descriptions
    :param picks: (string or list of int or list of strings) channels to epochs. If you pass all, it will select all the
    channels found in the raw object
    :param tmin: (float) how long before each event of interest to epochs. IN SECONDS!!!
    :param tmax: (float) how long before each event of interest to epochs. IN SECONDS!!!
    :param events_not_to_epoch: (list of strings) name of the events not to epochs. This is more handy than passing
    all the ones we want to epochs, because usually there are more events you are interested about than events you are
    not interested about
    :param baseline: (tuple of floats) passed to the mne epoching function to apply baseline correction and what is
    defined as the baseline
    :param reject_by_annotation: (boolean) whether or not to discard trials based on annotations that were made
    previously
    :param meta_data_column: (list of strings) list of the column names of hte metadata. The metadata are generated
    by parsing the extensive trial description you might have as events. For example, if you have something like this:
    Face/short/left, the meta data generator will create a table:
    col1  | col2  | col3
    Face    short   left
    The list you pass is the name of the different columns. So you would pass: ["category", "duration", "position"]:
    category  | duration  | position
    Face         short        left
    :return:
    """
    if picks == "all":
        picks = raw.info["ch_names"]
    if isinstance(baseline, list):
        baseline = tuple(baseline)
    print('Performing epoching')

    # We only want to epochs certain events. The config file specifies which ones should not be epoched (fixation
    # for ex). The function below returns only the events we are interested in!
    if events_not_to_epoch is not None:
        events_of_interest = {key: events_dict[key] for key in events_dict if not
        any(substring in key for substring in events_not_to_epoch)}
    else:
        events_of_interest = events_dict

    # Epoching the data:
    # The events are all events and the event_id are the events which we want to use for the
    # epoching. Since we are passing a dictionary we can also use the provided keys to acces
    # the events later
    epochs = mne.Epochs(raw, events=events, event_id=events_of_interest, tmin=tmin,
                        tmax=tmax, baseline=baseline, verbose='ERROR', picks=picks,
                        reject_by_annotation=reject_by_annotation)
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    # Adding the meta data to the table. The meta data are created by parsing the events strings, as each substring
    # contains specific info about the trial:
    if meta_data_column is not None:
        epochs = create_metadata_from_events(epochs, meta_data_column)

    return epochs


def automated_artifact_detection(epochs, standard_deviation_cutoff=4, trial_proportion_cutoff=0.1, channel_types=None,
                                 aggregation_function="peak_to_peak"):
    """
    This function perform a basic artifact rejection from the each electrode separately
    and then also removes the trial from all eletrodes if it is considered an outlier
    for more than X percent of the electrodes. NOTE: THIS FUNCTION DOES NOT DISCARD THE TRIALS THAT ARE DETECTED AS
    BEING BAD!!!
    :param epochs: (mne epochs object) epochs for which to reject trials.
    :param standard_deviation_cutoff: (float or int) number of standard deviation the amplitude of the data  a given
    trial must be to be considered an outlier
    :param trial_proportion_cutoff: (int between 0 and 1) proportion of channels for which the given trial must be
    considered "bad" (as defined by std factor) for a trial to be discarded all together
    :param channel_types: (dict: "chan_type": True) dictionary containing a boolean for each channel type to select
    which channels are considered by this step
    :param aggregation_function: (string) name of the function used to aggregate the data within trials and channels
    across time. If you set "mean", the mean will be computed within trial and channel. The standard deviation thereof
    will be used to compute the rejection theshold
    :return:
    """
    # Selecting electrodes of interest:
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    picks = mne.pick_types(epochs.info, **channel_types)

    if aggregation_function == "peak_to_peak" or aggregation_function == "ptp":
        spec_func = np.ptp
    elif aggregation_function == "mean" or aggregation_function == "average":
        spec_func = np.mean
    elif aggregation_function == "auc" or aggregation_function == "area_under_the_curve":
        spec_func = np.trapz
    else:
        raise Exception("You have passed a function for aggregation that is not support! The argument "
                        "\naggregation_function must be one of the following:"
                        "\npeak_to_peak"
                        "\nmean"
                        "\nauc")
    # ------------------------------------------------------------------------------------------------------------------
    # Computing thresholds and rejecting trials based on it:
    # Compute the aggregation (average, ptp, auc...) within trial and channel across time:
    trial_aggreg = spec_func(epochs.get_data(picks=picks), axis=2)
    # Computing the standard deviation across trials but still within electrode of the aggregation measure:
    stdev = np.std(trial_aggreg, axis=0)
    # Computing the average of the dependent measure across trials within channel:
    average = np.mean(trial_aggreg, axis=0)
    # Computing the artifact boundaries (beyond which a trial will be considered artifactual) by taking the mean +- n
    # times the std of the aggregated meaasure:
    artifact_thresh = np.array([average - stdev * standard_deviation_cutoff,
                                average + stdev * standard_deviation_cutoff])
    print(epochs.ch_names)
    print(artifact_thresh)
    # Comparing the values of each trial of a given channel against the threshold of that specific channel. Boolean
    # outcome:
    rejection_matrix = \
        np.array([np.array((trial_aggreg[:, i] > artifact_thresh[1, i]) & (trial_aggreg[:, i] > artifact_thresh[0, i]))
                  for i in range(trial_aggreg.shape[1])]).astype(float)
    # trial_proportion_cutoff dictates the proportion of channel for whom trial must be marked as bad to discard that
    # specific trial across all channels. Averaging the rejection matrix across channels to reject specific trials:
    ind_trials_to_drop = np.where(
        np.mean(rejection_matrix, axis=0) > trial_proportion_cutoff)[0]

    return epochs, ind_trials_to_drop


def find_interruption_index(events, event_dict, interuption_landmark, interruption_block_num=None):
    """
    This function asks the user whether there was an interruption in the experiment. If there was one, the function
    will ask which block the interruption occured in to find the index of the interruption. This is required by some
    functions that compute averaging accross the experiment. Indeed, if you had interruption, the signal change a lot
    inbetween, and averaging across the experiment without regard for the interruption is meaningless.
    :param events: (np array) contain the events indices and their identity
    :param event_dict: (dict) mapping between the events identity and their description
    :param interuption_landmark: (string) name of what to use to refer to for the interruption findings.
    :param interruption_block_num: (None or int) if you already know when the interruption occured, it can be passed
    in this function to avoid passing things manually.
    :return: interruption_index: int, index of where the interruption occured.
    """

    # Asking whether there was an interruption. If the interruption block number is not none, then no need to ask:
    if interruption_block_num is None:
        interuption = \
            input("Was your experiment interrupted at some point?")
    else:
        interuption = "yes"

    if interuption.lower() == "yes":
        if interruption_block_num is None:
            interruption_block_num = \
                input("In which {0} was your experiment interrupted?".format(
                    interuption_landmark))
        else:
            interruption_block_num = str(interruption_block_num)
        # Creating the description string for when the interruption happened:
        interuption_event_desc = interuption_landmark + "_" + interruption_block_num
        # Looking for all events fitting the description:
        evts = [event_dict[desc]
                for desc in event_dict if interuption_event_desc in desc]
        # Now looking for the index of the earliest occurence:
        interruption_index = int(min([evt[0] for evt in events if evt[2] in evts]))
    else:
        interruption_index = False

    return interruption_index


def frequency_bands_computations(raw, frequency_range=None, njobs=1, bands_width=10, channel_types=None,
                                 method="filter_bank", do_baseline_normalization=True, interruption_index=None):
    """
    This function computes the envelope in specified frequency band. It further has the option to compute envelope
    in specified bands within the passed frequency to then do baseline normalization to account for 1/f noise.
    :param raw: (mne raw object) raw mne object containing our raw signal
    :param frequency_range: (list of list of floats) frequency of interest
    :param bands_width: (float or int) width of the frequency bands to loop overs
    :param njobs: (int) number of parallel processes to compute the high gammas
    :param channel_types: (dict) name of the channels for which the high gamma should be computed. This is important
    to avoid taking in electrodes which we might not want
    :param method: (string) method to use to compute the high gamma. Options:
    filter_bank_ band pass the filter in different freq bands and compute the envelope in each. And then averaging
    across those envelopes
    band_pass_filter: compute the envelope of the signal band passed in the set freqs
    :param do_baseline_normalization: (bool) whether or not to do baseline normalization
    :return: frequency_band_signal: (mne raw object) dictionary containing raw objects with high gamma in the different
    frequency bands
    """

    def divide_by_average(data, int_ind):
        print('Dividing channels amplitude by average amplitude')
        if not isinstance(data, np.ndarray):
            raise TypeError('Input value must be an ndarray')
        if data.ndim != 2:
            raise TypeError('The data should have two dimensions!')
        else:
            if int_ind is None:
                norm_data = data / data.mean(axis=1)[:, None]
                # Check that the normalized data are indeed what they should be. We are dividing the data by a constant.
                # So if we divide the normalized data by the original data, we should get a constant:
                if len(np.unique((data[0, :] / norm_data[0, :]).round(decimals=15))) != 1:
                    raise Exception(
                        "The normalization of the frequency band went wrong! The division of normalized vs non "
                        "normalized data returned several numbers, which shouldn't be the case!")
                return norm_data
            else:
                data_before_int = data[:, 0:int_ind]
                norm_data_before_int = data_before_int / \
                                       data_before_int.mean(axis=1)[:, None]
                data_after_int = data[:, int_ind:]
                norm_data_after_int = data_after_int / \
                                      data_after_int.mean(axis=1)[:, None]
                return np.concatenate([norm_data_before_int, norm_data_after_int], axis=1)

    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    if frequency_range is None:
        frequency_range = [70, 150]
    print('-' * 40)
    print('Welcome to frequency bands computation')
    print(njobs)
    # Selecting the channels for which to compute the frequency band:
    picks = mne.pick_types(raw.info, **channel_types)
    if method == "filter_bank":
        # Getting the index of the channels for which the frequency band should NOT be computed
        not_picks = [ind for ind, ch in enumerate(
            raw.info["ch_names"]) if ind not in picks]
        # Creating copies of the raw for the channels for which the frequency band should be computed, and another for
        # the channels for which it shouldn't be computed, to avoid messing up the channels indices:
        freq_band_raw = raw.copy().pick(picks)
        if len(not_picks) != 0:
            rest_raw = raw.copy().pick(not_picks)

        # We first create the dictionary to store the final data:
        bands_amp = []
        # We then create the frequency bins to loop over:
        bins = []
        for i, freq in enumerate(range(frequency_range[0], frequency_range[1], bands_width)):
            bins.append([freq, freq + bands_width])

        for freq_bin in bins:
            print('')
            print('-' * 40)
            print('Computing the frequency in band: ' + str(freq_bin))
            print('changes are enacted')
            # Filtering the signal and apply the hilbert:
            print('Computing the envelope amplitude')
            band_power = \
                freq_band_raw.copy().filter(
                    freq_bin[0], freq_bin[1], n_jobs=njobs).apply_hilbert(envelope=True)

            # Now, dividing the amplitude by the mean, channel wise:
            if do_baseline_normalization:
                print('Divide by average')
                bands_amp.append(divide_by_average(
                    band_power[:][0], interruption_index))
            else:
                bands_amp.append(band_power[:][0])

        # Deleting unused variables to make some space:
        del band_power
        info = freq_band_raw.info

        # Converting this to 3D numpy array:
        bands_amp = np.array(bands_amp)
        # Finally, all the bands must be averaged back together:
        frequency_band = bands_amp.mean(axis=0)
        # Checking that the averaging across frequency bands works as we want it and if not raise an error:
        for i in range(0, 100):
            ch_ind = np.random.randint(0, bands_amp.shape[1])
            sample_ind = np.random.randint(0, bands_amp.shape[2])
            # Extracting the data of all each freq band and compute the mean thereof:
            test_avg = np.mean(bands_amp[:, ch_ind, sample_ind])
            observer_avg = frequency_band[ch_ind, sample_ind]
            if (test_avg - observer_avg).round(decimals=14) != 0:
                print(test_avg - observer_avg)
                raise Exception("There was an issue in the averaging across frequency bands in the frequency "
                                "bands computations!")
        # Recreating mne raw object:
        frequency_band_signal = mne.io.RawArray(frequency_band, info)
        # Adding back the untouched channels:
        if len(not_picks) != 0:
            frequency_band_signal.add_channels([rest_raw])
    # If the option to do band pass filter is used:
    elif method == "band_pass_filter":
        # Compute the amplitude of hilbert transform in the specified frequency band:
        frequency_band_signal = raw.copy().filter(frequency_range[0], frequency_range[1],
                                                  n_jobs=njobs, picks=picks).apply_hilbert(envelope=True)
        if do_baseline_normalization:
            print('Divide by average')
            # Divide the signal by the average:
            norm_sig = frequency_band_signal.get_data(
            ) / frequency_band_signal.get_data().mean(axis=1)[:, None]
            # Recreate the raw:
            frequency_band_signal = mne.io.RawArray(
                norm_sig, frequency_band_signal.info)
    else:
        raise Exception(
            "You have passed a high gamma computation method that is not supported")

    return frequency_band_signal


def erp_computation(raw, frequency_range=None, njobs=1, channel_types=None, **kwargs):
    """
    The erp computation consist in low passing the raw signal to extract only low freq signal.
    :param raw: (mne raw object) raw mne object containing our raw signal
    :param frequency_range: (list of list of floats) frequency of interest
    :param njobs: (int) number of parallel processes to compute the high gammas
    :param channel_types: (dict) type of channel for which the ERP should be computed. It should be of the format:
    {ecog: True, seeg: True...}
    :param kwargs: arguments that can be passed to the mne raw.filter function. Check this page to find the options:
    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
    :return: erp_raw: (mne raw object) computed erp signal
    frequency bands
    """
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    if frequency_range is None:
        frequency_range = [0, 30]
    print('-' * 40)
    print('Welcome to erp computation')
    print(njobs)
    picks = mne.pick_types(raw.info, **channel_types)
    # Filtering the signal according to the passed parameters:
    erp_raw = raw.copy().filter(
        frequency_range[0], frequency_range[1], n_jobs=njobs, picks=picks, **kwargs)

    return erp_raw


def add_fiducials(raw, fs_directory, subject_id):
    """
    This function add the estimated fiducials to the montage and compute the transformation
    :param raw: (mne raw object) data to which the fiducials should be added
    :param fs_directory: (path string) path to the freesurfer directory
    :param subject_id: (string) name of the subject
    :return: mne raw object and transformation
    """
    montage = raw.get_montage()
    # If the coordinates are in mni_tal coordinates:
    if montage.get_positions()['coord_frame'] == "mni_tal":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        montage.add_mni_fiducials(subjects_dir)
        trans = 'fsaverage'
    else:
        montage.add_estimated_fiducials(subject_id, fs_directory)
        trans = mne.channels.compute_native_head_t(montage)
    raw.set_montage(montage, on_missing="warn")

    return raw, trans


def plot_electrode_localization(mne_object, subject_info, preprocessing_parameters, step_name, subject_id=None,
                                fs_subjects_directory=None, data_type="ieeg", file_extension=".png",
                                channels_to_plot=None, montage_space="T1", plot_elec_name=False):
    """
    This function plots and saved the psd of the chosen electrodes.
    :param mne_object: (mne object: raw, epochs, evoked...) contains the mne object with the channels info
    :param subject_info: (custom object) contains info about the subject
    :param preprocessing_parameters: (custom object) contains the preprocessing info, required to generate the
    save directory
    :param step_name: (string) name of the step that this is performed under to save the data
    :param subject_id: (string) name of the subject! Not necessary if you want to plot in mni space
    :param fs_subjects_directory: (string or pathlib path) path to the free surfer subjects directory. Not required if
    you want to plot in mni space
    :param data_type: (string) type of data that are being plotted
    :param file_extension: (string) extension of the pic file for saving
    :param channels_to_plot: (list) contains the different channels to plot. Can pass list of channels types, channels
    indices, channels names...
    :param montage_space: (string)
    :param plot_elec_name: (string) whethre or not to print the electrodes names onto the snapshot!
    :return:
    """
    if channels_to_plot is None:
        channels_to_plot = ["ecog", "seeg"]
    if fs_subjects_directory is None and montage_space.lower() != "mni":
        raise Exception("For the electrodes plotting, you didn't pass any free surfer directory yet asked to plot the "
                        "electrodes in T1 space. \nThat doesn't work, you should either plot in MNI space or pass a "
                        "freesurfer dir")
    # Adding the estimated fiducials and compute the transformation to head:
    mne_object, trans = add_fiducials(mne_object,
                                      preprocessing_parameters.fs_dir, subject_id)
    # If we are plotting the electrodes in MNI space, fetching the data:
    if montage_space.lower() == "mni":
        fs_subjects_directory = mne.datasets.sample.data_path() + '/subjects'
        subject_id = "fsaverage"
        fetch_fsaverage(subjects_dir=fs_subjects_directory, verbose=True)

    # Set the path to where the data should be saved:
    save_path = path_generator(subject_info.participant_save_root, step_name,
                               signal="raw", previous_steps_list=preprocessing_parameters.analyses_performed,
                               figure=True)
    brain_snapshot_files = []
    # Setting the two views
    snapshot_orientations = {
        "left": {"azimuth": 180, "elevation": None},
        "front": {"azimuth": 90, "elevation": None},
        "right": {"azimuth": 0, "elevation": None},
        "back": {"azimuth": -90, "elevation": None},
        "top": {"azimuth": 0, "elevation": 180},
        "bottom": {"azimuth": 0, "elevation": -180}
    }
    # We want to plot the seeg and ecog channels separately:
    for ch_type in channels_to_plot:
        try:
            data_to_plot = mne_object.copy().pick(ch_type)
        except ValueError:
            mne.utils.warn("You have attempted to plot {0} channels, but none where found in your signal".
                           format(ch_type),
                           RuntimeWarning)
            continue
        # Plotting the brain surface with the electrodes and making a snapshot
        if ch_type == "ecog":
            fig = plot_alignment(data_to_plot.info, subject=subject_id, subjects_dir=fs_subjects_directory,
                                 surfaces=['pial'], coord_frame='head', trans=trans)
        else:
            fig = plot_alignment(data_to_plot.info, subject=subject_id, subjects_dir=fs_subjects_directory,
                                 surfaces=['white'], coord_frame='head', trans=trans)

        for ori in snapshot_orientations.keys():
            if plot_elec_name:
                full_file_name = file_name_generator(save_path, subject_info.files_prefix,
                                                     "elecloc" + ch_type + "_deg" + ori + "_names",
                                                     file_extension,
                                                     data_type=data_type)
            else:
                full_file_name = file_name_generator(save_path, subject_info.files_prefix,
                                                     "elecloc" + ch_type + "_deg" + ori,
                                                     file_extension,
                                                     data_type=data_type)
            mne.viz.set_3d_view(fig, **snapshot_orientations[ori])
            xy, im = snapshot_brain_montage(
                fig, data_to_plot.info, hide_sensors=False)
            fig_2, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(im)
            ax.set_axis_off()
            if plot_elec_name is True:
                for ch in list(xy.keys()):
                    ax.annotate(ch, xy[ch], fontsize=14, color="white",
                                xytext=(-5, -5), xycoords='data', textcoords='offset points')
            plt.savefig(full_file_name, transparent=True)
            plt.close()
            brain_snapshot_files.append(full_file_name)
        mne.viz.close_3d_figure(fig)

    return None


def relocate_fs_folder(source, destination):
    """
    This function copies the participant freesurfer directory to an editable place, inside the derivatives of the bids
    :param source: (path string) path to the source folder to copy
    :param destination: (path string) path to where the folder should by copied
    :return:
    """
    # Copying the fs dir to the correct location:
    shutil.copytree(source, destination, dirs_exist_ok=True)

    return None


def ieeg_t12mni(mne_obj, subjects_dir, subject, template='fsaverage_sym', ch_types=None):
    """
    This function converts the surface electrodes (ECoG) channels position to MNI coordinates
    Created on Thu Mar 30 12:45:21 2023

    @author: simonhenin
    """
    import nibabel as nib
    if ch_types is None:
        ch_types = ["ecog", "seeg"]
    from scipy.spatial import KDTree
    # Prepare dict to save channels loc:
    mni_coords = {ch: None for ch in mne_obj.ch_names}
    # Looping through each channel type:
    for ch_type in ch_types:
        # Handling ecog and seeg differently:
        if ch_type == "ecog":
            # Get only the ecog channels:
            try:
                ecog_obj = mne_obj.copy().pick_types(ecog=True)
            except ValueError:
                mne.utils.warn("No ecog channels for t1 to mni conversion!".
                               format(ch_type),
                               RuntimeWarning)
                continue
            # Extract the channels position:
            ch_pos = ecog_obj.get_montage().get_positions()['ch_pos']
            # When extracting the position directly, they get returned in meters, convert to mm:
            ch_pos = {ch: ch_pos[ch] * 1e3 for ch in ch_pos.keys()}
            # create lookup tables for each hemisphere
            lookup_vertices = {}
            for hemi in ['lh', 'rh']:
                fname = str(Path(subjects_dir, subject, 'surf', hemi + '.pial'))
                vertices, _ = mne.read_surface(fname)
                lookup_vertices[hemi] = KDTree(vertices)

            # Convert each channel to mni:
            for ch in ch_pos.keys():
                # If there are spaces in the channel name, removing it for freesurfer:
                ch_fs = ch.replace(" ", "")
                if np.isnan(ch_pos[ch][0]):
                    mni_coords[ch] = np.array([np.nan, np.nan, np.nan])
                    continue
                if ch_pos[ch][0] < 0:
                    hemi = 'lh'  # this can be determined by the x-coordinate (x < 0 = 'lh')
                else:
                    hemi = 'rh'

                _, nearest_vertex_index = lookup_vertices[hemi].query(ch_pos[ch])

                # write a label file to a temporary directory
                label_file = str(Path(os.getcwd(), 'ch.{}.label'.format(ch_fs)))

                with open(label_file, 'w') as fid:
                    fid.write('%s\n' % (label_file))
                    fid.write('#!ascii label  , from subject %s vox2ras=TkReg\n1\n' % subject)
                    fid.write(
                        '%i %.9f %.9f %.9f 0.0000000' % (nearest_vertex_index, ch_pos[ch][0], ch_pos[ch][1], ch_pos[ch][2]))

                # run mri_label2label
                trglabel = str(Path(os.getcwd(), 'ch.{}.converted.label'.format(ch_fs)))
                os.system('mri_label2label --srclabel ' + label_file + ' --srcsubject ' + subject + \
                          ' --trgsubject ' + template + ' --trglabel ' + trglabel + ' --regmethod surface --hemi ' + hemi + \
                          ' --trgsurf pial --paint 6 pial --sd ' + str(subjects_dir))

                # get the new coordinates from the trglabel file
                with open(trglabel, 'r') as fid:
                    coord = fid.readlines()[2].split()  # Get the third line
                mni_coords[ch] = np.array(coord[1:-1]).astype(np.float) * 1e-3

                # Remove the temporary files:
                os.remove(label_file)
                os.remove(trglabel)

        if ch_type == "seeg":
            # Get only the ecog channels:
            try:
                seeg_obj = mne_obj.copy().pick_types(seeg=True)
            except ValueError:
                mne.utils.warn("No seeg channels for t1 to mni conversion!".
                               format(ch_type),
                               RuntimeWarning)
                continue
            # Extract the channels position:
            ch_pos = seeg_obj.get_montage().get_positions()['ch_pos']
            # Convert to a numpy array:
            t1_coords = np.array([ch_pos[ch] * 1e3 for ch in ch_pos])

            # read in the talairach file in manually
            talxfm_file = Path(subjects_dir, subject, 'mri', 'transforms', 'talairach.xfm')
            xfm = []
            with open(talxfm_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):

                    if line.find('Linear_Transform') >= 0:
                        print(i)
                        # get the next 3 lines and exit
                        xfm = lines[i + 1::]
                        xfm[-1] = xfm[-1].replace(';', '')
            xfm = [x.replace('\n', '') for x in xfm]
            xfm = [x.split(' ') for x in xfm]
            xfm = np.asarray(xfm, dtype=float)
            assert xfm.shape == (3, 4), 'Something went wrong reading in the talairach.xfm file'

            # some other stuff we need from T1 header
            t1w = nib.load(Path(subjects_dir, subject, 'mri', 'orig.mgz'))
            n_orig = t1w.header.get_vox2ras()
            t_orig = t1w.header.get_vox2ras_tkr()

            # Compute the transform:
            B = np.append(t1_coords, np.ones([t1_coords.shape[0], 1]), axis=1)
            mni_tal = np.dot(np.dot(xfm, n_orig), np.dot(np.linalg.inv(t_orig), B.T)).T

            for ind, ch in enumerate(ch_pos.keys()):
                mni_coords[ch] = mni_tal[ind, 0:3] * 1e-3

    # Convert these new coordinates to a dataframe to save to tsv:
    mni_coords = pd.DataFrame(mni_coords, index=["x", "y", "z"]).T
    mni_coords["size"] = None

    return mni_coords


def bad_channels_to_bids(subject, bids_path, bad_channels, description="outside_brain"):
    """
    This function loads the bids channels.tsv and for all channel passed in bad_channels and adds a description
    in the file. That way, the file can then be parsed to reject channels based on this information later on!
    :param subject: (string) name of the subject
    :param bids_path: (mne bids bidsPath object) contains the path to the bids data of that subject, this is where
    the channels.tsv lives
    :param bad_channels: (list of strings) list of the channel names to describe
    :param description: (string) description to add for this channels
    :return: None, things are saved to file
    """

    # Looking for the channel file:
    channel_tsv_file = find_files(bids_path.directory, naming_pattern="*_channels", extension=".tsv")
    if len(channel_tsv_file) == 0:
        print("WARNING: No channel tsv found for sub-{}!".format(subject))
        return
    assert len(channel_tsv_file) == 1, "The number of *_channels.tsv does not match what is expected!"
    # Loading the channels tsv:
    channel_tsv = pd.read_csv(channel_tsv_file[0], sep="\t")

    # Making sure that all the annotated channels exist in the bids file:
    non_existing_channels = [channel
                             for channel in bad_channels
                             if channel not in channel_tsv["name"].to_list()]
    if len(non_existing_channels) > 0:
        raise Exception("The following channels do not exist in sub-{} channels.tsv:"
                        "\n{}"
                        "\nCheck spelling!".format(subject, non_existing_channels))

    # We can now loop through the annotated channels to extend the annotations:
    for channel in bad_channels:
        # Find whether there is already an existing annotation for this channel:
        tsv_annotation = channel_tsv.loc[channel_tsv["name"] == channel, "status_description"].item()
        # Checking whether the annotation from the one file is already present in the tsv annotation:
        if isinstance(tsv_annotation, float) and math.isnan(tsv_annotation):
            ch_annot = description
        elif description in tsv_annotation:
            print("Channel {} was already annotated as {} in the bids channels tsv. It will be skipped".format(
                channel, description))
            continue
        else:
            # Otherwise, appending the new annotation to the existing one:
            ch_annot = "/".join([tsv_annotation, description])
        # And editing the channels tsv:
        channel_tsv.loc[channel_tsv["name"] == channel, "status_description"] = ch_annot

    # Saving the annotation to the bids channels tsv file:
    channel_tsv.to_csv(channel_tsv_file[0], index=False, sep="\t")

    return None


def roi_mapping(mne_object, list_parcellations, subject_id, fs_dir, step,
                subject_info, preprocessing_parameters):
    """
    This function maps the electrodes on different atlases. You can pass whatever atlas you have the corresponding
    free surfer parcellation for.
    :param mne_object: (mne raw or epochs object) object containing the montage to extract the roi
    :param list_parcellations: (list of string) list of the parcellation files to use to do the mapping. Must match the
    naming of the free surfer parcellation files.
    :param subject_id: (string) name of the subject to do access the fs recon
    :param fs_dir: (string or pathlib path object) path to the freesurfer directory containing all the participants
    :param step: (string) name of the step to save the data accordingly
    :param subject_info: (custom object) contains info about the patient
    :param preprocessing_parameters: (custom object) contains info about the preprocessing parameters, required to
    generate the save directory
    :return: labels_df: (dict of dataframe) one data frame per parcellation with the mapping between roi and channels
    """
    from freesurfer.wang_labels import get_montage_volume_labels_wang
    labels_df = {parcellation: pd.DataFrame()
                 for parcellation in list_parcellations}
    for parcellation in list_parcellations:
        if mne_object.get_montage().get_positions()['coord_frame'] == "mni_tal":
            sample_path = mne.datasets.sample.data_path()
            subjects_dir = Path(sample_path, 'subjects')
            # Convert the montge from mni to mri:
            montage = mne_object.get_montage()
            montage.apply_trans(mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))
            if parcellation == "wang15_mplbl":
                labels, _ = get_montage_volume_labels_wang(montage, "fsaverage", subjects_dir=None,
                                                           aseg='wang15_mplbl', dist=2)
            else:
                labels, _ = \
                    mne.get_montage_volume_labels(montage, "fsaverage", subjects_dir=subjects_dir, aseg=parcellation)
        else:
            if parcellation == "wang15_mplbl":
                labels, _ = get_montage_volume_labels_wang(mne_object.get_montage(), subject_id, subjects_dir=fs_dir,
                                                           aseg=parcellation, dist=2)
            else:
                labels, _ = mne.get_montage_volume_labels(
                    mne_object.get_montage(), subject_id, subjects_dir=fs_dir, aseg=parcellation)
        # Appending the electrodes roi to the table:
        for ind, channel in enumerate(labels.keys()):
            labels_df[parcellation] = labels_df[parcellation].append(
                pd.DataFrame({"channel": channel, "region": "/".join(labels[channel])}, index=[ind]))
    # Saving the results. This step is completely independent from what happened on the data:
    save_path = path_generator(subject_info.participant_save_root, step,
                               signal="raw", previous_steps_list=preprocessing_parameters.analyses_performed,
                               figure=False)
    # Creating the directory if it doesn't exists:
    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)
    # Looping through the different mapping:
    for mapping in labels_df.keys():
        file_name = file_name_generator(save_path, subject_info.files_prefix, "elecmapping_" + mapping, ".csv",
                                        data_type="ieeg")
        # Saving the corresponding mapping:
        labels_df[mapping].to_csv(Path(file_name), index=False)

    return labels_df


def description_ch_rejection(raw, bids_path, channels_description, subject_info, discard_bads=True):
    """
    This function enables to discard channels based on the descriptions found in the _channel.tsv file in the bids.
    A string or list of strings must be passed to be compared to the content of the _channel file to discard those
    matching
    :param raw: (mne_raw object) contains the data and channels to investigate
    :param bids_path: (mne_bids object) path to the _channel.tsv file
    :param channels_description: (str or list) contain the channels descriptions to set as bad channels.
    :param subject_info: (subject_info object) contains info about the participants
    :param discard_bads: (boolean) whether or not to discard the channels that were marked as bads as well
    :return:
    """
    if isinstance(channels_description, str):
        channels_description = [channels_description]

    # Loading the channels description file:
    channel_info_file = find_files(
        bids_path.directory, naming_pattern="*_channels", extension=".tsv")[0]
    channel_info = pd.read_csv(channel_info_file, sep="\t")
    # Looping through the passed descriptions:
    bad_channels = []
    for desc in channels_description:
        desc_bad_channels = []
        subject_info.desc_bad_channels[desc] = []
        # Looping through each channel:
        for ind, row in channel_info.iterrows():
            if isinstance(row["status_description"], str):
                if desc in row["status_description"]:
                    bad_channels.append(row["name"])
                    desc_bad_channels.append(row["name"])
        subject_info.desc_bad_channels[desc].extend(desc_bad_channels)

    # Discarding the channels that were marked as bad as well!
    if discard_bads:
        for ind, row in channel_info.iterrows():
            if row["status"] == "bad":
                bad_channels.append(row["name"])
    # Remove any redundancies:
    bad_channels = list(set(bad_channels))
    # Set these channels as bad:
    raw.info['bads'].extend(bad_channels)

    return raw, bad_channels


def laplace_mapping_validator(mapping, data_channels):
    """
    This function checks the mapping against the channels found in the data. If things don't add up, raise an error
    :param mapping: (dict) contains the mapping between channels to reference and which channels to use to do the
    reference. Format: {ch_name: {"ref_1": ch, "ref_2": ch or None}}
    :param data_channels: (list of string) list of the channels found in the data. This is used to control that all
    the channels found in the mapping are indeed found in the data, to avoid issues down the line
    :return:
    """
    if not all([channel in data_channels for channel in mapping.keys()]) \
            or not all([mapping[channel]["ref_1"] in data_channels or mapping[channel]["ref_1"] is None
                        for channel in mapping.keys()]) \
            or not all([mapping[channel]["ref_2"] in data_channels or mapping[channel]["ref_2"] is None
                        for channel in mapping.keys()]):
        # Printing the name of the channels that are not present in the data:
        print("The following channels are present in the mapping but not in the data:")
        print([channel for channel in mapping.keys() if channel not in data_channels])
        print([mapping[channel]["ref_1"]
               for channel in mapping.keys() if mapping[channel]["ref_1"] not in data_channels])
        print([mapping[channel]["ref_2"]
               for channel in mapping.keys() if mapping[channel]["ref_2"] not in data_channels])
        raise Exception("The mapping contains channels that are not in the data!")
    # Checking that there is never the case of having both reference as None:
    if any([mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None for channel in mapping.keys()]):
        invalid_channels = [channel for channel in mapping.keys() if
                            mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None]
        mne.utils.warn("The channels {0} have two None reference. They will be set to bad! If this is not intended,"
                       "please review your mapping!".format(invalid_channels))

    return None


def remove_bad_references(reference_mapping, bad_channels, all_channels):
    """
    The reference mapping in the mapping file is agnostic to which channels are bad, it is only based on the grid
    and strips organization. This function integrates the bad channels information to the mapping. With laplace mapping,
    there are two cases in which a channel should be rejected: if the channel being referenced is bad or if both
    reference channels are bad. This channels identifies these cases and updates the mapping such that those channels
    are excluded. Note that in the case where only one of the two reference is bad but not the other, only the other
    will be used, reducing to a bipolar.
    Additionally, if some of the channels found in the data are not found in the mapping, the data can
    be set as bad if the discard_no_ref_channels is set to True
    :param reference_mapping: (dict) contain for each channel the reference channel according to our scheme
    :param bad_channels: (string list) list of the channels that are marked as bad
    :param all_channels: (string list) list of all the channels
    :return:
    """
    new_reference_mapping = reference_mapping.copy()
    # Looping through each channel to reference to combine bad channels information:
    for channel in reference_mapping.keys():
        # If the channel being referenced is bad, then it is bad:
        if channel in bad_channels:
            print("The channel {} to reference is bad and will be ignored".format(channel))
            new_reference_mapping.pop(channel)
            continue
        # If both the references are bad, setting the channel to bad too, as it can't be referenced!
        elif reference_mapping[channel]["ref_1"] in bad_channels \
                and reference_mapping[channel]["ref_2"] in bad_channels:
            print("The reference channels {} and {} to reference {} are both bad and {} cannot be referenced".format(
                reference_mapping[channel]["ref_1"], reference_mapping[channel]["ref_2"], channel, channel))
            new_reference_mapping.pop(channel)
            continue
        # But if only one of the two reference is bad, setting that one ref to None:
        elif reference_mapping[channel]["ref_1"] in bad_channels:
            new_reference_mapping[channel]["ref_1"] = None
        elif reference_mapping[channel]["ref_2"] in bad_channels:
            new_reference_mapping[channel]["ref_2"] = None

        # As a result of setting one of the reference to None if bad, some channels located close to edges might have
        # only None as references, in which case they can't be referenced. This channels need to be removed from the
        # mapping
        if new_reference_mapping[channel]["ref_1"] is None and new_reference_mapping[channel]["ref_2"] is None:
            print("The reference channels {} cannot be referenced, because both surrounding channels are None as a "
                  "result of bad channels removal".format(channel))
            new_reference_mapping.pop(channel)

    # Removing any duplicates there might be (some channels might be both bad themselves and surrounded by bads):
    new_bad_channels = [channel for channel in all_channels if channel not in list(new_reference_mapping.keys())]
    # Compute the proportion of channels that are bad because surrounded by bad:
    laplace_bad_cnt = len(new_bad_channels) - len(bad_channels)
    # Print some info about what was discarded:
    print("{0} dropped because bad or surrounded by bad".format(new_bad_channels))
    print("{} bad channels".format(len(bad_channels)))
    print("{} surrounded by bad channels".format(laplace_bad_cnt))

    return new_reference_mapping, new_bad_channels


def laplace_ref_fun(to_ref, ref_1=None, ref_2=None):
    """
    This function computes the laplace reference by subtracting the mean of ref_1 and ref_2 to the channel to reference:
    ch = ch - mean(ref_1, ref_2)
    The to-ref channel must be a matrix with dim: [channel, time]. The ref_1 and ref_2 must have the same dimensions AND
    the channel rows must match the order. So if you want to reference G2 with G1 and G2 and well as G3 with G2 and G4,
    then your matrices must be like so:
    to_ref:                ref_1              ref_2
    [                 [                       [
    G2 ..........     G1 ..............       G3 ..............
    G3 ..........     G2 ..............       G4 ..............
    ]                 ]                       ]
    Otherwise, you would be referencing  wrong triplets!
    :param to_ref: (numpy array) contains the data to reference.
    :param ref_1: (numpy array) Contains the first ref to do the reference (the ref_1 in mean(ref_1, ref_2)). Dimension
    must match to_ref
    :param ref_2: (numpy array) Contains the second ref to do the reference (the ref_2 in mean(ref_1, ref_2)). Dimension
    must match to_ref
    :return: referenced_data (numpy array) data that were referenced
    """
    # Check that the sizes match:
    if not to_ref.shape == ref_1.shape == ref_2.shape:
        raise Exception("The dimension of the data to subtract do not match the data!")
    referenced_data = to_ref - np.nanmean([ref_1, ref_2], axis=0)

    return referenced_data


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


def laplacian_referencing(raw, reference_mapping, channel_types=None,
                          n_jobs=1, relocate_edges=True,
                          subjects_dir=None, subject=None, montage_space=None):
    """
    This function performs laplacian referencing by subtracting the average of two neighboring electrodes to the
    central one. So for example, if you have electrodes G1, G2, G3, you can reference G2 as G2 = G2 - mean(G1, G2).
    The user can pass a mapping in the format of a dictionary. The dictionary must have the structure:
    {
    "ch_to_reference": {
        "ref_1": "ref_1_ch_name" or None,
        "ref_1": "ref_1_ch_name" or None,
        },
    ...
    }
    If the user doesn't pass a mapping, he will be given the opportunity to generate it manually through command line
    input. If no mapping exists, we recommend using the command line to generate it, as the formating is then readily
    consistent with the needs of the function. The function so far only works with raw object, not epochs nor evoked
    :param raw: (mne raw object) contains the data to reference
    :param reference_mapping: (dict or None) dict of the format described above or None. If None, then the user will
    have the opportunity to create it manually
    :param channel_types: (dict) which channel to consider for the referencing
    :param n_jobs: (int) n_jobs to compute the mapping. Not really useful as we loop through each channel independently
    so setting it to more than 1 will not really do anything. But might be improved in the future.
    :param relocate_edges: (boolean) whether or not to relocate the electrodes that have only one ref!
    :param subjects_dir: (string) directory to the freesurfer data. This is necessary, as the edges get relocated,
    the ecog channels need to be projected to the brain surface.
    :param subject: (string) Name of the subject to access the right surface
    :param montage_space: (string) name of the montage space of the electrodes, either T1 or MNI
    :return:
    mne raw object: with laplace referencing performed.
    """
    # Get the channels of interest.
    if channel_types is None:
        channel_types = {"ecog": True, "seeg": True, "exclude": []}
    channels_of_int = [raw.ch_names[ind] for ind in mne.pick_types(raw.info, **channel_types)]

    # Validate the reference mapping:
    laplace_mapping_validator(reference_mapping, channels_of_int)
    # Adjust reference mapping based on bad channels information:
    reference_mapping, bad_channels = remove_bad_references(reference_mapping, raw.info["bads"], channels_of_int)

    # ------------------------------------------------------------------------------------------------------------------
    # Performing the laplace reference:
    # Extract data to get the references and avoid issue with changing in place when looping:
    ref_data = raw.get_data()
    data_chs = raw.ch_names
    # Get the size of a channel matrix to handle the absence of ref_2 for corners:
    mat_shape = np.squeeze(raw.get_data(picks=0)).shape
    empty_mat = np.empty(mat_shape)
    empty_mat[:] = np.nan
    # performing the laplace referencing:
    for ch in reference_mapping.keys():
        if reference_mapping[ch]["ref_1"] is None:
            ref_1 = empty_mat
        else:
            # Get the index of the reference channel:
            ind = data_chs.index(reference_mapping[ch]["ref_1"])
            ref_1 = np.squeeze(ref_data[ind, :])
        if reference_mapping[ch]["ref_2"] is None:
            ref_2 = empty_mat
        else:
            # Get the index of the reference channel:
            ind = data_chs.index(reference_mapping[ch]["ref_2"])
            ref_2 = np.squeeze(ref_data[ind, :])
        raw.apply_function(laplace_ref_fun, picks=[ch], n_jobs=n_jobs, verbose=None,
                           ref_1=ref_1, ref_2=ref_2,
                           channel_wise=True)
        # Relocating if needed:
        if relocate_edges:
            # If one of the two reference is only Nan, then there was one ref missing, in which case the channel must
            # be replaced, (bitwise or because only if one of the two is true)
            if np.isnan(ref_1).all() ^ np.isnan(ref_2).all():
                print("Relocating channel " + ch)
                montage = raw.get_montage()
                # Get the indices of the current channel
                ch_ind = montage.ch_names.index(ch)
                # Get the single reference:
                ref = reference_mapping[ch]["ref_1"] if reference_mapping[ch]["ref_1"] is not None \
                    else reference_mapping[ch]["ref_2"]
                ref_ind = montage.ch_names.index(ref)
                # Compute the center between the two:
                x, y, z = (montage.dig[ch_ind]["r"][0] + montage.dig[ref_ind]["r"][0]) / 2, \
                          (montage.dig[ch_ind]["r"][1] + montage.dig[ref_ind]["r"][1]) / 2, \
                          (montage.dig[ch_ind]["r"][2] + montage.dig[ref_ind]["r"][2]) / 2,
                montage.dig[ch_ind]["r"] = np.array([x, y, z])
                # Adding the montage back:
                raw.set_montage(montage, on_missing="warn")

    # Projecting the ecog channels to the surface if they were relocated:
    if relocate_edges:
        if len(mne.pick_types(raw.info, ecog=True)) > 0:
            project_elec_to_surf(raw, subjects_dir, subject, montage_space=montage_space)

    return raw, reference_mapping, bad_channels


def remove_channel_tsv_description(channel_tsv_df, description):
    """
    This function removes a specific description from a bids channel tsv file. This is useful if previous iterations
    added wrong annotation to the channels tsv
    :param channel_tsv_df: (df) mne bids channel tsv pandas data frame
    :param description: (string) description to remove
    :return:
    """
    # Find all the channels for which the current description is found
    desc_channels = channel_tsv_df.loc[channel_tsv_df["status_description"].str.contains(description, na=False),
                                       "name"].to_list()
    for channel in desc_channels:
        # Get the description string of the current channel:
        ch_desc = channel_tsv_df.loc[channel_tsv_df["name"] == channel, "status_description"].item().split("/")
        # Remove the current description:
        ch_desc.remove(description)
        # Add the cleaned string back in:
        channel_tsv_df.loc[channel_tsv_df["name"] == channel, "status_description"] = "/".join(ch_desc)

    return channel_tsv_df


def annotate_channels_tsv(bids_path, channels, description, overwrite=True):
    """
    This functions adds specific annotation to the channels .tsv in the bids. The overwite option enables locating any
    electrode that was already described in the exact same way as the the current description to overwrite it to
    start fresh.
    :param bids_path: (mne_bids BIDS_path object) contains the path to where the channels tsv can be found
    :param channels: (list) list of channels to annotate
    :param description: (string) description of all the channels passed
    :param overwrite: (bool) whether or not to remove the current description from the channels tsv before adding it
    anew, to remove what was added from previous iterations
    :return:
    """
    # Find the channels.tsv file:
    channel_tsv_file = find_files(bids_path.directory, naming_pattern="*_channels", extension=".tsv")
    assert len(channel_tsv_file) == 1, "There were more or less than 1 channels.tsv file found for this subject!"
    # Loading the channels tsv:
    channel_tsv = pd.read_csv(channel_tsv_file[0], sep="\t")

    # Cleaning the channel tsv from the current description if required:
    if overwrite:
        channel_tsv = remove_channel_tsv_description(channel_tsv, description)

    # Making sure that all the annotated channels exist in the bids file:
    non_existing_channels = [channel
                             for channel in channels
                             if channel not in channel_tsv["name"].to_list()]
    if len(non_existing_channels) > 0:
        raise Exception("The following channels do not exist in channels.tsv:"
                        "\n{}"
                        "\nCheck spelling!".format(non_existing_channels))

    # Now, looping through each channel:
    for channel in channels:
        tsv_annotation = channel_tsv.loc[channel_tsv["name"] == channel, "status_description"].item()
        # Check if the
        if isinstance(tsv_annotation, float) and math.isnan(tsv_annotation):
            ch_annot = description
        elif description in tsv_annotation:
            print("Channel {} was already annotated as {} in the bids channels tsv. It will be skipped".format(
                channel, description))
            continue
        else:
            # Otherwise, appending the new annotation to the existing one:
            ch_annot = "/".join([tsv_annotation, description])

        # Adding the annotation to this channel:
        channel_tsv.loc[channel_tsv["name"] == channel, "status_description"] = ch_annot

    # Save the channel tsv back:
    channel_tsv.to_csv(channel_tsv_file[0], index=False, sep="\t")

    return None
