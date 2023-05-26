""" This script contains all the helper functions for the visual responsiveness
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Katarina Bendtz, Simon Henin
    katarina.bendtz@tch.harvard.edu
    Simon.Henin@nyulangone.org
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel, ranksums, wilcoxon

import pingouin as pg

import mne.stats

from general_helper_functions.pathHelperFunctions import find_files
from general_helper_functions.data_general_utilities import (baseline_scaling,
                                                             compute_dependent_variable,
                                                             load_epochs)


def test_sustained_zscore(y, threshold=2.5, window_sec=0.050, sr=512, alternative="two_tailed"):
    """
    This function checks if a specific zscore threshold is exceeded in consecutive samples for a specific duration.
    So say you want to test whether in a specific list, you want to test whether a threshold is exceeded for a duration
    of 50ms, this function will tell you. This function returns whether or not somewhere in a time series the
    threshold is exceeded in consecutive samples for the set duration.
    :param y: list or 1D array (of zscores)
    :param threshold: (float) threshold to test the y data against
    :param window_sec: (float or int) duration of the window over which the threshold must be exceeded for the
    test to be considered passed.
    :param sr: (int) sampling rate of the signal in Hz
    :param alternative: (string) upper, lower or two tailed test. In other words, whether you want to only test if the
    threshold is exceeded in the postive (i.e. higher than), in the negative (lower than the inverse of the thresh)
    or in both directions. So for example, say your y data go from -5 to 5 and want to test against 2.5 threshold, if
    you say "upper", will be considered only exceeding of the threshold above 0 (i.e. anything above 2.5), if you say
    "lower", anything below -2.5, and if you say "two_tailed", the sign is not considered: any exceedence of +-2.5 will
    be considered!
    :return:
    h0: boolean: whether or not the null hypothesis (no exceedence of threshold) is rejected: False: not rejected, True:
    rejected
    [onset_sec, offset_sec]: list of floats: the onset and offset times in seconds of where the threshold was exceeded.
    The offset sample is found in the case where the threshold is found to be exceeded for the stated duration or longer
    by looking for the first sample where the data go below threshold again. CAREFUL: the onset and offset are computed
    by taking the first sample as t0. If your data were cropped before feeding into that function, you need to adjust by
    adding the onset of your cropping. Say you cropped from 50 to 300ms, you need to add 50ms to the onset and offsets
    [onset_samp, offset_samp]: same as above but in sample instead of seconds
    """
    # Handling data dimensions
    if isinstance(y, np.ndarray):
        if len(y.shape) > 1:
            raise Exception("You have passed an numpy array of more than 1D! This function only works with 1D numpy "
                            "array or unnested list!")
    elif isinstance(y, list):
        if isinstance(y[0], list):
            raise Exception("You have passed a nested list! This function only works with 1D numpy "
                            "array or unnested list!")
        elif isinstance(y[0], np.ndarray):
            raise Exception("You have passed a list of numpy arrays!This function only works with 1D numpy "
                            "array or unnested list!")
    if threshold > 0 and alternative == "lower":
        print("WARNING: You are looking for something that is below {} zscore, which is a positive value."
              "\nBeing below a positive zscore value means being towards no difference, which is a very weird "
              "\nthing to do! Be careful!".format(threshold))
    # Convert the time window from ms to samples:
    window_samp = int(window_sec * (sr / 1))
    # Binarizing the data according to the tail we are interested in
    if alternative == "upper":  # Upper tail means we are looking for sample above threshold
        y_bin = y > threshold
    elif alternative == "lower":  # Lower tail means we are looking for sample below threshold
        y_bin = y < threshold
    elif alternative == "two_tailed":  # For two tailed, we are looking for samples above the threshold or below the
        # negative of the threshold, i.e. values superior to the absolute of the zscore
        y_bin = np.absolute(y) > threshold
    else:
        raise Exception("You have passed {} as an alternative. Only upper, lower or two_tailed are supported!"
                        "".format(alternative))
    # Set h0
    h0 = True
    # Looping through each True in the binarize y:
    for ind in np.where(y_bin)[0]:
        if ind + window_samp < len(y_bin):
            if all(y_bin[ind:ind + window_samp]):
                h0 = False
                # Finding the offset of the significant window:
                onset_samp = ind
                if len(np.where(np.diff(y_bin[ind:].astype(int)) == -1)[0]) > 0:
                    offset_samp = onset_samp + np.where(np.diff(y_bin[ind:].astype(int)) == -1)[0][0]
                else:
                    offset_samp = len(y) - 1
                # Convert to me:
                onset_sec, offset_sec = onset_samp * (1 / sr), offset_samp * (1 / sr)
                break
        else:
            break
    if h0:
        onset_samp, offset_samp = None, None
        onset_sec, offset_sec = None, None
    return h0, [onset_sec, offset_sec], [onset_samp, offset_samp]


def aggregated_stat_test(data_df, twin, groups="channel", test="wilcoxon", alternative="upper",
                         p_val=0.05):
    """
    This function performs "aggregated stats", i.e. classical statistical tests like wilcoxon or t_test on averaged
    data over time windows. This function takes in a data frame as well as a string describing the test to perform
    as well as a couple parameters for the said test.
    :param data_df: (pandas data frame) contains the data to be tested. The dataframe should have a column called
    value containing the values to compare, a column called condition containing a string describing to which condition
    a given value belongs. There should only be two conditions. Furthermore, the dataframe should have a column named
    whatever you like specified by the variable "groups" that contain a string describing to which group a set of values
    belong to which a separate test should be performed. Practically, if you want to test baseline vs onset period
    for  every electrode, the group can be "channels" to perform a test for each channel.
    :param twin: (list of strings) contains the time window being tested. This is just to format the table
    :param groups: (string) column name from the data_df containing the group to which the test should be fitted
    separately.
    :param test: (string) name of the test. Shouuld be either "t_test" or "wilcoxon", other tests are not supported as
    of now.
    :param alternative: (string) alternative of the test: upper vs lower tail.
    :param p_val: (float) p value threshold to consider something significant.
    :return:
    results_df: a pandas data frame storing the results of the test
    """
    print("=" * 40)
    print("Welcome to aggregated_stat_test")
    # Looping through each group:
    results_df = pd.DataFrame()
    for group in data_df[groups].unique():
        print("Performing test for group: {}".format(group))
        # Extract the data of this group:
        group_data = data_df.loc[data_df[groups] == group]
        # Extract the two conditions:
        y_data = [group_data.loc[group_data["condition"] == cond, "value"].to_list()
                  for cond in ["onset", "baseline"]]
        if test.lower() == "t-test" or test.lower() == "t_test" or test.lower() == "t test":
            statistic, pvalue = ttest_rel(*y_data, alternative=alternative)
        elif test.lower() == "wilcoxon_rank_sum":
            statistic, pvalue = ranksums(*y_data, alternative=alternative)
        elif test.lower() == "wilcoxon_signed_rank":
            x = np.array(y_data[0]) - np.array(y_data[1])
            statistic, pvalue = wilcoxon(x, alternative=alternative)
        elif test.lower() == "bayes_t_test":
            t_val, pvalue = ttest_rel(*y_data, alternative=alternative)
            # Call the bayes factor the statistics to be consistent with the rest:
            statistic = pg.bayesfactor_ttest(t_val, len(y_data[0]),
                                             paired=True, alternative=alternative,
                                             r=0.7071)
            # Set the pvalue to 0.01 if the bayes factor is more than 3 as our threshold is significance.
            # This is more for backward compatibility than anything else:
            pvalue = 0.01 if statistic > 3 else 0.5
        else:
            raise Exception("This test is not supported!")
        # Compute the effect size if the test wasn't the bayes t-test:
        if test.lower() != "bayes_t_test":
            # https://en.wikipedia.org/wiki/Effect_size
            effect_strength = ((np.mean(y_data[0]) - np.mean(y_data[1])) / np.mean(y_data[1])) * 100
        else:
            # Otherwise, set the effect strength as the posterior probability. Not exactly correct,
            # but keeps things consistent with down the line
            effect_strength = 1 / (1 + math.exp(-math.log(statistic)))
        h0 = True if pvalue > p_val else False
        results_df = results_df.append(pd.DataFrame({
            "subject": group.split("-")[0],
            "channel": group,
            "metric": None,
            "reject": not h0,
            "stat": statistic,
            "pval": pvalue,
            "onset": twin[0] if not h0 else None,
            "offset": twin[1] if not h0 else None,
            "effect_strength": effect_strength
        }, index=[0]))
    return results_df.reset_index(drop=True)


def sustained_zscore_test(data_df, onset, groups="channel", z_thresh=2.5, dur_thresh=0.050,
                          alternative="upper", sfreq=512):
    """
    This function perform a "sustained_zscore" test, checking if a given time series of zscore exceeds a zscore
    threshold for longer than a duration threshold.
    :param data_df: (dataframe) contains the data to be tested. Each row should contain the value of a given group.
    :param onset: (float) time from which this is investigated to make the onset be correct.
    :param groups: (string) name of the column containing the group variable for which to perform the test.
    :param z_thresh: (float) zscore threshold to consider something relevant
    :param dur_thresh: (float) duration that the exceedence of the zscore threshold needs to last to be considered
    relevant (in seconds)
    :param alternative: (string) upper, lower or both, i.e. the tails.
    :param sfreq: (float) sampling frequency of the time series to be able to compute the duration correctly.
    :return:
    results_df: a pandas data frame storing the results of the test
    """
    print("=" * 40)
    print("Welcome to sustained_zscore_test")
    # Var to store the results
    test_results = pd.DataFrame()
    for group in data_df[groups].unique():
        print("Performing test for group: {}".format(group))
        # Get the data of this group:
        y = data_df.loc[data_df[groups] == group, "values"].item()
        # Check whether the zscore exceed threshold for the stated duration
        h0, sig_window_sec, sig_window_samp = test_sustained_zscore(y,
                                                                    threshold=z_thresh,
                                                                    window_sec=dur_thresh,
                                                                    sr=sfreq,
                                                                    alternative=alternative)
        # Compute the strength of the effect by averaging over the time window that was found of interest:
        if not h0:
            effect_strength = np.mean(y[sig_window_samp[0]:sig_window_samp[1]])
        else:
            effect_strength = None
        # Create results table:
        test_results = test_results.append(pd.DataFrame({
            "subject": group.split("-")[0],
            "channel": group,
            "metric": None,
            "stat": None,
            "pval": None,
            "reject": not h0,
            "onset": onset + sig_window_sec[0] if sig_window_sec[0] is not None
            else None,
            "offset": onset + sig_window_sec[1] if sig_window_sec[0] is not None
            else None,
            "effect_strength": effect_strength
        }, index=[0]))
    return test_results


def cluster_based_permutation_test(data_df, sfreq, groups="channel", onset=0, n_perm=1048, p_val=0.05):
    """
    This function performs a 1 sample cluster based permutation test.
    :param data_df: (dataframe) contains the data to be tested. Each row should contain the value of a given group.
    :param p_val: (float) p value threshold to consider something significant.
    :param sfreq: (float) sampling frequency of the time series to be able to compute the duration correctly.
    :param onset: (float) time from which this is investigated to make the onset be correct.
    :param n_perm: (int) number of permutations to perform
    :param groups: (string) name of the column containing the group variable for which to perform the test.
    :return:
    results_df: a pandas data frame storing the results of the test
    """
    print("=" * 40)
    print("Welcome to cluster_based_permutation_test")
    test_results = pd.DataFrame()

    for group in data_df[groups].unique():
        print("Performing test for group: {}".format(group))
        x = data_df.loc[data_df[groups] == group, "values"].item()
        t_obs, clusters, cluster_pv, h0 = mne.stats.permutation_cluster_1samp_test(x,
                                                                                   n_permutations=n_perm)
        # Extract the significance and onset offset:
        h0 = True
        obs_pval = None
        onset_sec = None
        offset_sec = None
        effect_strength = None
        for ind, cluster in enumerate(clusters):
            if cluster_pv[ind] < p_val:
                h0 = False
                # Get cluster begining and end:
                onset_sec, offset_sec = cluster[0][0] * (1 / sfreq), \
                                        cluster[0][-1] * (1 / sfreq)
                effect_strength = \
                    np.mean(np.mean(x, axis=0)[cluster[0][0]:cluster[0][-1]])
                obs_pval = cluster_pv[ind]
                break
        # If no clusters were found to be significant:
        if h0 is True:
            # If clusters were found:
            if len(cluster_pv) > 0:
                # Take the lowest cluster p value:
                obs_pval = min(cluster_pv)
            else:
                # otherwise set p value to 1
                obs_pval = 1
        # Generate a result table:
        test_results = test_results.append(pd.DataFrame({
            "subject": group.split("-")[0],
            "channel": group,
            "metric": None,
            "reject": not h0,
            "stat": None,
            "pval": obs_pval,
            "onset": onset + onset_sec if onset_sec is not None
            else onset_sec,
            "offset": onset + offset_sec if onset_sec is not None
            else onset_sec,
            "effect_strength": effect_strength
        }, index=[0]))
    return test_results.reset_index(drop=True)


def get_mni_coordinates(bids_path, picks):
    """
    This function generates a dataframe containing the channels names, types and mni coordinates
    :param bids_path: (mne_bids bids path object) path to the bids directory of the given subject and task
    :param picks: (list of string) list of the channels names to pick
    :return:
    mni_coordinates: (pd dataframe) contains the channel names, types and x y z coordinates in mni space
    """
    coord_file = "*space-fsaverage_electrodes"
    # Loading the coordinate file:
    recon_file = find_files(bids_path.directory,
                            naming_pattern=coord_file, extension=".tsv")
    channel_info_file = find_files(
        bids_path.directory, naming_pattern="*channels", extension=".tsv")
    # Load the file:
    channels_coordinates = pd.read_csv(
        recon_file[0], sep='\t')  # Loading the coordinates
    channel_info = pd.read_csv(channel_info_file[0], sep='\t')

    selected_channels = channels_coordinates.loc[channels_coordinates["name"].isin(picks)]
    mni_coordinates = pd.DataFrame()
    for channel in picks:
        x = selected_channels.loc[selected_channels["name"] == channel, "x"].item()
        y = selected_channels.loc[selected_channels["name"] == channel, "y"].item()
        z = selected_channels.loc[selected_channels["name"] == channel, "z"].item()
        channels_types = channel_info.loc[channel_info["name"] == channel, "type"].item()
        mni_coordinates = mni_coordinates.append(pd.DataFrame({
            "channels": channel,
            "ch_types": channels_types,
            "x": x,
            "y": y,
            "z": z
        }, index=[0]))
    mni_coordinates = mni_coordinates.reset_index(drop=True)
    return mni_coordinates


def prepare_test_data(root, signal, baseline_method, baseline_window, test_window, metric, test,
                      subject, baseline_time=(None, 0), crop_time=None, condition=None, session="V1", task_name="Dur",
                      preprocess_folder="epoching", preprocess_steps="desbadcharej_notfil_autbadcharej_lapref",
                      channel_types=None, get_mni_coord=True, montage_space="T1", picks_rois=None,
                      multitaper_parameters=None, aseg="aparc.a2009s+aseg", scal=1e0):
    """
    This function loads the epochs and format them according to the test passed
    :param root: (string or pathlib object) path to the bids root
    :param signal: (string) name of the signal to investigate
    :param baseline_method: (string) name of the method to compute the baseline correction, see baseline_rescale from
    mne for more details
    :param baseline_window: (list of two floats) contains the onset and offset of the baseline
    :param test_window: (list of two floats) contains the onset and offset of the test data
    :param metric: (string) name of the method to use to compute the data aggregation if wilcoxon or t_test is passed
    as a test
    :param test: (string) name of the test to use. Dictates the way the data will be formated.
    :param subject: (string) name of the subject
    :param baseline_time: (list of two floats) onset and offset for baseline correction
    :param crop_time: (list of two floats) time points to crop the epochs
    :param condition: (string) name of the condition to use
    :param session: (string) name of the session
    :param task_name: (string) name of the task
    :param preprocess_folder: (string) name of the preprocessing folder
    :param preprocess_steps: (string) name of the preprocessing step to use
    :param channel_types: (dict or None) channel_type: True for the channel types to load
    :param get_mni_coord: (boolean) whether or not to return the MNI coordinates!
    :param montage_space: (string) space of the electrodes localization, either T1 or MNI
    :param picks_rois: (list) list of ROI according to aseg to get the electrodes from. The epochs will be returned only
    with electrodes within this roi list
    :param multitaper_parameters: (dict) contains info to filtering steps on the data to get specific frequency bands
    {
        "freq_range": [8, 13],
        "step": 1,
        "n_cycle_denom": 2,
        "time_bandwidth": 4.0
    }
    :param aseg: (string) which segmentation to use. Relevant if you want to get channels only from a given ROI
    :param scal: (float) scale of the data if rescale needed
    :return:
    pandas df: data formated correctly for the test to use
    """
    print("=" * 40)
    print("Preparing sub-{} data".format(subject))
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    # Load the data of the relevant subject with the right parameters (baseline correction...)
    epochs, mni_coord = load_epochs(root, signal, subject,
                                    session=session,
                                    task_name=task_name,
                                    preprocess_folder=preprocess_folder,
                                    preprocess_steps=preprocess_steps,
                                    channel_types=channel_types,
                                    condition=condition,
                                    baseline_method=baseline_method,
                                    baseline_time=baseline_time,
                                    crop_time=crop_time,
                                    select_vis_resp=False,
                                    vis_resp_folder=None,
                                    get_mni_coord=get_mni_coord,
                                    aseg=aseg,
                                    montage_space=montage_space,
                                    picks_roi=picks_rois,
                                    filtering_parameters=multitaper_parameters)
    # Format the data correctly depending on the test:
    if test == "wilcoxon_signed_rank" or test == "wilcoxon_rank_sum" or test == "t_test" \
            or test == "bayes_t_test" or "bayes_signed_rank":
        data_df = format_freq_test_data(epochs, subject, baseline_window, test_window, metric, scal)
    elif test == "cluster_based":
        data_df = format_cluster_based_data(epochs, subject, baseline_window, test_window)
    elif test == "sustained_zscore":
        data_df = format_sustained_zscore_data(epochs, test_window, subject)
    else:
        raise Exception("The passed test is not supported!")

    return data_df, epochs.info["sfreq"], mni_coord, epochs


def format_freq_test_data(epochs, subject, baseline_window, test_window, metric, scal):
    """
    This function format the data for frequentist test. It takes the epochs, crops them in time windows and generates
    data frame containing the data in two different time windows to compare (usually baseline vs something else).
    :param epochs: (mne epochs object) epochs object to format into data frames for stat tests
    :param subject: (string) name of the subject to add to the data frame to keep things logical.
    :param baseline_window: (list of floats) time window to compute the baseline (i.e. the condition against which to
    test)
    :param test_window: (list of floats) time window to compute the test data (i.e. the condition to test against
    baseline)
    :param metric: (string) metric to use to aggregate data in the specified time window (mean, ptp, auc...)
    :param scal: (float) scaling factor for the data (data are stored in volts by default, leading to very small
    values for things that are in microV...).
    :return:
    data_df: a pandas data frame storing the data in the right format for the test of interest!
    """
    print("=" * 40)
    print("Welcome to format_freq_test_data")
    # Cropping the data in the baseline:
    baseline_epochs = epochs.copy().crop(tmin=baseline_window[0],
                                         tmax=baseline_window[1])
    # And in the time window to compare against:
    onset_epochs = epochs.copy().crop(tmin=test_window[0],
                                      tmax=test_window[1])
    # Now computing the dependent variables for the onset and baseline:
    baseline_dependent_variable = \
        compute_dependent_variable(baseline_epochs,
                                   metric=metric)
    # Scale the data:
    baseline_dependent_variable["value"] = baseline_dependent_variable["value"].apply(lambda x: x * scal)
    onset_dependent_variable = \
        compute_dependent_variable(onset_epochs,
                                   metric=metric)
    # Scale the data:
    onset_dependent_variable["value"] = onset_dependent_variable["value"].apply(lambda x: x * scal)
    # Adding to each table a column marking the condition:
    baseline_dependent_variable["condition"] = "baseline"
    onset_dependent_variable["condition"] = "onset"
    # Combine the two:
    data_df = pd.concat([baseline_dependent_variable, onset_dependent_variable], ignore_index=True)
    # Add the subject ID:
    data_df["subject"] = subject

    return data_df


def format_cluster_based_data(epochs, subject, baseline_window, test_window):
    """
    This function formats data for a 1 sample cluster base test, i.e. it takes the data in the baseline and in the test
    window and subtract the two.
    :param epochs: (mne epochs object) contains the data to compute the difference
    :param subject: (string) name of the subject
    :param baseline_window: (list of two floats) contains the onset and offset of the baseline
    :param test_window: (list of two floats) contains the onset and offset of the test data
    :return:
    data_df: a pandas data frame storing the data in the right format for the test of interest!
    """
    print("=" * 40)
    print("Welcome to format_cluster_based_data")
    data_df = pd.DataFrame()
    # Compute baseline and onset:
    baseline_data = epochs.copy().crop(tmin=baseline_window[0],
                                       tmax=baseline_window[1])
    onset_data = epochs.copy().crop(tmin=test_window[0],
                                    tmax=test_window[1])
    # Looping through each channel to compute the difference between the two:
    for channel in baseline_data.ch_names:
        bs = np.squeeze(baseline_data.get_data(picks=channel))
        ons = np.squeeze(onset_data.get_data(picks=channel))
        # It can  be that because of rounding the two arrays are not the same size, in which case, equating size
        # by taking the smallest
        if bs.shape[1] != ons.shape[1]:
            min_len = min([bs.shape[1], ons.shape[1]])
            bs = bs[:, 0:min_len]
            ons = ons[:, 0:min_len]
        diff = bs - ons
        # Add to the data_df frame:
        data_df = data_df.append(pd.DataFrame(
            {"subject": subject,
             "channel": channel,
             "values": [diff]
             }
        ))
    return data_df


def format_sustained_zscore_data(epochs, test_window, subject):
    """
    This function computes zscore over a specified time window and stores it into a dataframe.
    :param epochs: (mne epoch object) contains the data to use to compute the z score
    :param test_window: (list of two floats) onset and offset of test time window
    :param subject: (string) name of the subject
    :return:
    data_df: a pandas data frame storing the data in the right format for the test of interest!
    """
    print("=" * 40)
    print("Welcome to format_sustained_zscore_data")
    data_df = pd.DataFrame()
    # Compute evoked response:
    evoked = epochs.average()
    # Convert to zscore:
    baseline_scaling(evoked, correction_method="zscore")
    # Crop according to the defined window:
    evoked.crop(tmin=test_window[0], tmax=test_window[1])
    # Convert to a dataframe:
    for channel in evoked.ch_names:
        # Add to the data_df frame:
        data_df = data_df.append(pd.DataFrame(
            {"subject": subject,
             "channel": channel,
             "values": [np.squeeze(evoked.get_data(picks=channel))]
             }, index=[0]))

    return data_df


def compute_latencies(epochs, results, baseline=(-0.3, 0), onset=(0.05, 0.35), sig_window_sec=0.050, alpha=0.05):
    """
    This function computes the latency of responses of significant channels using a sliding t-test with a minimum
    significance duration. The onset is defined as the time point in which the response is significant.
    :param epochs: (list of mne epochs object) contains the data of each single subject
    :param results: (pd dataframe) results of the onset responsiveness detection pipeline. This tells us which channel
    was found to be significant. We perform the latency computations only on these ones
    :param baseline: (tuple) time window of the baseline
    :param onset: (tuple) time window of the stimulus onset period
    :param sig_window_sec:
    :param alpha:
    :return:
    """
    results["latency"] = None
    # First, extract the results that were significant:
    sig_results = results.loc[results["reject"] == True]
    # Extract the channel names:
    channels = sig_results["channel"].to_list()
    # Convert the epochs list to a dictionary to iter over subjects:
    epochs = {epoch.ch_names[0].split("-")[0]: epoch for epoch in epochs}
    # Loop through each relevant channel:
    for channel in channels:
        # Exctract the data:
        data_baseline = np.squeeze(epochs[channel.split("-")[0]].copy().crop(baseline[0],
                                                                             baseline[1]).get_data(picks=channel))
        data_onset = np.squeeze(epochs[channel.split("-")[0]].copy().crop(onset[0],
                                                                          onset[1]).get_data(picks=channel))
        if data_baseline.shape[1] != data_onset.shape[1]:
            min_samp = np.min([data_baseline.shape[1], data_onset.shape[1]])
            data_baseline = data_baseline[:, 0:min_samp]
            data_onset = data_onset[:, 0:min_samp]
        # Compute the sliding t-test between the two:
        y = data_baseline - data_onset
        y_stat, y_pval = ttest_1samp(y, 0, axis=0, alternative="two-sided")
        y_bin = y_pval < alpha
        # Convert the time window from ms to samples:
        window_samp = int(sig_window_sec * (epochs[channel.split("-")[0]].info["sfreq"] / 1))
        h0 = True
        # Looping through each True in the binarize y:
        for ind in np.where(y_bin)[0]:
            if ind + window_samp < len(y_bin):
                # If all the samples from here to + window_samp are significant, then we found a significant chunk
                if all(y_bin[ind:ind + window_samp]):
                    h0 = False  # Set h0 to false
                    # Finding the offset of the significant window:
                    onset_samp = ind
                    # Convert to me:
                    onset_sec = onset_samp * (1 / epochs[channel.split("-")[0]].info["sfreq"]) + onset[0]
                    break
            else:
                break
        if h0:
            onset_sec = None
        results.loc[results["channel"] == channel, "latency"] = onset_sec
    return results



