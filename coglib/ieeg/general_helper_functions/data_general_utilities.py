""" This script contains various functions performing operations on the data
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Simon Henin
    Simon.Henin@nyulangone.org
"""
import os
from pathlib import Path

import mne.time_frequency
import numpy as np
import pandas as pd
import mne
import scipy
from mne.baseline import rescale
from mne_bids import BIDSPath

from general_helper_functions.pathHelperFunctions import find_files
from mne.stats.cluster_level import _find_clusters, _cluster_indices_to_mask, _cluster_mask_to_indices, \
    _pval_from_histogram, _reshape_clusters


def cluster_test(x_obs, null_dist, z_threshold=None, adjacency=None, tail=1, max_step=None, exclude=None,
                 t_power=1, step_down_p=0.05, do_zscore=True):
    """
    This function performs a cluster based permutation test on a single observation array with respect to a null
    distribution. This is useful in case where for example decoding was performed on a single subject and a null
    distribution was obtained by shuffling the labels and performing the analysis a 1000 times. You are left with one
    array of observed value and several arrays constituting your null distribution. In a classical cluster based
    permutation test, a statistical test will be performed and cluster-summed, and then compared to cluster-sum values
    obtained by shuffling the observation across groups. In this case, because there is only one observation, that
    doesn't work. Instead, the observed data and the null distribution get z scored. Then, cluster sum are computed
    both on the x and h0 to assess which clusters are significant.
    NOTE: this was created by selecting specific bits from:
    https://github.com/mne-tools/mne-python/blob/eb14b9c55c65573a27624533e9224dcf474f6ad5/mne/stats/cluster_level.py#L684
    :param x_obs: (1 or 2D array) contains the observed data for which to compute the cluster based permutation test
    :param null_dist: (x.ndim + 1 array) contains the null distribution associated with the observed data. The dimensions
    must be as follows: [n, p, (q)] where n are the number of observation (i.e. number of permutation that were used
    to generate the null distribution), p and (q) correspond to the dimensions of the observed data
    (time and frequency, or only time, or time x time...)
    :param z_threshold: (float) z score threshold for something to be considered eligible for a cluster
    :param adjacency: (scipy.sparse.spmatrix | None | False) see here for details:
    https://mne.tools/stable/generated/mne.stats.spatio_temporal_cluster_test.html
    :param tail: (int) 1 for upper tail, -1 lower tail, 0 two tailed
    :param max_step: (int) see here for details:
    https://mne.tools/stable/generated/mne.stats.spatio_temporal_cluster_test.html
    :param exclude: (bool array or None) array of same dim as x for excluding specific parts of the matrix from analysis
    :param t_power: (float) power by which to raise the z score by. When set to 0, will give a count of locations in
    each cluster, t_power=1 will weight each location by its statistical score.
    :param step_down_p: (float) To perform a step-down-in-jumps test, pass a p-value for clusters to exclude from each
    successive iteration.
    :param do_zscore: (boolean) if the data are zscores already, don't redo the z transform
    :return:
    x_zscored: (x.shape np.array) observed values z scored
    h0_zscore: (h0.shape np.array) null distribution values z scored
    clusters: (list) List type defined by out_type above.
    cluster_pv: (array) P-value for each cluster.
    p_values: (x.shape np.array) p value for each observed value
    H0: (array) Max cluster level stats observed under permutation.
    """
    print("=" * 40)
    print("Welcome to cluster_test")
    # Checking the dimensions of the two input matrices:
    if x_obs.shape != null_dist.shape[1:]:
        raise Exception("The dimension of the observed matrix and null distribution are inconsistent!")

    # Get the original shape:
    sample_shape = x_obs.shape
    # Get the number of tests:
    n_tests = np.prod(x_obs.shape)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')
    # Step 1: Calculate z score for original data
    # -------------------------------------------------------------
    if do_zscore:
        print("Z scoring the data:")
        x_zscored = zscore_mat(x_obs, null_dist, axis=0)
        h0_zscore = [zscore_mat(null_dist[i], np.append(x_obs[None], null_dist, axis=0)) for i in range(null_dist.shape[0])]
    else:
        x_zscored = x_obs
        h0_zscore = [null_dist[i] for i in range(null_dist.shape[0])]

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # Step 2: Cluster the observed data:
    # -------------------------------------------------------------
    print("Finding the cluster in the observed data:")
    out = _find_clusters(x_zscored, z_threshold, tail, adjacency,
                         max_step=max_step, include=include,
                         partitions=None, t_power=t_power,
                         show_info=True)
    clusters, cluster_stats = out

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        clusters = _cluster_indices_to_mask(clusters, 20)

    # Compute the clusters for the null distribution:
    if len(clusters) == 0:
        print('No clusters found, returning empty H0, clusters, and cluster_pv')
        return x_zscored, h0_zscore, np.array([]), np.array([]), np.array([]), np.array([])

    # Step 3: repeat permutations for step-down-in-jumps procedure
    # -------------------------------------------------------------
    n_removed = 1  # number of new clusters added
    total_removed = 0
    step_down_include = None  # start out including all points
    n_step_downs = 0
    print("Finding the cluster in the null distribution:")
    while n_removed > 0:
        # actually do the clustering for each partition
        if include is not None:
            if step_down_include is not None:
                this_include = np.logical_and(include, step_down_include)
            else:
                this_include = include
        else:
            this_include = step_down_include
        # Find the clusters in the null distribution:
        _, surr_clust_sum = zip(*[_find_clusters(mat, z_threshold, tail, adjacency,
                                                 max_step=max_step, include=this_include,
                                                 partitions=None, t_power=t_power,
                                                 show_info=True) for mat in h0_zscore])
        # Compute the max of each surrogate clusters:
        h0 = [np.max(arr) if len(arr) > 0 else 0 for arr in surr_clust_sum]
        # Get the original value:
        if tail == -1:  # up tail
            orig = cluster_stats.min()
        elif tail == 1:
            orig = cluster_stats.max()
        else:
            orig = abs(cluster_stats).max()
        # Add the value from the original distribution to the null distribution:
        h0.insert(0, orig)
        h0 = np.array(h0)
        # Extract the p value of the max cluster by locating the observed cluster sum on the surrogate cluster sums:
        cluster_pv = _pval_from_histogram(cluster_stats, h0, tail)

        # figure out how many new ones will be removed for step-down
        to_remove = np.where(cluster_pv < step_down_p)[0]
        n_removed = to_remove.size - total_removed
        total_removed = to_remove.size
        step_down_include = np.ones(n_tests, dtype=bool)
        for ti in to_remove:
            step_down_include[clusters[ti]] = False
        if adjacency is None and adjacency is not False:
            step_down_include.shape = sample_shape
        n_step_downs += 1

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(x_obs).T
    for cluster, pval in zip(clusters, cluster_pv):
        if isinstance(cluster, np.ndarray):
            p_values_[cluster.T] = pval
        elif isinstance(cluster, tuple):
            p_values_[cluster] = pval

    return x_zscored, h0_zscore, clusters, cluster_pv, p_values_.T, h0


def stack_evoked(evoked_list):
    """
    This function stacks mne evoked objects as if they all came from one subject. This is particularly helpful for
    intracranial recordings, were each subject has different coverage. We can pretend everything came from one subject
    by simply stacking everything. That leads to one super gathering all the electrodes you have in your sample.
    :param evoked_list: (list of mne evoked) list of all the evoked objects to stack. You should have as many evoked
    as you have participants.
    :return:
    (mne evoked object) contain all the channels of all participants combined
    """
    # Create list to hold the data of each patient:
    ch_names = []
    montage = mne.channels.make_dig_montage()
    channel_types = []
    sfreq = []
    data = []
    tmin = []
    nave = []
    # Looping through all the evoked:
    for evoked in evoked_list:
        # Fetching the info and data from this specific evoked object
        ch_names.extend(evoked.ch_names)
        montage = montage.__add__(evoked.get_montage())
        channel_types.extend(evoked.get_channel_types())
        sfreq.append(evoked.info["sfreq"])
        data.append(evoked.data)
        tmin.append(evoked.times[0])
        nave.append(evoked.nave)
    # Convert the data to a numpy array:
    data = np.concatenate(data)
    if len(set(sfreq)) > 1:
        raise Exception("The different evoked objects you are trying to concatenate have different sampling "
                        "rates!")
    if len(set(tmin)) > 1:
        raise Exception("The different evoked objects have different tmin. They cannot be appended!")
    if len(set(nave)) > 1:
        mne.utils.warn("The different evoked objects were generated by averaging different number of trials. We will"
                       "take the max number of trials as the default",
                       RuntimeWarning)
    # Creating the info:
    info = mne.create_info(ch_names, ch_types=channel_types, sfreq=sfreq[0])
    info.set_montage(montage)
    # Creating the evoked object:
    evoked = mne.EvokedArray(data, info, tmin=tmin[0], nave=max(nave))

    return evoked


def moving_average(data, window_size, axis=-1, overlapping=False):
    """
    This function performs moving average of multidimensional arrays. Shouthout to
    https://github.com/NGeorgescu/python-moving-average and
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/43200476#43200476 for the inspiration
    :param data: (numpy array) data on which to perform the moving average
    :param window_size: (int) number of samples in the moving average
    :param axis: (int) axis along which to perform the moving average
    :param overlapping: (boolean) whether or not to perform the moving average in an overlapping fashion or not. If
    true, fully overlapping, if false, none overelapping (i.e. moving from window size to window size)
    data = [1, 2, 3, 4, 5, 6]
    overlapping:
    mean(1, 2, 3), mean(2, 3, 4), mean(3, 4, 5)...
    non-overlapping:
    mean(1, 2, 3), mean(4, 5, 6)...
    :return:
    mvavg: (np array) data following moving average. Note that the dimension will have changed compared to the original
    matrix
    """
    if overlapping:
        # Bringing the axis over which to average to first position to have everything happening on first dim thereafter
        data_swap = data.swapaxes(0, axis)
        # Compute cumsum and divide by the window size:
        data_cum_sum = np.cumsum(data_swap, axis=0) / window_size
        # Adding a row of zeros to the first dimension:
        if data_cum_sum.ndim > 1:
            # Add zeros in the first dim:
            data_cum_sum = np.vstack([[0 * data_cum_sum[0]], data_cum_sum])
        else:  # if array is only 1 dim:
            data_cum_sum = np.array([0, *data_cum_sum])
        # Compute the moving average by subtracting the every second row of the data by every other second:
        mvavg = data_cum_sum[window_size:] - data_cum_sum[:-window_size]
        # Bringing back the axes to the original dimension:
        return np.swapaxes(mvavg, 0, axis)
    else:
        # Bringing the axis over which to average to first position to have everything happening on first dim thereafter
        data_swap = data.swapaxes(0, axis)
        # Handling higher dimensions:
        data_dim = data_swap[:int(len(data_swap) / window_size) * window_size]
        # Reshape the data, such that along the 1st dimension, we have the n samples of the independent bins:
        data_reshape = data_dim.reshape(int(len(data_swap) / window_size), window_size, *data_swap.shape[1:])
        # Compute the moving avereage along the 1dim (this is how it should be done based on the reshape above:
        mvavg = data_reshape.mean(axis=1)
        return mvavg.swapaxes(0, axis)


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    This function performs baseline correction on the data. The default is to compute the mean over the entire baseline
    and dividing each data points in the entire epochs by it. Another option is to substract baseline from each time
    point
    :param epochs: (mne epochs object) epochs on which to perform the baseline correction
    :param correction_method: (string) options to do the baseline correction. Options are:
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
          source: https://github.com/mne-tools/mne-python/blob/main/mne/baseline.py
    :param baseline: (tuple) which bit to take as the baseline
    :param picks: (None or list of int or list of strings) indices or names of the channels on which to perform the
    correction. If none, all channels are used
    :param n_jobs: (int) number of jobs to use to preprocessing the function. Can be ran in parallel
    :return: none, the data are modified in place
    """
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None


def set_channels_types(raw, bids_path):
    """
    This function reads the bids tsv file to extract the channel types and set the channels accordingly:
    :param raw: (mne raw object) raw data to set the channel type for
    :param bids_path: (mne BIDSPath) path object to the bids directory
    :return: raw (mne raw object) with set channels types
    """
    # Looking for the channel tsv:
    channel_tsv = find_files(
        bids_path.directory, naming_pattern="*channels", extension=".tsv")
    channels_info = pd.read_csv(channel_tsv[0], sep='\t')  # Loading the info
    # Getting the channel type of each channel available:
    ch_type_map_avail = {
        row["name"]: row["type"].lower()
        for ind, row in channels_info.iterrows()
    }
    # Setting the channel type accordingly:
    raw.set_channel_types(ch_type_map_avail)
    return raw


def set_annotations(raw, bids_path):
    """
    This function read the events from the events file of the correct data. This is to circumvent an mne bug, it gets
    confused when we have both ieeg and eeg data in the same task and can't decide which annotations to pick and create
    wrong ones. Therefore, creating the annotations ourselves. Will become useless once they have fixed the bug
    :param raw: (mne raw object)
    :param bids_path: (m,ne bids path object) path object to the bids directory
    :return:
    raw (mne raw object) containing all the data
    """
    # Looking for the channel tsv:
    events_tsv = find_files(bids_path.directory,
                            naming_pattern="*events", extension=".tsv")
    events_df = pd.read_csv(events_tsv[0], sep='\t')  # Loading the info
    # Create the annotations from the events tsv:
    # Creating annotation from onset, duration and description:
    my_annotations = mne.Annotations(onset=np.array(events_df.onset),
                                     duration=np.array(events_df.duration),
                                     description=np.array(events_df.trial_type))

    # Setting the annotation in the raw signal
    raw.set_annotations(my_annotations)
    return raw


def set_montage(raw, bids_path, montage_space="T1"):
    """
    This function sets the montage on the raw data according to the passed montage space. Natively, mne_bids will
    read electrodes localization in the coordinates that were last saved. But we want to control which one to load,
    which is why this function is used! Accepted montage space: T1 or MNI
    :param raw: (mne raw object) contains the data to which the montage will be added
    :param bids_path: (mne_bids path object) contains path information to find the correct files
    :param montage_space: (string) choose which space you want the montage to be in. Accepted: T1 or MNI
    :return:
    raw (mne object) with montage set properly
    """
    # Handle montage type
    if montage_space.upper() == "T1":
        coord_file = "*space-Other_electrodes"
        coord_frame = "mri"
    elif montage_space.upper() == "MNI":
        coord_file = "*space-fsaverage_electrodes"
        coord_frame = "mni_tal"
    else:
        raise Exception("You have passed a montage space that is not supported. It should be either T1 or MNI! Check "
                        "your config")

    # Loading the coordinate file:
    recon_file = find_files(bids_path.directory,
                            naming_pattern=coord_file, extension=".tsv")
    # Load the file:
    channels_coordinates = pd.read_csv(
        recon_file[0], sep='\t')  # Loading the coordinates
    # From this file, getting the channels:
    channels = channels_coordinates["name"].tolist()
    # Get the position:
    position = channels_coordinates[["x", "y", "z"]].to_numpy()
    # Create the montage:
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, position)), coord_frame=coord_frame)

    # And set the montage on the raw object:
    raw.set_montage(montage, on_missing='warn')

    return raw


def find_outliers(data, m=4., func="median"):
    """
    This function finds outliers that are m*sd away from the mean. It returns the indices of the rows considered
    outliers.
    :param data: (np array) contains the data from which the outliers must be rejected
    :param m: (float or int) sd multiplier to consider a data point an outlier.
    :param func: (string) function to use to compute the central value of the data. Either mean or median
    :return:
    """
    if func.lower() == "median":
        foo = np.median
    elif func.lower() == "mean":
        foo = np.mean
    else:
        raise Exception("You have passed a function to remove outliers that is not supported. Use either mean or "
                        "median")
    # Subtract the center value to all data points in the data and take the absolute value thereof
    d = np.abs(data - foo(data))
    # Compute the center value of the distribution
    mdev = foo(d)
    # Divide each data point by the mean
    s = d / mdev if mdev else np.zeros(len(d))
    # We can now compare each value in s to the m factor, as this will tell us whether the data are more than
    # m sd away from the mean
    return np.where(s > m)[0]


def list_subjects(root, prefix="sub-"):
    """
    This function lists all the "subjects" found in a given folder.
    :param root:
    :param prefix:
    :return:
    subject_list (list) list with the name of the different subjects
    """
    list_folders = os.listdir(root)
    subject_list = [folder.split("-")[1]
                    for folder in list_folders if prefix in folder]

    return subject_list


def compute_dependent_variable(epochs, metric="mean", conditions=None):
    """
    This function computes the dependent variable on the epochs according to the passed metric. The epochs are first
    converted to a dataframe and then, the data are aggregated according to the metric function
    :param epochs: (mne epochs object) epochs on which the dependent variable should be computed
    :param metric: (string) metric to use to compute the dependent variable. So far, mean, ptp, auc, max and mix
    supported
    :param conditions: (string) name of the column of the epochs meta data to append to the dependent var in the
    condition column. If you haven't created the meta data, that functionality won't work
    :return:
    dependent_var_data (pd.dataFrame) contains the data aggregated across the duration of the epochs according to the
    given metric.
    """
    # Matching metric to its function
    if metric.lower() == "mean":
        metric_fun = np.mean
    elif metric.lower() == "ptp":
        metric_fun = np.ptp
    elif metric.lower() == "auc":
        metric_fun = np.trapz
    elif metric.lower() == "max":
        metric_fun = np.max
    elif metric.lower() == "min":
        metric_fun = np.min
    else:
        raise TypeError("The passed metric is not supported! Make sure you set the metric in the config file to mean, "
                        "ptp, auc, max or min. Check spelling")

    # Scaling of 1.0 applied because the scale is signal and not channel dependent in our case:
    scalings = dict(eeg=1e0, seeg=1e0, ecog=1e0)

    # Converting the epoched data to a dataframe:
    df_epochs = epochs.to_data_frame(long_format=True, scalings=scalings)

    # Appyling the function set above on data grouped by trials (i.e. epochs):
    dependent_var_data = \
        df_epochs.groupby(['channel', 'epoch'], group_keys=False)[
            "value"].aggregate(metric_fun).reset_index()
    if conditions is not None:
        if isinstance(conditions, str):
            # Adding a column for the condition that is to be compared down the line:
            try:
                dependent_var_data["condition"] = \
                    [epochs.metadata.loc[i, conditions]
                     for i in dependent_var_data["epoch"]]
            except AttributeError:
                raise Exception("In order to use the condition variable, you need to have generated the metadata of "
                                "the epochs first!")
            except KeyError:
                raise Exception(
                    "You have passed a condition that does not exist in your epoch meta data!")
        elif isinstance(conditions, list):
            # Adding the different requested conditions to the dataframe:
            for condition in conditions:
                try:
                    dependent_var_data[condition] = [epochs.metadata.loc[i, condition]
                                                     for i in dependent_var_data["epoch"]]
                except AttributeError:
                    raise Exception(
                        "In order to use the condition variable, you need to have generated the metadata of "
                        "the epochs first!")
                except KeyError:
                    raise Exception(
                        "You have passed a condition that does not exist in your epoch meta data!")
        else:
            raise TypeError(
                "The condition variable must be either a string or a list!")

    # Returning the computed dependent variable:
    return dependent_var_data


def mean_confidence_interval(data, confidence=0.95, axis=0):
    """
    This function computes the mean and the 95% confidence interval from the data, according to the axis.
    :param data: (numpy array) data on which to compute the confidence interval and mean
    :param confidence: (float) interval of the confidence interval: 0.95 means 95% confidence interval
    :param axis: (int) axis from the data along which to compute the mean and confidence interval
    :return:
    mean (numpy array) mean of the data along specified dimensions
    mean - ci (numpy array) mean of the data minus the confidence interval
    mean + ci (numpy array) mean of the data plus the confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=axis), scipy.stats.sem(a, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def epochs_mvavg(epochs, window_ms):
    """
    This function computes a non-overlapping moving average on mne epochs object.
    :param epochs: (mne epochs object) epochs to smooth
    :param window_ms: (int) window size in milliseconds
    :return:
    epochs: the smoothed mne epochs object
    """
    n_samples = int(np.floor(window_ms * epochs.info["sfreq"] / 1000))
    epochs_data = moving_average(epochs.get_data(), n_samples, axis=-1, overlapping=False)
    times = moving_average(epochs.times, n_samples)
    info = epochs.info
    info['sfreq'] = 1 / (n_samples / epochs.info['sfreq'])
    epochs = mne.EpochsArray(epochs_data, info, tmin=times[0], events=epochs.events,
                             event_id=epochs.event_id, on_missing="warning", metadata=epochs.metadata)
    return epochs


def load_epochs(root, signal, subject, session="V1", task_name="Dur", preprocess_folder="epoching",
                preprocess_steps="desbadcharej_notfil_autbadcharej_lapref", channel_types=None, condition=None,
                baseline_method=None, baseline_time=(None, 0), crop_time=None,
                select_vis_resp=False, vis_resp_folder="high_gamma_wilcoxon_onset_activation_no_fdr",
                aseg="aparc.a2009s+aseg", montage_space="T1", get_mni_coord=False, picks_roi=None,
                filtering_parameters=None, mvavg_window_ms=None):
    """
    This function loads epochs data of a given participant according to the different passed parameters (session, folder
    preprocessing steps...). It further performs some data preparation such as baseline correction, selecting specific
    conditions. There is furthermore the option to load the roi for each electrode.
    :param root: (string or pathlib object) path to the bids root
    :param signal: (string) name of the signal to investigate
    :param baseline_method: (string) name of the method to compute the baseline correction, see baseline_rescale from
    mne for more details
    :param subject: (string) name of the subject
    :param baseline_time: (list of two floats) onset and offset for baseline correction
    :param crop_time: (list of two floats) time points to crop the epochs
    :param select_vis_resp: (boolean) whether or not to select only the visually responsive channels
    :param vis_resp_folder: (string) name of the folder containing the visually responsiveness results. The visual resp
    analysis can be ran in several different ways, you must choose which option you want!
    :param condition: (string) name of the condition to use
    :param session: (string) name of the session
    :param task_name: (string) name of the task
    :param preprocess_folder: (string) name of the preprocessing folder
    :param preprocess_steps: (string) name of the preprocessing step to use
    :param channel_types: (dict or None) channel_type: True for the channel types to load
    :param aseg: (string) segmentation file to use from the freesufer folder
    :param montage_space: (string) montage space: "T1" or "MNI"
    :param get_mni_coord: (boolean) whether or not to return the mni coordinates of the electrodes for the given subject
    :param picks_roi: (list of strings) contains a list of ROI according to the segmentation passed in aseg. If
    something is passed (as opposed to None), only electrodes found within the said ROI will be selected!
    :param filtering_parameters: (dict) contains the multitaper parameters. Must have the format:
    {
        "freq_range": [8, 13],
        "step": 1,
        "n_cycle_denom": 2,
        "time_bandwidth": 4.0
    }
    :param mvavg_window_ms: (int or None) moving average window to smooth the data in a non-overlapping fashion
    :return:
    """
    print("=" * 40)
    print("loading sub-{} epochs".format(subject))
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    file_dir = str(Path(root, "derivatives", "preprocessing", "sub-" + subject,
                        "ses-" + session, "ieeg", preprocess_folder, signal, preprocess_steps))
    data_file = find_files(file_dir,
                           naming_pattern="*-epo", extension=".fif")
    try:
        epochs = mne.read_epochs(data_file[-1],
                                 verbose='error', preload=True)
    except IndexError:
        raise Exception("No data found for sub-{}".format(subject))

    # There is a bug from MNE such that the montage coordinate space is ignore and considered to be head always.
    # Setting the montage to the correct coordinates frame:
    from mne.io.constants import FIFF
    for d in epochs.info["dig"]:
        d['coord_frame'] = FIFF.FIFFV_COORD_MRI

    # Getting only the visual responsive channels if required:
    if select_vis_resp:
        # Generating the path to the visual responsiveness folder:
        vis_resp_files = find_files(Path(root, "derivatives", "visual_responsiveness",
                                         "sub-super", "ses-" + session, "ieeg", "results",
                                         vis_resp_folder, preprocess_steps), naming_pattern="*vis_resp_all_results",
                                    extension=".csv")
        assert len(vis_resp_files) == 1, "ERROR: there was not 1 folder for visual responsiveness results!"
        # Get the visually responsive channels:
        picks = get_vis_resp_channels(subject, vis_resp_files[0])
        try:
            epochs.pick(picks)
        except ValueError:
            print("WARNING: For sub-{} there were not onset responsive electrodes".format(subject))
            return None, None
    # Get the picks:
    picks = mne.pick_types(epochs.info, **channel_types)
    epochs.pick(picks)
    # Pick the data of the passed condition:
    if condition is not None:
        epochs = epochs[condition]
    # Performing baseline correction if required:
    if baseline_method is not None:
        baseline_scaling(epochs, correction_method=baseline_method, baseline=baseline_time)

    # Do the multitaper, i.e. filter in a specified band
    if filtering_parameters is not None:
        freqs = np.arange(filtering_parameters["freq_range"][0], filtering_parameters["freq_range"][1],
                          filtering_parameters["step"])
        n_cycles = freqs / filtering_parameters["n_cycle_denom"]
        if filtering_parameters["method"] == "multitaper":
            tfr = mne.time_frequency.tfr_multitaper(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=False,
                picks=epochs.ch_names,
                time_bandwidth=filtering_parameters["time_bandwidth"],
                verbose=True)
        elif filtering_parameters["method"] == "wavelet":
            tfr = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=False,
                picks=epochs.ch_names,
                output="power",
                verbose=True)

        # Do baseline correction:
        if filtering_parameters["baseline_mode"] is not None:
            tfr.apply_baseline(filtering_parameters["baseline_win"], mode=filtering_parameters["baseline_mode"])
        # Extract the data in the frequency band:
        data = np.mean(tfr.data, axis=-2)
        # Adjust the sfreq:
        info = epochs.info
        info["sfreq"] = data.shape[-1] / (epochs.times[-1] - epochs.times[0])
        # Shove it back in the mne epochs object:
        epochs = mne.EpochsArray(data, epochs.info, tmin=epochs.times[0], events=epochs.events,
                                 event_id=epochs.event_id, on_missing="warning", metadata=epochs.metadata)
    # Crop the epochs if requried
    if crop_time is not None:
        epochs.crop(tmin=crop_time[0], tmax=crop_time[1])
    if mvavg_window_ms is not None:
        epochs = epochs_mvavg(epochs, mvavg_window_ms)

    bids_path = BIDSPath(root=root, subject=subject,
                         session=session,
                         datatype="ieeg",
                         task=task_name)
    # Getting the mni coordinates if required:
    if get_mni_coord:
        mni_coord = get_mni_coordinates(bids_path, epochs.ch_names)
        # Appending the participant ID to the channel name:
        mni_coord["channels"] = ["-".join([subject, ch]) for ch in mni_coord["channels"].to_list()]
    else:
        mni_coord = None

    # Now, rename the channels by appending the subject ID to ensure that the channels name are unique to the subject:
    epochs.rename_channels({ch: "-".join([subject, ch]) for ch in epochs.ch_names})
    # Getting the ROI if required:
    if picks_roi is not None:
        # Now get the labels:
        labels, _ = mne.get_montage_volume_labels(
            epochs.get_montage(), "sub-" + subject, subjects_dir=Path(root, "derivatives", "fs"), aseg=aseg)
        roi_picks = find_channels_in_roi(picks_roi, labels)
        if len(roi_picks) < 1:
            print("WARNING: For sub-{} there were not electroodes found in the following regions {}".format(subject,
                                                                                                            picks_roi))
            epochs = None
        else:
            # Picking only these electrodes
            print("The following electrodes were found in the ROIs {}".format(picks_roi))
            print(roi_picks)
            epochs.pick(roi_picks)

    return epochs, mni_coord


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


def get_vis_resp_channels(subject, vis_resp_file):
    """
    This function loads and parses the visual responsiveness results
    :param subject: (string) name of the subject for whom to load the visual responsive channels
    :param vis_resp_file: (pathlib path or string) path to the file containing the results of the visual responsiveness
    :return:
    picks: list of visually responsive channels
    """
    print("=" * 20)
    print("Welcome to get_vis_resp_channels")
    # Load the results:
    vis_resp_results = pd.read_csv(vis_resp_file)
    # Get the results from only this subject:
    sub_results = vis_resp_results.loc[vis_resp_results["subject"] == subject]
    # Get the name of the onset responsive channels:
    picks = sub_results.loc[sub_results["reject"] == True, "channel"].to_list()
    # Removing the subject ID from the picks:
    picks = [pick.split("-")[1] for pick in picks]
    print("The following channels were found to be onset responsive for sub-" + subject)
    print(picks)

    return picks


def find_channels_in_roi(roi, channels_labels):
    """
    This function checks which channels are found in a given ROI. The format of the ROI must be a list of labels that
    constitute that roi. The labels are of the format outputted by mne.get_montage_volume_labels, a dict with a list
    of labels for a given channel according to a specific parcellation.
    :param roi: (dict) key of  the dictionary corresponds to the ROI in question, and within it there is a string
    with all the labels from the free surfer parcellation of interest
    For example:
    "occipital": ["ctx-rh-lateraloccipital", "ctx-lh-lateraloccipital", "ctx-rh-inferiorparietal",
    "ctx-lh-inferiorparietal", "ctx-rh-pericalcarine", "ctx-lh-pericalcarine", "ctx-rh-cuneus", "ctx-lh-cuneus"]
    :param channels_labels: (dict) output of the mne function: mne.get_montage_volume_labels
    :return: (list of strings) list of the channels found in the ROI of interest
    """
    roi_channels = []
    # Looping through each channel found in the channels_labels dict:
    for ch in channels_labels.keys():
        # Looping through each label of this specific channel:
        for label in channels_labels[ch]:
            if label in roi:
                roi_channels.append(ch)
                break

    # Return the channels
    return roi_channels


def zscore_mat(x, h0, axis=0):
    """
    This function computes a zscore between a value x and a
    :param x: (float) a single number for which to compute the zscore with respect ot the y distribution to the
    :param h0: (1d array) distribution of data with which to compute the std and mean:
    :param axis: (int) which axis along which to compute the zscore for the null distribution
    :return: zscore
    """
    assert isinstance(x, np.ndarray) and isinstance(h0, np.ndarray), "x and y must be numpy arrays!"
    assert len(h0.shape) == len(x.shape) + 1, "y must have 1 dimension more than x to compute mean and std over!"
    try:
        zscore = (x - np.mean(h0, axis=axis)) / np.std(h0, axis=axis)
    except ZeroDivisionError:
        Exception("y standard deviation is equal to 0, can't compute zscore!")

    return zscore
