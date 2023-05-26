"""
This script contains all the rsa helper functions
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import time
from scipy.stats import spearmanr
import itertools
import random
import pandas as pd
from statsmodels.stats import multitest
from scipy import stats
import numpy as np
import mne.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.spatial.distance import cdist
from mne.stats.cluster_level import _pval_from_histogram
from collections import Counter

from general_helper_functions.data_general_utilities import moving_average

np.seterr(divide='ignore')


def rsa_shuffle_label_test(observed_matrix, shuffle_matrices, predicted_matrices, correlation_method,
                           alpha=0.05):
    """
    This function tests the significance of correlation difference between 2 theories by computing the different between
    the two and comparing the obtained value to what would be obtained when shuffling the labels to get decoding results
    :param observed_matrix: (np array) observed matrix to correlate to theories matrices
    :param shuffle_matrices: (np array) matrices obtained through labels shuffling
    :param predicted_matrices: (dict of numpy array) matrices predicted by theories, on matrix per theory
    :param correlation_method: (string) correlation method to use,
    :param alpha: (Float) significance threshold to perform the comparison between the theories comparison
    :return:
    obs_val: (float) observed correlation different between the two theories
    perm_p_val: (float) quantile of the oberved difference among the null distribution
    null_dist_list: (np array) difference obtained using the shuffled matrices
    obs_corr: (np array or float) results of the correlation
    """
    print("-" * 40)
    print("Welcome to compute_correlation_theories")
    print("Computing {0} correlation between data and {1} predicted matrices".
          format(correlation_method.lower(),
                 list(predicted_matrices.keys())))
    obs_corr, obs_corr_pos = \
        compute_correlation_theories([observed_matrix],
                                     predicted_matrices,
                                     method=correlation_method)
    # But also computing the random distribution by computing correlations of the permutated
    # set to the observed matrices:
    rand_corr, rand_corr_pos = \
        compute_correlation_theories(shuffle_matrices,
                                     predicted_matrices,
                                     method=correlation_method)
    # Compute theories correlations significance:
    corr_sig = {}
    for theory in predicted_matrices.keys():
        corr_sig[theory + "_p-val"] = _pval_from_histogram(np.array([obs_corr[theory].item()]),
                                                           rand_corr[theory].to_numpy(), 1)[0]
    # If none of the theories is significantly correlated with the data, then there is no point computing correlation
    # with the theories:
    if np.any(np.array([corr_sig[theory] for theory in corr_sig.keys()]) <= alpha):
        # If two theories are to be contrasted:
        if len(predicted_matrices) == 2:
            # Compute the difference between both theories correlation values from observed data:
            obs_val = \
                obs_corr_pos[list(predicted_matrices.keys())[0]].item() - \
                obs_corr_pos[list(predicted_matrices.keys())[1]].item()
            # Same for the null distribution
            null_dist_list = [
                rand_corr_pos[list(predicted_matrices.keys())[0]][i] -
                rand_corr_pos[list(predicted_matrices.keys())[1]][i]
                for i in
                range(len(rand_corr_pos[list(predicted_matrices.keys())[0]]))
            ]
        elif len(predicted_matrices) == 1:
            # If only one theory to consider, just take the correlation value
            obs_val = obs_corr[list(predicted_matrices.keys())[0]].item()
            # And the null distribution as is
            null_dist_list = rand_corr[list(predicted_matrices.keys())[0]].to_list()
        else:
            raise Exception("There were more than two predicted matrices, which is not supported by this function! "
                            "\nMake sure to have only two or one specific predicted matrix!")
        # Adding the observed correlation difference to ground it:
        null_dist_list.append(obs_val)
        # Compute the p value comparing the null distribution with the observed value:
        p_val = _pval_from_histogram(np.array([obs_val]), np.array(null_dist_list), 0)
    else:
        p_val = [1]
        null_dist_list = None
        obs_val = None

    return obs_corr, obs_val, p_val[0], null_dist_list, corr_sig


def compute_correlation_theories(observed_matrix, theories_matrices, method="pearson", upper_only=True):
    """
    Compute the correlation between the predicted and obtained matrices
    :param observed_matrix: (list of np arrays) contains the decoding matrices
    :param theories_matrices: (dict of arrays) contains the theories predicted matrices
    :param method: (string) method to use to compute the correlation. Three options supported: pearson, spearman and
    partial_correlation. In the partial correlation, one of the theory matrix will be used as a correlate
    :param upper_only: (boolean) test the upper half of the matrices only
    :return: (pd data frame) contains the correlation between the matrices
    """
    print("-" * 40)
    print("Welcome to compute_correlation_theories")
    print("Computing {0} correlation between data and {1} predicted matrices".
          format(method.lower(),
                 list(theories_matrices.keys())))
    supported_method = ["pearson", "spearman", "partial", "semi-partial", "kendall"]
    # Flatten the decoding scores and theory matrices:
    if upper_only:
        observed_matrix_flat = [observed_matrix[i][np.triu_indices(observed_matrix[i].shape[-1])]
                                for i in range(0, len(observed_matrix))]
        theory_matrices_flat = {theory: theories_matrices[theory][np.triu_indices(theories_matrices[theory].shape[-1])]
                                for theory in theories_matrices.keys()}
    else:
        observed_matrix_flat = [observed_matrix[i].flatten()
                                for i in range(0, len(observed_matrix))]
        theory_matrices_flat = {theory: theories_matrices[theory].flatten(
        ) for theory in theories_matrices.keys()}
    if method.lower() == "pearson" or method.lower() == "spearman" or method.lower() == "kendall":
        # Computing the correlation coefficient between the matrices of each cross validation folds:
        correlation_results = pd.DataFrame({
            theory: [pg.corr(observed_matrix_flat[i], theory_matrices_flat[theory],
                             method=method.lower())["r"].item()
                     for i in range(0, len(observed_matrix_flat))]
            for theory in theories_matrices.keys()
        })
    elif method.lower() == "partial":
        # For partial correlation, we need to convert the data to data frames:
        theories = list(theories_matrices.keys())
        data_dfs = [pd.DataFrame({
            "scores": observed_matrix_flat[i],
            theories[0]: theory_matrices_flat[theories[0]],
            theories[1]: theory_matrices_flat[theories[1]]})
            for i in range(0, len(observed_matrix))]
        # We now perform the partial correlation always holding one theory constant while checking the other:
        correlation_results = {}
        for ind, theory in enumerate(theories):
            correlation_results[theory] = [pg.partial_corr(data=data_dfs[i], x='scores', y=theory,
                                                           covar=theories[ind - 1])["r"].item()
                                           for i in range(len(data_dfs))]
        # Convert the dict to a dataframe to keep things consistent:
        correlation_results = pd.DataFrame(correlation_results)
    elif method.lower() == "semi-partial":
        # For partial correlation, we need to convert the data to data frames:
        theories = list(theories_matrices.keys())
        data_dfs = [pd.DataFrame({
            "scores": observed_matrix_flat[i],
            theories[0]: theory_matrices_flat[theories[0]],
            theories[1]: theory_matrices_flat[theories[1]]})
            for i in range(0, len(observed_matrix))]
        # We now perform the partial correlation always holding one theory constant while checking the other:
        correlation_results = {}
        for ind, theory in enumerate(theories):
            correlation_results[theory] = [pg.partial_corr(data=data_dfs[i], x='scores', y=theory,
                                                           y_covar=theories[ind - 1])["r"].item()
                                           for i in range(len(data_dfs))]
        # Convert the dict to a dataframe to keep things consistent:
        correlation_results = pd.DataFrame(correlation_results)
    else:
        raise Exception("You have passed {0} as correlation method, but only {1} supported".format(method.lower(),
                                                                                                   supported_method))

    # Correct the correlation results to be positively defined between 0 and 1:
    correlation_results_corrected = correlation_results.apply(lambda x: (x + 1) / 2)

    return correlation_results, correlation_results_corrected


def subsample_matrices(matrix, start_time, end_time, intervals_of_interest):
    """
    This function extracts bits of  a bigger matrix and puts it back together. It is basically subselecting only the
    bits of your matrix you care about. This enables for ex to do rsa only on bits of the temporal generalization matrix
    as opposed to all of it. The coordinate of the times of interest is a bit complicated. It is a dictionary
    containing two keys: x and y. x constitutes the "columns", while "y" constitute the rows. Because we are subsampling
    several squares of a bigger matrix (though subselecting only 1 square would be a just a specific case), the idea
    is that we can loop through the columns and then within that loop through the rows to make sure we don't mess up the
    order. Imagine you have a matrix like below and want to subsample the squares within it

                                     X
    _______________________________________________________________________
    |          ____                      ____                              |
    |         |    |                    |    |                             |
    |         |  1 |                    |  3 |                             |
    |         |____|                    |____|                             |
    |                                                                      |
    |                                                                      |
    |          ____                      ____                              |        Y
    |         |    |                    |    |                             |
    |         |  2 |                    |  4 |                             |
    |         |____|                    |____|                             |
    |                                                                      |
    |                                                                      |
    |______________________________________________________________________|

    The idea is that we will have the outer loop be looping through the x intervals and the inner loop looping through
    the Y. We can then stack vertically in the inner loop and horizontally in the outer loop, to be sure the order
    doesn't get all messed up. I.e. we first sample 1, then 2 and stack them vertically, same for 3 and 4 and then
    we stack the two matrices horizontally and that's it
    This means that the coordinates should be a list of x and y coordinates, but without repetition. I.e. you don't need
    to pass the same x coordinates twice for the samples in the same row.
    NOTE: THIS FUNCTION WILL ONLY WORK PROPERLY IF YOU SAMPLE UNIFORMLY IN THE X AND Y AXIS, YOU MUST HAVE A SQUARE
    MATRIX IN THE END!
    :param matrix: (2d np array) matrix to subsample
    :param intervals_of_interest: (dict of list) coordinate in times of each bits of the matrix you want to extract.
    Keys: x and y, see above for in depth explanation
    :param start_time: (float) time of the start time point of the passed matrix
    :param end_time: (float) time of the end time point of the passed matrix
    :return:
    subsampled_matrix (np.array): numpy array of the subsampled matrix according to what was expected
    new_time_vect (np.array): the truncated time vector of only the remaining time points.
    connection_indices (np.array): the coordinates within the subsample matrix in which there is the discontunity
    (to plot later on)
    sub_matrices_dict (dictionary): contains the subsampled squares but not stacked together. The x and y coordinates
    in temporal generalization matrix corresponds to test and train times respectively. Therefore, creating keys
    based on it.
    """
    # Generate the fitting time vector:
    time_vect = np.around(np.linspace(start_time, end_time, num=matrix.shape[0]), decimals=3)
    # Check that the passed times of interest are compatible with this function. Because we are dealing with floats,
    # need to convert to strings with 3 decimals. We assume that anything beyond 4 decimals isn't relevant given the
    # scale is in seconds, anything beyond that would be nanosecs:
    x_lengths = ["{:.3f}".format(x_coord[1] - x_coord[0]) for x_coord in intervals_of_interest["x"]]
    y_lengths = ["{:.3f}".format(y_coord[1] - y_coord[0]) for y_coord in intervals_of_interest["y"]]
    if len(set(x_lengths)) > 1 or len(set(y_lengths)) > 1:
        raise Exception("You have passed times of interest with inconsistent x and y length. This doesn't work because "
                        "\nthen, the matrices that you want to concatenate won't be of the same sizes")
    # Now computing the x and y length in samples
    x_lengths_samples = [np.where(time_vect <= x_coord[1])[0][-1] + 1 - np.where(time_vect >= x_coord[0])[0][0]
                         for x_coord in intervals_of_interest["x"]]
    y_lengths_samples = [np.where(time_vect <= y_coord[1])[0][-1] + 1 - np.where(time_vect >= y_coord[0])[0][0]
                         for y_coord in intervals_of_interest["y"]]
    x_length = min(x_lengths_samples)
    y_length = min(y_lengths_samples)
    matrix_columns = []
    new_time_vect = []
    connection_indices = []
    sub_matrices_dict = {}
    for col_ind, interval_x in enumerate(intervals_of_interest["x"]):
        matrix_sample = []
        for row_ind, interval_y in enumerate(intervals_of_interest["y"]):
            # Finding the time points corresponding to the start and end of the predicition
            x1 = np.where(time_vect >= interval_x[0])[0][0]
            x2 = x1 + x_length
            y1 = np.where(time_vect >= interval_y[0])[0][0]
            y2 = y1 + y_length
            matrix_sample.append(matrix[y1:y2, x1:x2])
            new_time_vect.append(time_vect[y1:y2])
            if row_ind == 0:
                connection_indices.append(len(matrix[y1:y2, x1:x2]) - 1)
            else:
                connection_indices.append(
                    connection_indices[-1] + len(matrix[y1:y2, x1:x2]))
            # Also add to a dictionary, to avoid having to break it down again after wards if needed:
            key = "Train_{}:{}-Test_{}:{}".format(interval_y[0], interval_y[1], interval_x[0], interval_x[1])
            sub_matrices_dict[key] = matrix[y1:y2, x1:x2]
        # Stacking the matrices sample horizontally:
        matrix_columns.append(np.concatenate(matrix_sample, axis=0))
    # Convert new time to a numpy array:
    new_time_vect = np.concatenate(new_time_vect, axis=0)
    # Removing repetitions:
    new_time_vect = np.unique(new_time_vect)
    # Same for the connection indices:
    connection_indices = np.unique(connection_indices)
    # Removing the last index, because we don't need it:
    connection_indices = connection_indices[:-1]
    # Finally, stacking the columns horizontally:
    return np.concatenate(matrix_columns, axis=1), new_time_vect, connection_indices, sub_matrices_dict


def label_shuffle_test_2d(observed_values, permutation_values, p_value_thresh=0.05, fdr_correction="fdr_bh"):
    """
    This function compares the distribution of observed values againt a null distribution generated by shuffling labels.
    The oberved values must be of the same dimentions except that the permutation values must have an extra dimension,
    representing the repetitions.
    This function compares the decoding scores observed against the results obtained when shuffling the labels. For each
    decoding score obtained (either time resolves or temporal generalization), its quantile along all the values
    obtained through permutation is computed. If the quantile is inferior to the threshold, it is considered significant
    :param observed_values: (np array) contains the observed decoding scores
    :param permutation_values: (np array) contains the decoding scores obtained by shuffling the labels.
    :param p_value_thresh: (float) p-value threshold for significance.
    :param fdr_correction: which method to use for FDR
    :return: diag_significance_mask: (np array of boolean) significance mask for the diagonal
    matrix_sig_mask (np array of floats and nan) contains the scores values but only the ones which are significant.
    """
    # Preallocate for storing the significance mask:
    p_values = np.zeros(observed_values.shape)
    # Generate the null distribution by concatenating the observed value to the permutation one:
    null_distribution = np.concatenate([permutation_values, np.expand_dims(observed_values, axis=0)], axis=0)
    # Now looping through each row and columns of the decoding matrix to compare obtained scores to the permutation
    # scores:
    for row_i in range(p_values.shape[0]):
        for col_i in range(p_values.shape[1]):
            # Find the position in the distribution of the obs value:
            null = np.append(np.squeeze(null_distribution[:, row_i, col_i]),
                             observed_values[row_i, col_i])
            p_values[row_i, col_i] = _pval_from_histogram(np.array([observed_values[row_i, col_i]]), null, 1)
    if fdr_correction is not None:
        _, p_val_flat, _, _ = multitest.multipletests(p_values.flatten(), method=fdr_correction)
        p_values = p_val_flat.reshape(p_values.shape)
    # Binarize the significance matrix based on p_value:
    sig_mask = p_values < p_value_thresh
    # The significance mask has nan where we have non-significant values, and the actual value where they are
    # significant
    matrix_sig_mask = observed_values.copy()
    matrix_sig_mask[~sig_mask] = np.nan
    # Creating the significance flag: true if there are significant bits in the matrix:
    significance_flag = np.any(sig_mask)

    return p_values, matrix_sig_mask, significance_flag


def compute_super_subject_rsa(epochs_list, condition, groups_condition=None, equalize_trials=True, binning_ms=None,
                              method="within_vs_between", shuffle_labels=False, zscore=False, min_per_label=2,
                              n_features=None, n_folds=5, verbose=False, n_channel_subsample=None,
                              feat_sel_diag=True, store_intermediate=False, metric="correlation"):
    """
    This function packages the different RSA stages to enable using parallelization.
    :param epochs_list: (list of mne epochs objects) contains single subjects data to be used to run the RSA
    :param condition: (list of strings) list of the different conditions to run the RSA on
    :param groups_condition: (string or None) group condition the trials of the first condition in case of nested
    designs. For example, when the condition is "identity", each "identity" belongs to a specific category: face,
    objects... So if you pass "identity" as condition and "category" as groups_condition, you will have the labels be
    something like:
    face_01
    face_02
    face_03...
    And the group labels be something like this:
    face
    face
    face
    Having this information can be useful for subsequent operations
    :param equalize_trials: (boolean) whether or not to equate the number of trials between the different classes
    being investigated. So say you do faces vs objects, if you have 5 faces and 6 objects, this option will downsample
    objects to be also 5
    :param binning_ms: (None or int) duration in ms of the bins of non-overlapping moving average.
    :param method: (string) which method to use to compute the RSA. There is only one: within_vs_between, but it is
    in case we want to add more options
    :param shuffle_labels: (boolean) whether or not to shuffle the labels. This function can be called a 1000 times
    with random label shuffle to generate a null distribution
    :param zscore: (boolean) whether or not to zscore the data while doing the RSA. This is done within the
    rsa computations
    :param min_per_label: (int) minimum number of samples you want to have per class. Any excess will be downsampled
    :param n_features: (int) how many features you want to use to compute the RSA (i.e. channels). If this is not
    set to None, the features will be selected with select K best based on zscores with cross validation
    :param n_folds: (int) number of folds for the feature selection
    :param verbose: (boolean) Whether or not to print info to the command line
    :param n_channel_subsample: (None or int) in order to investigate the robustness of our results, the exact
    same pipeline is repeated but by randomly subsampling a subset of electrodes. This method enables ensuring that
    the results are not driven by a single patient or a limited group of channels.
    :return:
    cross_temporal_mat: (numpy array) cross temporal rsa matrix that was computed
    rdm_diag: (numpy array) rdm at each time points along the diagonal
    data: (numpy array) data that were used to compute the rsa
    labels: (numpy array) labels of the data, i.e. trial conditions
    channels: (list) channels that were used here
    """
    data, labels, groups, times, sfreq, channels = \
        equate_super_subject_trials(epochs_list, condition, groups_condition=groups_condition,
                                    min_per_label=min_per_label, verbose=verbose)
    if equalize_trials:
        data, labels, groups = equalize_label_counts(data, labels, groups=groups)

    # Compute a moving average:
    if binning_ms is not None:
        n_samples = int(np.floor(binning_ms * sfreq / 1000))
        data = moving_average(data, n_samples, axis=-1, overlapping=False)

    if n_channel_subsample is not None:
        # Randomly select n electrodes:
        data = data[:, np.random.choice(data.shape[1], size=n_channel_subsample, replace=False), :]

    # Compute the RSA:
    if method == "within_vs_between":
        # This method computes the cross temporal rsm for each identity separately. 3
        # The returned matrices are informative of how the representation of a given identity is
        # maintained along time
        cross_temporal_mat, rdm_diag, first_pres_labels, second_pres_labels, sel_features = \
            within_vs_between_cross_temp_rsa(data,
                                             labels,
                                             metric=metric,
                                             zscore=zscore,
                                             onset_offset=[times[0],
                                                           times[-1]],
                                             n_features=n_features,
                                             n_folds=n_folds,
                                             shuffle_labels=shuffle_labels,
                                             verbose=verbose,
                                             feat_sel_diag=feat_sel_diag,
                                             store_intermediate=store_intermediate
                                             )
        sel_features = [channels[inds] for inds in sel_features]
        split_half_rdm = None
    elif method == "all_to_all_within_vs_between":
        cross_temporal_mat, rdm_diag, split_half_rdm = \
            all_to_all_within_class_dist(data, labels, metric=metric, n_bootsstrap=20,
                                         shuffle_labels=shuffle_labels,
                                         fisher_transform=True, verbose=verbose,
                                         n_features=n_features,
                                         n_folds=n_folds,
                                         feat_sel_diag=feat_sel_diag
                                         )
        first_pres_labels = labels
        second_pres_labels = None
        sel_features = None

    else:
        raise Exception(
            "No other method than within_vs_betweeen is supported ATM for the super subject!")

    return cross_temporal_mat, rdm_diag, first_pres_labels, second_pres_labels, channels, sel_features, split_half_rdm


def equalize_label_counts(data, labels, groups=None):
    """
    This function equalizes the counts of each labels, i.e. the number of trials per conditions. This function
    assumes that the data are in the format trials x channels x time and labels must have the same dimension as
    the first dimension as the data array
    :param data: (numpy array) trials x channels x time
    :param labels: (numpy array) label, i.e. condition of each trial
    :param groups: (numpy array) same size as labels and tracks the groups a given label belongs to.
    :return:
    equalized_data: (np array) trials x channels x time with the same number of trials per condition
    equalized_labels: (np array) trials with the same number of trials per condition
    equalized_groups: (np array or none)
    """
    # Equalizing the labels counts if needed:
    equalized_data = []
    equalized_labels = []
    if groups is not None:
        if groups.shape != labels.shape:
            raise Exception("ERROR: The groups array shape does not match the labels array shape! "
                            "\ngroups: {} \nvs \nlabels: {}".format(groups.shape, labels.shape))
        equalized_groups = []
    else:
        equalized_groups = None
    # Get the minimal counts in the labels:
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_label_counts = min(counts)
    # Now, looping through every unique label to randomly pick the min:
    for label in unique_labels:
        # Get the index of this  label:
        label_ind = np.where(labels == label)[0]
        # Randomly picking min:
        picked_ind = label_ind[np.random.choice(label_ind.shape[0], min_label_counts, replace=False)]
        # Fetching these data:
        equalized_data.append(data[picked_ind, :, :])
        equalized_labels.extend(labels[picked_ind])
        if groups is not None:
            equalized_groups.extend(groups[picked_ind])

    # Concatenating things back together:
    equalized_labels = np.array(equalized_labels)
    equalized_data = np.concatenate(equalized_data, axis=0)
    if groups is not None:
        equalized_groups = np.array(equalized_groups)

    return equalized_data, equalized_labels, equalized_groups


def equate_super_subject_trials(sub_epochs, condition, min_per_label=2, groups_condition=None, verbose=False):
    """
    This function combines the data of several subjects by equating the trial counts according to a specific condition
    such that all subjects have the exact same trial counts for the given condition. This enables us to then concatenate
    the subjects data together along the trial dimensions, essentially adding channels to the data.
    :param sub_epochs: (dict of mne epochs) epochs for each subject to equate trials across
    :param condition: (string) name of the condition of interest. Must match one of the column of the metadata objects
    :param min_per_label: (int) minimum number of trial per condition
    :param groups_condition: (string or None) group condition the trials of the first condition in case of nested
    designs. For example, when the condition is "identity", each "identity" belongs to a specific category: face,
    objects... So if you pass "identity" as condition and "category" as groups_condition, you will have the labels be
    something like:
    face_01
    face_02
    face_03...
    And the group labels be something like this:
    face
    face
    face
    Having this information can be useful for subsequent operations
    :param verbose: (Boolean) whether or not to print additional information to the command line
    :return:
    data_arrays: (np array) super subject data, trials x channels x samples
    labels: (np array) label of each trial according to the condition passed. So if you passed category, then for each
    trial, you'll have face, object, face...
    groups: (np array) group to which each trial belongs
    times: (np array) times arrays with one time in sec for each sample. If these do not match across subjects,
    error is thrown
    sfreq: (float) sampling frequency that was common across subjects. If it isn't error is thrown
    channels: (list) list of channels that were used
    """
    # Looping through each subject to reject any trials that has less than the expected number of trials:
    sub_labels = {subject: None for subject in sub_epochs.keys()}
    sub_counts = {subject: None for subject in sub_epochs.keys()}
    times = []
    sfreq = []
    channels = []

    # ===============================================================================
    # Extract single subject trials information:
    for sub in sub_epochs.keys():
        # Equate the trial counts to match what is expected:
        if min_per_label is not None:
            sub_epochs[sub] = remove_too_few_trials(sub_epochs[sub], condition=condition,
                                                    min_n_repeats=min_per_label, verbose=verbose)
        # Get trial counts of that subject:
        unique, counts = np.unique(sub_epochs[sub].metadata[condition].to_numpy(), return_counts=True)
        sub_counts[sub] = dict(zip(unique, counts))
        # Get the labels:
        sub_labels[sub] = sub_epochs[sub].metadata[condition].to_numpy()
        # Getting time and sfreq just to make sure that these add up across subs:
        times.append(sub_epochs[sub].times)
        sfreq.append(sub_epochs[sub].info["sfreq"])
    # Make sure that all the sfreqs are the same:
    if len(set(sfreq)) > 1:
        raise Exception("ERROR: The sampling frequency is not consistent across the different subjects!")
    else:
        sfreq = sfreq[0]
    # If sfreq is the same across subjects, we can compare the onset and number of samples across all to make sure that
    # time fully adds up:
    if len(set([ti[0] for ti in times])) > 1:
        raise Exception("ERROR: The onset of the epochs is not the same across all the subjects!")
    if len(set([len(tim) for tim in times])) > 1:
        raise Exception("ERROR: The length of the epochs is not consistent across subjects!")
    else:
        times = times[0]

    # Compute super subject trials information:
    # Get the intersection of condition across subjects:
    intersection = set.intersection(*map(set, [sub_labels[sub] for sub in sub_labels.keys()]))

    # ===============================================================================
    # Generate the super subject data:
    # Now, extracting the data according to the intersection:
    data_arrays = []
    labels_arrays = []
    # Handling groups:
    if groups_condition is not None:
        groups = []
    else:
        groups = None
    # Looping through each label:
    for ind, label in enumerate(intersection):
        # Get the minimal counts for this specific label:
        min_count = min([sub_counts[subject][label] for subject in sub_epochs.keys()])
        label_data = []
        # Handling groups:
        if groups_condition is not None:
            label_groups = []
        else:
            label_groups = None
        for sub in sub_epochs.keys():
            if ind == 0:
                channels.extend(sub_epochs[sub].ch_names)
            # Extract this subject epochs:
            sub_label_data = sub_epochs[sub].copy()[label].get_data()
            # Get the indices of the trials to pick:
            trials_ind = np.random.choice(sub_label_data.shape[0], min_count, replace=False)
            # Randomly picking n trials out of this:
            label_data.append(sub_label_data[trials_ind, :, :])
            # If requied, fetching the group condition from each trial. So if you passed category as groups condition,
            # it will fetch the category of each trial you have passed
            if groups_condition is not None:
                label_groups.append(sub_epochs[sub].copy()[label].metadata[groups_condition].to_numpy()[trials_ind])

        # Extend the labels list
        labels_arrays.extend([label] * label_data[0].shape[0])
        data_arrays.append(np.concatenate(label_data, axis=1))
        # Handling the group information:
        if groups_condition is not None:
            # Depending on the design, it could be that across different participants, the group a given trial belongs
            # to varies. If that is the case, there is a high likelyhood that things will break. Therefore, throwing
            # error. So say you pass task demand as a grouping condition, in subject 1, the first 4 face identities
            # might be task relevant, but for subject, it would be task irrelevant. In that case, everything breaks!
            # Finding out whether there is a unique set of group across subjects:
            unique_group_sets = [list(x) for x in set(tuple(x) for x in label_groups)]
            if len(unique_group_sets) > 1:
                raise Exception("ERRO: The group of each condition differed across subjects!")
            # Otherwise, adding the unique group set to the group variable
            groups.extend(unique_group_sets[0])

    # Convert the whole to arrays:
    labels = np.array(labels_arrays)
    data_arrays = np.concatenate(data_arrays, axis=0)
    # Making sure that the number of channels matches between the channel array and the final matrix:
    assert len(channels) == data_arrays.shape[1]
    channels = np.array(channels)
    if groups_condition is not None:
        groups = np.array(groups)

    return data_arrays, labels, groups, times, sfreq, channels


def equate_offset(epochs, cropping_dict):
    """
    This function excise some time points from certain conditions that differ in durations to make the offset consistent
    between different durations. This enables to crop out chunks of data at specified time points and sew the rest
    back together.
    :param epochs: (mne epochs) epochs to chop and sew back up.
    :param cropping_dict: (dictionary) contains the info about what and when to crop:
    "1500ms": {
          "excise_onset": 1.0,
          "excise_offset": 1.5
        }
    The dictionary key corresponds to a specific experimental condition wished to be cropped, excise_onset to when to
    start cropping and excise_offset when to stop
    :return:
    equated_epochs: mne epochs object with epochs cropped
    """
    # Looping through each condition for which some time needs to be excised:
    conds_epochs = []
    for cond in cropping_dict.keys():
        # Getting time and rounding at 3 decimals to avoid weird indexing issues
        times = np.around(epochs.times, decimals=3)
        # Get the data of that one condition:
        cond_epochs = epochs.copy()[cond]
        # Now, get the time points to excise:
        # The onset is the first point in the time vector that is superior or equal to the onset
        excise_onset_ind = np.where(times >= cropping_dict[cond]["excise_onset"])[0][0]
        # The offset is the last point in time that is inferior or equal to the offset. Need to add 1 to it,
        # because in python, slicing doesn't take the last point (i.e. :n-1). But in the case where our offset is at
        # 2 sec for ex, and the time vector goes from 0 to 2.5, then we want to take the point 2.0 in, not go only until
        # 1.98 or something like that.
        excise_offset_ind = np.where(times <= cropping_dict[cond]["excise_offset"])[0][-1] + 1
        print("Excising data from {} to {} from condition".format(times[excise_onset_ind],
                                                                  times[excise_offset_ind - 1],
                                                                  cond))
        # Excising:
        cond_epochs_data = np.delete(cond_epochs.get_data(), range(excise_onset_ind, excise_offset_ind), axis=-1)
        # Create a new epochs object:
        conds_epochs.append(mne.EpochsArray(cond_epochs_data, cond_epochs.info, events=cond_epochs.events,
                                            tmin=cond_epochs.times[0], event_id=cond_epochs.event_id,
                                            baseline=cond_epochs.baseline,
                                            metadata=cond_epochs.metadata, on_missing="warn"))
    # Combining the epochs data:
    equated_epochs = mne.concatenate_epochs(conds_epochs, add_offset=False)

    return equated_epochs


def regress_evoked(epochs):
    """
    This function computes the evoked responses and regresses it out from every single trial
    :param epochs: (mne epochs object) epochs from which the evoked should be regressed
    :return: (mne epochs object) epochs from which the evoked response is regressed from
    """
    print("=" * 40)
    print("Welcome to regress_evoked")
    # Compute the evoked:
    evoked = epochs.average()
    # Extracting the data from the mne objects:
    epochs_data = epochs.get_data()
    evoked_data = evoked.get_data()
    print("Regressing evoked response out of every trial per channel")
    for channel in range(epochs_data.shape[1]):
        ch_evk = evoked_data[channel, :]
        for trial in range(epochs_data.shape[0]):
            epochs_data[trial, channel, :] = sm.OLS(epochs_data[trial, channel, :], ch_evk).fit().resid
    # Packaging everything back into an mne epochs object:
    epochs_regress = mne.EpochsArray(epochs_data, epochs.info, events=epochs.events,
                                     tmin=epochs.times[0], event_id=epochs.event_id, baseline=epochs.baseline,
                                     metadata=epochs.metadata, on_missing="warn")
    return epochs_regress


def create_prediction_matrix(start, end, predicted_intervals, matrix_size):
    """
    This function generates binary matrix of zero and ones according to theories predictions. This can then later be
    compared to the results of the decoding. 1 is for when a theory predicts above chance decoding, 0 for no decoding
    :param start: (float or int) start time of the decoding matrix. In secs
    :param end: (float or int) end time of the decoding matrix. In secs
    :param predicted_intervals: (dict of list of floats) contains the predicted onsets and offsets of above chance
    decoding
    in the start to end time vector. The format is like so:
    {
        "x": [[0.3, 0.5], [0.8, 1.5]...],
        "y": [[0.3, 0.5], [0.8, 1.5]...],
    }
    IMPORTANT: The x and y must both have the same number of entries!
    :param matrix_size: (int) size of the matrix tio generate. Must be the same size as the decoding matrix to compare
    it to.
    :return: (dict) predicted_matrix binary matrix containing predicted significant decoding
    """
    if len(predicted_intervals["x"]) != len(predicted_intervals["y"]):
        raise Exception("The x and y coordinates of the predicted intervals have different lengths! That doesn't work!")
    # Create a matrix of zeros of the correct size:
    predicted_matrix = np.zeros((matrix_size, matrix_size))
    # Generating a time vector matching the matrix size:
    time_vect = np.around(np.linspace(start, end, num=matrix_size), decimals=3)
    # The theories make prediction such that there will be decoding within specific time windows. Looping through those:
    for ind, interval_x in enumerate(predicted_intervals["x"]):
        # Finding the time points corresponding to the start and end of the prediction
        # The onset is the first point in the time vector that is superior or equal to the onset
        onset_x = np.where(time_vect >= interval_x[0])[0][0]
        # The offset is the last point in time that is inferior or equal to the offset. Need to add 1 to it,
        # because in python, slicing doesn't take the last point (i.e. :n-1). But in the case where our offset is at
        # 2 sec for ex, and the time vector goes from 0 to 2.5, then we want to take the point 2.0 in, not go only until
        # 1.98 or something like that.
        offset_x = np.where(time_vect <= interval_x[1])[0][-1] + 1
        # Same for y:
        onset_y = np.where(time_vect >= predicted_intervals["y"][ind][0])[0][0]
        offset_y = np.where(time_vect <= predicted_intervals["y"][ind][1])[0][-1] + 1
        # Setting all these samples to 1:
        if not isinstance(predicted_intervals["predicted_vals"][ind], str):
            predicted_matrix[onset_y:offset_y, onset_x:offset_x] = predicted_intervals["predicted_vals"][ind]
        elif predicted_intervals["predicted_vals"][ind].lower() == "nan":
            predicted_matrix[onset_y:offset_y, onset_x:offset_x] = np.nan
        else:
            raise Exception("The predicted value must be either a number (float or int) or nan, check spelling!!")

    return predicted_matrix


def remove_too_few_trials(epochs, condition="identity", min_n_repeats=2, verbose=False):
    """
    This function removes the conditions for which there are less than min_n_repeats. So say you only want conditions
    for which you have at least 2 repeats, set min_n_repeats to 2.
    :param epochs: (mne epochs object) contains the data and metadata to remove conditions from
    :param condition: (string) name of the condition for which to equate. The string must match a column in the metadata
    :param min_n_repeats: (int) minimal number of repeats a condition must have to pass!
    :param verbose: (bool) whether or not to print information to the command line
    :return:
    epochs: (mne epochs object) the mne object with equated trials. Note that the function modifies data in place!
    """
    if verbose:
        print("Equating trials by downsampling {}".format(condition))
    # Get the meta data for that subject:
    sub_metadata = epochs.metadata.reset_index(drop=True)
    # Find the identity for which we have less than two trials:
    cts = sub_metadata.groupby([condition])[condition].count()
    id_to_remove = [identity for identity in cts.keys() if cts[identity] < min_n_repeats]
    if verbose:
        print("The following identity have less than two repetitions")
        print(id_to_remove)
    # Get the indices of the said identity to drop the trials:
    id_idx = sub_metadata.loc[sub_metadata[condition].isin(id_to_remove)].index.values.tolist()
    # Dropping those:
    epochs.drop(id_idx, verbose="error")
    return epochs


def cross_identity_cross_temp_rsm(data, labels, metric="correlation"):
    """
    This function computes cross temporal representation similiary between different identities. identity.
    So say you have 5 different trials identity repeated twice each. What this function does is compute all possible
    combinations of the single identities:
    id 1 : id 2
    id 1 : id 3
    ...
    id 2 : id 3
    ...
    And compute cross temporal rsm between those. This tells you how much the representation across different identities
    is sustained along time. The main purpose is as a comparison to the output of within_identity_crosstemp_rsm. This
    is basically a baseline for within condition
    :return:
    """
    print("=" * 40)
    print("Welcome to cross_identity_cross_temp_rsm")
    # Make sure the labels are a numpy array:
    assert isinstance(labels, np.ndarray), "The labels were not of type np.array!"
    # This function assumes that there are only two repetition of the same trial. Ensuring that this is the case:
    for ids in list(set(labels)):
        if len([el for el in labels if el == ids]) != 2:
            raise Exception("There were {} repetitions of id {}! This function only works with 2 repetition of each "
                            "ID!".format(len([el for el in labels if el == ids]), ids))
    # Compute the combinations of unique labels:
    identity_pairs = list(itertools.combinations(labels, 2))
    # Removing any pairs that are the same identity, because it is exactly what we don't want:
    ind_to_drop = [ind for ind, _ in enumerate(identity_pairs) if len(set(identity_pairs[ind])) == 1]
    identity_pairs = [val for ind, val in enumerate(identity_pairs) if ind not in ind_to_drop]

    n_pairs = len(identity_pairs)
    correlation_matrix = np.zeros((n_pairs, data.shape[-1], data.shape[-1]))

    # Looping through each identity:
    for ind, pair in enumerate(identity_pairs):
        # Getting the index of each face:
        id1_inds = np.where(labels == pair[0])[0]
        id2_inds = np.where(labels == pair[1])[0]
        # Now, looping through time:
        for t1 in range(0, data.shape[-1]):
            # Pick the data randomly of one of the two repetition of ID 1
            d1 = data[np.random.choice(id1_inds, size=1), :, t1]
            for t2 in range(0, data.shape[-1]):
                d2 = data[np.random.choice(id2_inds, size=1), :, t2]
                correlation_matrix[ind, t1, t2] = 1 - cdist(d2, d1, metric)

    return correlation_matrix


def rdm_regress_groups(rdm, groups_1, groups_2):
    """
    This function regresses out the group effect from the rdm. The group effect is encoded through the two groups
    variables here. Groups_1 matches the 1 dim of the rdm and groups 2 the 2nd dim and identified which group a trial
    belonged to. And so if the group of group 1 and 2 match at a given intersection, then this is a within group and if
    they don't across. Within is encoded as 1 and across as a zero. The matrix is then flattened and regressed out from
    the rdm. In other words:
            face, face, object, object
    face,     1     1     0       0
    face,     1     1     0       0
    object,   1     1     0       0
    object    1     1     0       0
    :param rdm: (2D numpy array) rdm from which the group should be regressed. The first dimension corresponds to the
    first set of trials to compute the rdm and the 2d the second set of trials
    :param groups_1: (numpy array) group to which the first set of trials belong to
    :param groups_2: (numpy array) group to which the second set of trials belong to
    :return:
    rdm: the same rdm but with the group information regressed out
    """

    # Generating the groups matrix:
    groups_regressor = np.zeros(rdm.shape)
    for i in range(groups_regressor.shape[0]):
        for ii in range(groups_regressor.shape[1]):
            if groups_1[i] == groups_2[ii]:
                groups_regressor[i, ii] = 1
            else:
                groups_regressor[i, ii] = 0

    # Flatten the two matrices:
    rdm_flat = rdm.flatten()
    groups_regressor_flat = groups_regressor.flatten()

    # Adding a couple of checks just to be sure the reshape is never messed up. My understanding is that it can't be
    # messed up the way I do it, but that way it really can't be!
    np.testing.assert_array_equal(rdm_flat.reshape(rdm.shape), rdm)
    np.testing.assert_array_equal(groups_regressor_flat.reshape(groups_regressor.shape), groups_regressor)

    # Regress teh groups regressor out:
    rdm_regress_flat = sm.OLS(rdm_flat, groups_regressor_flat).fit().resid

    # Finally, reconverting it to a square matrix:
    return rdm_regress_flat.reshape(rdm.shape)


def all_to_all_within_class_dist(data, labels, metric="correlation", n_bootsstrap=20, shuffle_labels=False,
                                 fisher_transform=True, verbose=False, n_features=None, n_folds=None,
                                 feat_sel_diag=True, half_rdm=True):
    """
    This function computes all trials to all trials distances and computes within class correlated distances in a
    cross temporal fashion.
    :param data:
    :param labels:
    :param metric:
    :param n_bootsstrap:
    :param shuffle_labels:
    :param fisher_transform:
    :param verbose:
    :param n_features:
    :param n_folds:
    :param feat_sel_diag:
    :param split_half_rdm:
    :return:
    """
    if verbose:
        print("=" * 40)
        print("Welcome to cross_identity_cross_temp_rsm")
    # Spearman is not implemented by default with cdist:
    if metric == "spearman":
        metric = lambda a, b: 1 - spearmanr(a, b, axis=0).correlation

    # Make sure the labels are a numpy array:
    assert isinstance(labels, np.ndarray), "The labels were not of type np.array!"
    # Shuffle labels if needed:
    if shuffle_labels:
        perm_ind = np.random.permutation(len(labels))
        labels = labels[perm_ind]

    # Pre-allocating for the diagonal RDM:
    rdm_diag = []
    if n_features is None or feat_sel_diag:
        # Preallocating for the rsa:
        rsa_matrix = np.zeros((data.shape[-1], data.shape[-1]))
        for t1 in range(0, data.shape[-1]):
            # Get the data at t1 from train set:
            d1 = np.squeeze(data[:, :, t1])
            if n_features is not None:
                # Perform the feature selection on the test set
                features_sel = SelectKBest(f_classif, k=n_features).fit(d1, labels)
            # Now looping through every other time point:
            for t2 in range(0, data.shape[-1]):
                # Get the data at t2 from the test set:
                d2 = np.squeeze(data[:, :, t2])

                # Compute the RDM:
                if n_features is not None:
                    rdm = cdist(features_sel.transform(d1), features_sel.transform(d2), metric)
                else:
                    rdm = cdist(d1, d2, metric)

                if metric == "correlation" and fisher_transform:
                    # Performing the fisher transformation of the correlation values (converting distances to
                    # correlation, fisher transform, back into distances):
                    rdm = 1 - np.arctanh(1 - rdm)

                # If we are along the diagona, store the rdm:
                if t1 == t2:
                    rdm_diag.append(rdm)

                # Compute the within class correlated distances::
                msk_within = np.meshgrid(labels, labels)[1] == \
                             np.meshgrid(labels, labels)[0]
                msk_between = np.meshgrid(labels, labels)[1] != \
                              np.meshgrid(labels, labels)[0]
                np.fill_diagonal(msk_within, False)
                np.fill_diagonal(msk_between, False)
                within_val = rdm[msk_within]
                across_val = rdm[msk_between]
                # Finally, computing the correlation between the rsa_matrix at t1 and t2:
                if n_bootsstrap is not None:
                    if len(within_val) != len(across_val):
                        # Find the minimal samples between the within and across:
                        min_samples = min([len(within_val), len(across_val)])
                        bootstrap_diff = []
                        for n in range(n_bootsstrap):
                            bootstrap_diff.append(np.mean(np.random.choice(across_val, min_samples, replace=False)) -
                                                  np.mean(np.random.choice(within_val, min_samples, replace=False)))
                        rsa_matrix[t1, t2] = np.mean(bootstrap_diff)
                    else:
                        rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)
                else:
                    rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)
    else:
        # Using stratified kfold to perform feature selection:
        skf = StratifiedKFold(n_splits=n_folds)
        folds_mat = []
        # Splitting the data in nfolds, selecting features on one fold and testing on the rest:
        for test_ind, feat_sel_ind in skf.split(data, labels):
            # Preallocate for the rdm:
            rsa_matrix = np.zeros((data.shape[-1], data.shape[-1]))
            # Compute the cross temporal RDM:
            for t1 in range(0, data.shape[-1]):
                # Perform the feature selection on the test set
                features_sel = SelectKBest(f_classif, k=n_features).fit(data[feat_sel_ind, :, t1], labels[feat_sel_ind])

                # Get the data at t1 from train set:
                d1 = np.squeeze(features_sel.transform(data[test_ind, :, t1]))

                # Now looping through every other time point:
                for t2 in range(0, data.shape[-1]):
                    # Get the data at t2 from the test set:
                    d2 = np.squeeze(features_sel.transform(data[test_ind, :, t2]))

                    # Compute the RDM:
                    rdm = cdist(d1, d2, metric)
                    if metric == "correlation" and fisher_transform:
                        # Performing the fisher transformation of the correlation values (converting distances to
                        # correlation fisher transform, back into distances):
                        rdm = 1 - np.arctanh(1 - rdm)

                    # Compute the within class correlated distances::
                    msk_within = np.meshgrid(labels[test_ind], labels[test_ind])[1] == \
                                 np.meshgrid(labels[test_ind], labels[test_ind])[0]
                    msk_between = np.meshgrid(labels[test_ind], labels[test_ind])[1] != \
                                  np.meshgrid(labels[test_ind], labels[test_ind])[0]
                    np.fill_diagonal(msk_within, False)
                    np.fill_diagonal(msk_between, False)
                    within_val = rdm[msk_within]
                    across_val = rdm[msk_between]
                    # Finally, computing the correlation between the rsa_matrix at t1 and t2:
                    if n_bootsstrap is not None:
                        if len(within_val) != len(across_val):
                            # Find the minimal samples between the within and across:
                            min_samples = min([len(within_val), len(across_val)])
                            bootstrap_diff = []
                            for n in range(n_bootsstrap):
                                bootstrap_diff.append(
                                    np.mean(np.random.choice(across_val, min_samples, replace=False)) -
                                    np.mean(np.random.choice(within_val, min_samples, replace=False)))
                            rsa_matrix[t1, t2] = np.mean(bootstrap_diff)
                        else:
                            rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)
                    else:
                        rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)
            # Append to the fold mat:
            folds_mat.append(rsa_matrix)
        # Average across folds:
        rsa_matrix = np.average(np.array(folds_mat), axis=0)
        # Compute the diagonal RDMs:
        rdm_diag = [cdist(data[:, :, t1], data[:, :, t1], metric) for t1 in range(0, data.shape[-1])]

    # Compute the RDM by subsampling
    # Computing rdm by splitting half of the electrodes if needed to avoid structures due to trial number issues:
    if half_rdm:
        # Randomly splitting the data in two halfs:
        split_half_msk = np.ones(data.shape[1], dtype=bool)
        split_half_msk[np.random.choice(range(data.shape[1]), size=int(data.shape[1] / 2))] = False
        split_half_rdm = []
        for t1 in range(0, data.shape[-1]):
            rdm_half = []
            for i in range(2):
                if i == 0:
                    rdm_half.append(cdist(data[:, split_half_msk, t1], data[:, split_half_msk, t1], metric))
                else:
                    rdm_half.append(cdist(data[:, ~split_half_msk, t1], data[:, ~split_half_msk, t1], metric))
            # average across both halves:
            split_half_rdm.append(np.mean(np.array(rdm_half), axis=0))

    else:
        split_half_rdm = None

    return rsa_matrix, rdm_diag, split_half_rdm


def within_vs_between_cross_temp_rsa(data, labels,
                                     metric="correlation", n_bootsstrap=100, zscore=False,
                                     sample_rdm_times=None,
                                     onset_offset=None, n_features=40, n_folds=5,
                                     shuffle_labels=False, fisher_transform=True,
                                     verbose=False, feat_sel_diag=True, store_intermediate=False):
    """
    This function computes cross temporal RDM by computing distances between two repetitions of the same trial
    identities as well as between different identities at different time points. Then, the difference is computed
    between the within (i.e. diagonal) and between (off diagonal condition). This results in one metric per time x time
    pixel that summarizes the extent to which identity is conserved between different time point.
    :param onset_offset:
    :param data: (numpy array) contains the data with shape trials x channels x time points for which to compute the
    rsa.
    :param labels: (numpy array) contains the labels of the first dimension of the data
    :param metric: (string) metric to use to compute the distances. See from scipy.spatial.distance import cdist for
    options
    :param n_bootsstrap: (int) number of bootstrap in case the number of samples in within class differs from across
    classes
    :param zscore: (boolean) whether or not to zscore the data at each distance computations
    :param sample_rdm_times: (list of time points) compute a sample RDM at this time points. This enables getting a
    :param n_features: (int) number of features to select. The features are selected in a quite complicated way.
    We are basically splitting the data such that the features are always selected on a different set of data than what
    is used to compute the correlation.
    :param n_folds: (int) number of folds for the feature selection
    :param shuffle_labels: (bool) wheher or not to shuffle the labels
    :param fisher_transform: (bool) whehther or not to fisher transform the correlation value before computing within
    vs between
    :param verbose: (bool) whether or not to print additional info to command line
    :param feat_sel_diag: (bool) whether to perform the feature selection on the diagonal only. If yes, then performing
    the feature selection only once on the "train set" for each time point along the "y axis".
    :return:
    rsa_matrix: (numpy array) cross temporal rsa matrix that was computed
    sample_rdm: (numpy array) representation dissimilarity matrix according to the time points passed under
    sample_rdm_times
    """
    # Deactivate the warnings, because in some cases, scikit learn will send so many warnings that the logs get
    # completely overcrowded:
    import warnings
    warnings.filterwarnings('ignore')
    if metric != "correlation" and fisher_transform:
        if verbose:
            print("WARNING: fisher transform only applies for correlation!")
    if onset_offset is None:
        onset_offset = [-0.3, 1.5]
    if verbose:
        print("=" * 40)
        print("Welcome to cross_identity_cross_temp_rsm")
    # Make sure the labels are a numpy array:
    assert isinstance(labels, np.ndarray), "The labels were not of type np.array!"
    # Shuffle labels if needed:
    if shuffle_labels:
        perm_ind = np.random.permutation(len(labels))
        labels = labels[perm_ind]
    # Use scikit learn cross validation to compute the split between "first" and "second" presentation:
    skf = StratifiedKFold(n_splits=2)
    # Preallocating for the rsa:
    rsa_matrix = np.zeros((data.shape[-1], data.shape[-1]))

    # Pre-allocating for the diagonal RDM:
    rdm_diag = []
    # Extract the indices of first and second pres_
    first_pres_ind, second_pres_ind = list(skf.split(data, labels))[0]
    # Extract the label of each:
    first_pres_labels = labels[first_pres_ind]
    second_pres_labels = labels[second_pres_ind]
    # Store a list of the features that were used:
    sel_features = []
    if verbose:
        print("=" * 40)
        print("Welcome to cross_identity_cross_temp_rsm")
        print("First presentation:", first_pres_labels, "\nSecond presentation:", second_pres_labels)
        print("Computing representation similarity between all time points")
    for t1 in range(0, data.shape[-1]):
        # Get the data at t1 from train set:
        d1 = np.squeeze(data[first_pres_ind, :, t1])
        if zscore:
            scaler = StandardScaler()
            scaler.fit(d1)
            d1 = scaler.transform(d1)
        # If the features selection is done along the diagonal only:
        if n_features is not None and feat_sel_diag:
            # Extract the features on the first split of the data at the current time point:
            features_sel = SelectKBest(f_classif, k=n_features).fit(d1, first_pres_labels)
            sel_features.append(features_sel.get_support(indices=True))
        # Now looping through every other time point:
        for t2 in range(0, data.shape[-1]):
            # Get the data at t2 from the test set:
            d2 = np.squeeze(data[second_pres_ind, :, t2])
            if zscore:
                d2 = scaler.transform(d2)
            # Compute the distance between all combinations of trials:
            # Selecting features if needed:
            if n_features is not None and not feat_sel_diag:
                # Prepare the rdm matrix:
                rdm = np.zeros([d1.shape[0], d2.shape[0]])
                # Selecting features with a cross validation to avoid double dipping:
                # Counting the number of events per halves:
                first_pres_labels_cts = Counter(first_pres_labels)
                second_pres_labels_cts = Counter(second_pres_labels)
                # If each item occurs less often than there are folds, then using k fold:
                if all(i < n_folds for i in list(first_pres_labels_cts.values())) \
                        or all(i < n_folds for i in list(second_pres_labels_cts.values())):
                    f_d1 = KFold(n_splits=n_folds)
                    f_d2 = KFold(n_splits=n_folds)
                else:
                    f_d1 = StratifiedKFold(n_splits=n_folds)
                    f_d2 = StratifiedKFold(n_splits=n_folds)
                # Looping through the folds of d1:
                for train_d1, test_d1 in f_d1.split(d1, first_pres_labels):
                    # Looping through d2 folds:
                    for train_d2, test_d2 in f_d2.split(d2, second_pres_labels):
                        # Extract the data for the feature selection:
                        feature_sel_data = np.concatenate([d1[train_d1, :], d2[train_d2, :]], axis=0)
                        feature_sel_labels = np.concatenate([first_pres_labels[train_d1],
                                                             second_pres_labels[train_d2]],
                                                            axis=0)
                        features_sel = SelectKBest(f_classif, k=n_features).fit(feature_sel_data, feature_sel_labels)
                        d1_test = features_sel.transform(d1[test_d1, :])
                        d2_test = features_sel.transform(d2[test_d2, :])
                        sub_rdm = cdist(d1_test, d2_test, metric)
                        for ind_1, rdm_row in enumerate(test_d1):
                            for ind_2, rdm_col in enumerate(test_d2):
                                rdm[rdm_row, rdm_col] = sub_rdm[ind_1, ind_2]
            elif n_features is not None and feat_sel_diag:
                # Compute the rdm based on the feature selection performed on this data:
                rdm = cdist(features_sel.transform(d1), features_sel.transform(d2), metric)
            else:
                rdm = cdist(d1, d2, metric)
            if metric == "correlation" and fisher_transform:
                # Performing the fisher transformation of the correlation values (converting distances to correlation,
                # fisher transform, back into distances):
                rdm = 1 - np.arctanh(1 - rdm)

            if t1 == t2:
                # Compute an all to all RDM for multidimensional scaling plots for showing the results:
                if n_features is not None and feat_sel_diag:
                    all_to_all_rdm = cdist(np.concatenate([features_sel.transform(d1), features_sel.transform(d2)]),
                                           np.concatenate([features_sel.transform(d1), features_sel.transform(d2)]),
                                           metric)
                elif n_features is None:
                    all_to_all_rdm = cdist(np.concatenate([d1, d2]),
                                           np.concatenate([d1, d2]), metric)
                else:  # It is not possible if the feature selection is not done along the diagonal!
                    all_to_all_rdm = rdm
                rdm_diag.append(all_to_all_rdm)

            # Create a mask with values == true for within condition, false otherwise:
            msk = np.meshgrid(second_pres_labels, first_pres_labels)[1] == \
                  np.meshgrid(second_pres_labels, first_pres_labels)[0]
            within_val = rdm[msk]
            across_val = rdm[~msk]
            # Finally, computing the correlation between the rsa_matrix at t1 and t2:
            if n_bootsstrap is not None:
                if len(within_val) != len(across_val):
                    # Find the minimal samples between the within and across:
                    min_samples = min([len(within_val), len(across_val)])
                    bootstrap_diff = []
                    for n in range(n_bootsstrap):
                        bootstrap_diff.append(np.mean(np.random.choice(across_val, min_samples, replace=False)) -
                                              np.mean(np.random.choice(within_val, min_samples, replace=False)))
                    rsa_matrix[t1, t2] = np.mean(bootstrap_diff)
                else:
                    rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)
            else:
                rsa_matrix[t1, t2] = np.mean(across_val) - np.mean(within_val)

    return rsa_matrix, rdm_diag, first_pres_labels, second_pres_labels, sel_features
