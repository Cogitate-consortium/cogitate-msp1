"""Helper functions for decoding analysis."""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import os
from pathlib import Path
import numpy as np

from scipy import stats

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from skimage.measure import block_reduce

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import cm, colors, colorbar

import time

import mne
from mne.stats import permutation_cluster_1samp_test, bootstrap_confidence_interval
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, get_coef, LinearModel)
from mne.baseline import rescale

from general_helper_functions.data_general_utilities import moving_average, baseline_scaling

font = {'weight': 'bold',
        'size': 14}

matplotlib.rc('font', **font)

# figure(figsize=(8, 6), dpi=80)
SUPPORTED_CLASSIFIERS = ["Perceptron", "linear_svm"]


def sum_of_square(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)


def plot_roi_specificity(root, rois, decoding_scores, cmap="Purples", vmin=0.5, vmax=1., filename=None):
    views = [{'azimuth': 0, 'elevation': -90}, {'azimuth': 180, 'elevation': -120}]
    surf = "inflated"
    parc = "aparc.a2009s"

    subjects_dir = Path(root, "derivatives", "fs")

    # Get the brain object:
    Brain = mne.viz.get_brain_class()

    # ====================================================================
    # Getting the surface labels:
    labels = mne.read_labels_from_annot("fsaverage", parc=parc, hemi='both', surf_name=surf,
                                        annot_fname=None, subjects_dir=subjects_dir, sort=True, verbose=None)

    # Plot the brain:
    brain = Brain('fsaverage', 'lh', surf, subjects_dir=subjects_dir, cortex='low_contrast',
                  background='white', size=(800, 400), alpha=1)

    # Generating the color scale for this group:
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Adding the color bar:
    # reindex the colormap so it starts deeper into the colorscale (e.g. color values = 0.1-1 over the scale of
    # vmin-vmax (ACC) of purple)
    cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))

    # Looping through each label in the data_df:
    for ii, label in enumerate(rois):
        # Get the data of that specific label:
        label_val = decoding_scores[ii]
        if label_val.ndim > 0:
            label_val = np.mean(label_val)
        # Get the label object for the specific label:
        label_obj = [label_object for label_object in labels if label in label_object.name]
        # assert len(label_obj) == 1, "There was more than one label matching the name: {}!".format(label)
        # Plot this label according to the color scheme:
        print('%s: %f' % (label, label_val))

        brain.add_label(label_obj[0], color=cmap(norm(label_val)), alpha=1, hemi="lh")

    # Finally, getting the different views to plot and save:
    fig = plt.figure(figsize=(8, 4))
    for view_ind, view in enumerate(views):
        if isinstance(view, str):
            brain.show_view(view)
        elif isinstance(view, dict):
            brain.show_view(**view)
        else:
            raise Exception("The view must be a list of strings or of dict!")
        im = brain.screenshot()
        ax = fig.add_axes([(view_ind * 0.5), 0.2, 0.5, 0.6])
        ax.axis('equal')
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.axis('off')
        # crop out whitespace
        nonwhite_pix = (im != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = im[nonwhite_row][:, nonwhite_col]
        plt.tight_layout()
        # Plot the surface
        ax.imshow(cropped_screenshot)
    ax = fig.add_axes([0.47, 0.1, 0.015, 0.13])
    # norm = mpl.colors.Normalize(vmin=0.5, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional',
                                   orientation="vertical")
    cb.set_ticks((vmin, vmax))
    cb.ax.tick_params(size=0, labelsize=8)
    cb.ax.set_title('ACC', fontsize=8)

    if filename is not None:
        plt.savefig(filename, dpi=150)


def plot_decoding_data(x, y, times, save_path=None, roi="", file_prefix="", sampling_freq=512):
    """

    :param x:
    :param y:
    :param times:
    :param save_path:
    :param roi:
    :param file_prefix:
    :param sampling_freq:
    :return:
    """
    # Averaging the data per trial type:
    cond_data = []
    for cond in set(y):
        cond_data.append(np.mean(x[np.where(y == cond)[0]], axis=0))
    x_heatmap = np.vstack(cond_data)

    # Plot a heatmap of that:
    fig, ax = plt.subplots()
    # Plotting the comparison:
    ax.imshow(x_heatmap, extent=list(times[[0, -1]]) + [0, x_heatmap.shape[0]], aspect="auto")
    ax.hlines(y=x_heatmap.shape[0] / 2, linestyles="-.", xmin=times[0], xmax=times[-1], color="k")
    ax.vlines(x=0, ymin=0, ymax=x_heatmap.shape[0], color="k", linestyles="-.")
    ax.set_ylabel("channels")
    ax.set_xlabel("Time")
    plt.title("Channels mean activations across trials")
    ax.set_yticks([x_heatmap.shape[0] / 4, 3 * x_heatmap.shape[0] / 4])
    ax.set_yticklabels(set(y))
    if save_path is not None:
        file_name = Path(save_path, file_prefix
                         + "heatmap_{0}.png".format(roi))
        fig.savefig(file_name, transparent=True)
    plt.close(fig)

    # Create the events dict:
    event_dict = {cond: int(i) for i, cond in enumerate(np.unique(y))}
    # Group the data per conditions:
    data = []
    events_desc = []
    for cond in np.unique(y):
        data.extend(x[(np.where(y == cond)[0])])
        events_desc.extend([event_dict[cond]] * len(np.where(y == cond)[0]))
    events = np.column_stack((np.arange(0, len(data) * (times[-1] - times[0]) * sampling_freq,
                                        (times[-1] - times[0]) * sampling_freq, dtype=int),
                              np.zeros(len(events_desc), dtype=int),
                              np.array(events_desc)))
    # Create the info:
    ch_names = ["ch_{}".format(i) for i in range(x.shape[1])]
    info = mne.create_info(ch_names, ch_types=["ecog"] * x.shape[1], sfreq=sampling_freq)
    new_epochs = mne.EpochsArray(data, info, tmin=times[0], events=events,
                                 event_id=event_dict)

    # -------------------------------
    # Compute the global field power (following tutorial here:
    # https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html):
    evoked = new_epochs.average()
    gfp = np.sum(evoked.data ** 2, axis=0)
    times = evoked.times * 1e3
    gfp = rescale(gfp, times, baseline=(None, 0))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, gfp, linewidth=2.5)
    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
    ci_low, ci_up = bootstrap_confidence_interval(evoked.data, random_state=0,
                                                  stat_fun=sum_of_square)
    ci_low = rescale(ci_low, evoked.times, baseline=(None, 0))
    ci_up = rescale(ci_up, evoked.times, baseline=(None, 0))
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('GFP')
    ax.set_title("Global field power across all electrodes in roi {}".format(roi))
    if save_path is not None:
        file_name = Path(save_path, file_prefix
                         + "gfp_{0}.png".format(roi))
        plt.savefig(file_name, transparent=True)
    # Z-scoring the data:
    baseline_scaling(new_epochs, correction_method="zscore")

    # Plot the image:
    new_epochs.plot_image(scalings={"ecog": 1}, show=False, units=dict(ecog='zscore'), combine="mean")
    if save_path is not None:
        file_name = Path(save_path, file_prefix
                         + "plot_image_{0}.png".format(roi))
        plt.savefig(file_name, transparent=True)
    plt.close()
    # Now computing the evoked responses:
    evk = {cond: None for cond in np.unique(y)}
    for cond in event_dict.keys():
        evk[cond] = new_epochs[event_dict[cond]].average()
    # Plotting all the electrodes on top of another:
    fig, axs = plt.subplots(len(event_dict.keys()))
    for ind, cond in enumerate(event_dict.keys()):
        evk[cond].plot(titles=cond, axes=axs[ind], scalings={"ecog": 1}, ylim={"ecog": [-10, 10]}, show=False)
        axs[ind].set_ylabel("zscore")
    plt.tight_layout()
    if save_path is not None:
        file_name = Path(save_path, file_prefix
                         + "plot_evoked_{0}.png".format(roi))
        fig.savefig(file_name, transparent=True)
        plt.close(fig)
    fig, ax = plt.subplots()
    # Plotting the comparison:
    mne.viz.plot_compare_evokeds(evk, combine='mean', axes=ax, show=False)
    ax.set_ylabel("zscore")
    if save_path is not None:
        file_name = Path(save_path, file_prefix
                         + "plot_compared_evoked_{0}.png".format(roi))
        fig.savefig(file_name, transparent=True)
        plt.close(fig)
    plt.close()

    return None


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


def compute_pseudotrials(x, y, groups=None, n_trials=5, pad_val=np.nan, permute_trials=True):
    """
    This function enables computation of pseudotrials. This consists in averaging n_trials together. The averaging
    of course happens separately for groups and y labels separately (i.e. only trials of matching groups are averaged).
    Additionally, the attribution to trial to a group of pseudotrial is randomized, such that we don't have the
    following trials being averaged together.
    :param x: (np array) contains the data to use for the decoding. The data must be of dimension: trials *
    channels * time points. The averaging will be done along the trials dimension
    :param y: (list or np array) decoding targets. Must be of the same dimension as the first dim of x
    :param groups: (list or np array) groups to which each trial belongs!
    :param n_trials: (int) number of trials to average together. So if you set that to 5, every 5 trials of the same
    y condition (and groups condition) will be averaged together
    :param pad_val: (int, float, nan...) what to pad the array with when doing the average. Say you want to average
    every 5 trials but you have 26. By setting to np.nan you will make up for the array to be of size 30 by padding with
    nan, having the last trial being picked being averaged with nan.
    :param permute_trials: (boolean) whether to shuffle the trials order before doing the averaging. If set to true,
    it averages the trials in the order they come in the arrays. But by setting it to true, the order is randomized,
    to avoid any sort of dependencies
    :return:
    new_x: (np.array) x data averaged together. So it will be of the size of x but with first dimension / n_trials
    new_y:(np.array) decoding targets matching with the new_x dimensions
    new_groups: (np.array) groups matching with the new_x dimensions
    """
    # Print the original shape of the data:
    print("-" * 40)
    print("Welcome to pseudotrials computations")
    print("Original data shape:")
    print("x: {0}".format(x.shape))
    print("y: {0}".format(y.shape))
    # Getting the unique labels and the counts:
    values, counts = np.unique(y, return_counts=True)
    [print("{0}: {1}".format(values[i], counts[i])) for i in range(len(values))]
    new_x = []
    new_y = []
    new_groups = []
    # Checking the size of the inputs to ensure that they match:
    if len(y) != x.shape[0]:
        raise Exception("The dimension of the targets labels doesn't match the size of the data matrix!")
    if groups is not None and len(groups) != x.shape[0]:
        raise Exception("The dimension of the groups labels doesn't match the size of the data matrix!")
    if groups is not None and len(groups) != len(y):
        raise Exception("The dimension of the targets labels doesn't match the size of the groups labels!")
    # Depending on whether or not we have groups, the computation of pseudotrials need to be accounting for it:
    if groups is not None:
        # Looping through each unique group:
        for group in set(groups):
            # Extracting only those trials and labels from this group:
            group_x, group_y = x[np.where(groups == group)[0]], y[np.where(groups == group)[0]]
            # Now, looping through each unique label from this subset of the data to perform the averaging:
            for label in set(group_y):
                # Extract the data:
                data = group_x[np.where(group_y == label)]
                if permute_trials:
                    data = np.take(data, np.random.permutation(data.shape[0]), axis=0)
                avg_x = block_reduce(data, block_size=tuple([n_trials, *[1] * len(data.shape[1:])]),
                                     func=np.nanmean, cval=pad_val)
                # Now generating the labels and group:
                new_x.append(avg_x)
                new_y += [label] * avg_x.shape[0]
                new_groups += [group] * avg_x.shape[0]
    else:
        # In case there are no groups, one level less to dig to:
        for label in set(y):
            # Get the data:
            data = x[np.where(y == label)]
            # Permute if needed:
            if permute_trials:
                data = np.take(data, np.random.permutation(data.shape[0]), axis=0)
            # Apply block reduce function:
            avg_x = block_reduce(data, block_size=tuple([n_trials, *[1] * len(data.shape[1:])]),
                                 func=np.nanmean, cval=pad_val)
            # Now generating the labels and group:
            new_x.append(avg_x)
            new_y += [label] * avg_x.shape[0]
        new_groups = None
    # Turning everything back to numpy arrays:
    new_x = np.concatenate(new_x, axis=0)
    new_y = np.array(new_y)
    if new_groups is not None:
        new_groups = np.array(new_groups)
    # Printing info about the pseudotrials resulting from the procedure:
    print("Pseudotrials data shape:")
    print("x: {0}".format(new_x.shape))
    print("y: {0}".format(new_y.shape))
    values, counts = np.unique(new_y, return_counts=True)
    [print("{0}: {1}".format(values[i], counts[i])) for i in range(len(values))]
    return new_x, new_y, new_groups


def pad_with_nan(array, shape):
    """
    This function pads a given array to be of the shape specified
    :param array: (np array) array to pad
    :param shape: (list of int or np array shape object) shape that the passed array must have
    :return: (np array) padded array to the size expected
    """
    pad_per_dim = []
    for dim_ind, dim_size in enumerate(array.shape):
        # Compare each axis to the desired shape:
        dim_pad = shape[dim_ind] - dim_size
        pad_per_dim.append([dim_pad // 2, dim_pad // 2 + dim_pad % 2])
    return np.pad(array, pad_per_dim,
                  mode='mean')


def slice_per(source, step, axis=0, overlapping=True):
    """
    This function takes a numpy array and splits it in sub arrays by taking n data points (specified as step).
    So say your array is of like so:
    1, 2, 3, 4, 5, 6,
    3, 4, 2, 2, 1, 3
    And you set axis to 1 and step to 2, it will return:
    array 1:
    1, 3, 5
    3, 2, 1
    array 2:
    2, 4, 6
    4, 2, 3
    :param source: (np array) source data to slice
    :param step: (int) steps to slice the data per
    :param axis: (int) axis along which to slice
    :param overlapping: (boolean) whether or not to do overlapping
    :return: (list of arrays) list of each slice of the array
    """
    if overlapping:
        return [source.take(indices=range(i, i + step), axis=axis) for i in range(source.shape[axis] - step)]
    else:
        return [source.take(indices=range(i, i + step), axis=axis) for i in
                range(0, source.shape[axis] - step, step)]


def equate_array_sizes(list_of_arrays):
    """
    This function equates arrays of different sizes. This function compute the size of the different arrays and find the
    one of highest size. This info is then passed to the pad  with nan function to pad the array
    :param list_of_arrays: (list of arrays) list of arrays to equate in size.
    :return:
    """
    # Get the size of all arrays in the list:
    shapes_list = np.array([array.shape for array in list_of_arrays])
    # Finding the max across all the lists:
    max_dim = np.max(shapes_list, axis=0)
    # Equating all the arrays to the same size:
    return [pad_with_nan(array, max_dim) for array in list_of_arrays]


def bin_time_win(data, sr, axis=-1, overlapping=False, bins_duration_ms=10, average=True):
    """
    This function bins data according to a specific bins duration. If average is True, that data will be binned
    in the last dimension every n samples, essentially adding a dimension that will then be stacked back onto the
    channels dimension (this is a bit of a weird thing to do when you have several channels, but with only one,
    it means your feature are basically over time as opposed to be over electrodes). If the average option is set to
    true, the data will be averaged in each bin, essentially reducing the time dimension
    1, 2, 1, 2, 1, 2, 1, 2 and you want 2ms bins, it will return this:
    2, 2, 2, 2
    1, 1, 1, 1
    :param data: (numpy array) data to bin
    :param sr: (int) sampling rate of the data
    :param axis: (int) axis along which to bin
    :param overlapping: (boolean) whether or not to do the binning overlapping
    :param bins_duration_ms: (int) duration of each bin in milliseconds
    :param average: (boolean) whether or not to average in each bin. If true, the data will be averaged any n ms (as
    set by bins_duration_ms), and  the overlapping will also apply
    :return: (np array): binned data
    """
    # Compute how many samples the time window corresponds to:
    n_samples = int(np.floor(bins_duration_ms * sr / 1000))
    if n_samples < 2:
        print("=" * 80)
        print("WARNING: The time window used for binning is less than two samples. "
              "Therefore, binning is not possible! Increase window length or don't bin")
        return
    if average:
        data_new = moving_average(data, n_samples, axis=-1, overlapping=overlapping)
    else:
        # Splitting the data:
        list_of_arrays = slice_per(
            data, n_samples, axis=axis, overlapping=overlapping)
        # Reshaping to flatten the second and third dim:
        list_of_arrays_2d = [array.reshape(
            (array.shape[0], -1)) for array in list_of_arrays]
        # Then, stacking these arrays along the second dimension, to keep things 3D:
        data_new = np.stack(list_of_arrays_2d, axis=2)

    return data_new


def file_name_generator(save_path, file_prefix, description, file_extension, data_type="ieeg"):
    """
    This function generates full file names according to the cogitate naming conventions:
    :param save_path: (pathlib path object or path string) root path to where the data should be saved
    :param file_prefix: (string) prfix of the file name
    :param description: (string) what some after teh prefix in the file name
    :param file_extension: (string) what comes after the description, if anything
    :param data_type: (string) data type the data are from
    :return: full_file_name (string) file path + file name
    """
    full_file_name = os.path.join(
        save_path, file_prefix + description + "_" + data_type + file_extension)

    return full_file_name


def get_cross_validation_data(cross_validation_iterator, x, y):
    """
    This function extracts the data for the train and test sets of each cross validation folds.
    :param cross_validation_iterator: (scikit learn cross validation iterators) See doc here:
    https://scikit-learn.org/stable/modules/cross_validation.html
    :param x: (numpy array) array containing the data to be used for the decoding and that will be split by the
    cross validation iteror
    :param y: (numpy array or list) list of the decoding targets that will be split in the same way as the x
    :return: x_train, y_train, x_test, y_test: (lists of arrays) holding the x and y of the train and test sets
    respectively
    """
    print("-" * 40)
    print("Welcome to get_cross_validation_data")
    # Preparing empty lists to hold train and test data of each folds:
    x_train, x_test, y_train, y_test = [], [], [], []
    # Extracting the splitted test from the y test:
    ctr = 0
    for train_index, test_index in cross_validation_iterator.split(x, y):
        x_train.append(x[train_index])
        y_train.append(y[train_index])
        x_test.append(x[test_index])
        y_test.append(y[test_index])
        # Print the sizes of the test and train sets:
        print("Fold {0} train set trial number={1}, test set trial number={2}".format(ctr, len(train_index),
                                                                                      len(test_index)))
        ctr = ctr + 1

    return x_train, y_train, x_test, y_test


def print_trials_counts(prefix, labels):
    """
    This function is a small utility to print trials counts
    :param labels: (list of strings) contains the targets to count
    :param prefix: (string) what to call the set we are printing the counts for
    :return:
    """
    print("{0} set trial number: {1}".format(prefix, len(labels)))
    targets_name, targets_counts = np.unique(labels, return_counts=True)
    print("{0} set trials counts:".format(prefix))
    [print("{0}: {1}".format(targets_name[i], targets_counts[i])) for i in range(len(targets_name))]
    return None


def temporal_generalization_decoding(clf, x, decoding_target, cross_validation_parameters, metric="accuracy",
                                     train_group=None, test_group=None,
                                     groups=None, n_pseudotrials=None, shuffle_labels=False, classifier_n_jobs=1,
                                     do_only_diag=False, average_scores=False, n_channel_subsample = None, verbose=False):
    """
    This function performs decoding in either a time resolved or temporal generalization fashion. This function can do
    so in a couple of different ways depending on the passed parameters.

    Option 1:
    Within group decoding:
        If train_group and test_group are both set to either None or both to the same string, the decoding will
        be performed within this group. The train and test_groups refers to a condition within which you want to train
        and test (so say task relevant and irrelevant). So if you set both to None, that mean that you want to
        train and test on all the trials. If you specify the same string in each, that means you want to train and test
        specifically within a given subset of your trials. For this option to work, you must set
        cross_validation_parameters["n_folds"] to not None and > 2. Otherwise you have no way to test your decoding
        accuracy.
    Cross task generalization decoding:
        If you pass two differrent groups under train_group and test_groups, then that means you want to train on a
        given group and test in the other. For this type of decoding, you must pass a list in groups of  the same
        length as decoding_target, identifying for each trial to which group it belongs to segregate them.
        When you use this function, you can skip cross_validation alltogether by setting
        cross_validation_parameters["n_folds"] to None, which mean training on all trials of the one condition and
        testing on all trials of the other. But if you do want to do cross fold validation, you can. And if you do
        want to do that, you have two other sub-options. In cross fold validation, the train set will be split in
        n_folds. You can then train on n-1 fold and test on all the trials of the other group to test your decoding.
        But you can also split the test set by the same amount as the training set if trial counts is something you
        are worried about
    :param clf: (scikit learn pipeline object) pipeline to be used for the decoding
    :param x: (np array) contains the data to be used for the decoding. The first dimension should represent the trials
    :param decoding_target: (list of strings) column of the epochs object meta data on which to perform the decoding
    :param cross_validation_parameters: (dict) parameters of the cross validation
        "n_folds": (int or None) how many folds to do the cross validation
        "split_generalization_set": (boolean) whether or not you want to split the test set as much as the train set
        in the case of cross task generalization decoding
    :param train_group: (string or None) groups on which to train the classifier. The string must match what is found
    in the groups list
    :param test_group: (string or None) groups on which to test the classifier. The string must match what is found
    in the groups list
    :param groups: (list) must be of same length as decoding_target, as it contains the group appartenance of each trial
    :param n_pseudotrials: (int or None) set to None if you do not want to use pseudotrials. If you use an int,
    then the trials will be averaged every n trials (so if = 2, the pseudotrials will be computed by gathering every
    two trials, and the
    :param shuffle_labels: (bool) whether or not to shuffle the labels before performing the decoding. This
    enables this function to be called many times to shuffle the labels and generate a null distribution.
    CAREFUL: this is not the same as do_test_label_shuffle. The latter consists of shuffling the labels of the test
    set after training to generate a null distribution. If you have set that to true, you should set the other to False
    because it wouldn't make any sense.
    :param classifier_n_jobs: (int) how many jobs to use to preprocessing the classifier in parallel
    :param do_only_diag: (boolean) whether to do only diagonal (time resolved) or temporal generalization decoding
    :param average_scores: (boolean) whether to average the decoding scores across folds
    :param verbose: (boolean) whether or not to print info to command line
    :return:
    """
    if verbose:
        print("-" * 40)
        print("Performing decoding")
    # Check the different inputs:
    if train_group is None and test_group is not None or train_group is not None and test_group is None:
        raise Exception("You have passed {0} as train condition, but {1} as test condition. This doesn't work! "
                        "\nThe train condition must either be both None or both strings!".
                        format(train_group, test_group))
    if isinstance(train_group, str) and groups is None:
        raise Exception("You have passed {0} as train and {1} as test groups, but you haven't passed a groups array to "
                        "\nidentify which trial belongs to which group".format(train_group, test_group))
    # First, specifiying the decoder:
    if do_only_diag is True:
        time_gen = SlidingEstimator(clf, n_jobs=classifier_n_jobs, scoring=metric,
                                    verbose="ERROR")
    else:
        time_gen = GeneralizingEstimator(clf, n_jobs=classifier_n_jobs, scoring=metric,
                                         verbose="ERROR")

    # randomly subsample the data (used for robustness checks)
    if n_channel_subsample is not None:
        x = x[:, np.random.choice(x.shape[1], n_channel_subsample, replace=False), :]
    # Shuffling the trials if needed (to generate null distributions for example):
    if shuffle_labels:
        decoding_target = decoding_target[np.random.permutation(len(decoding_target))]
    # Then, generating the pseudotrials if required:
    if n_pseudotrials is not None:
        x, decoding_target, groups = compute_pseudotrials(x, decoding_target, groups, n_trials=n_pseudotrials)

    # --------------------------------------------------------------------------------------------------------------
    # Option 1: train and test within the same condition
    # --------------------------------------------------------------------------------------------------------------
    if train_group == test_group:
        # Checking that the cross fold validation is set to true:
        if cross_validation_parameters["n_folds"] is None:
            raise Exception("If you are doing within task decoding, you must use cross fold validation to be able"
                            "\nto test your trained decoding on something!")
        # Get the classes:
        y = decoding_target
        # Setting the cross validation:
        if cross_validation_parameters["n_folds"] == "leave_one_out":
            n_folds = len(y)
        else:
            n_folds = cross_validation_parameters["n_folds"]
        # Prepare list to hold the results:
        scores = []
        # Creating cross val iterator:
        skf = StratifiedKFold(n_splits=n_folds)
        # Getting the indices of the test and train sets from cross folder validation:
        cv_iter = list(skf.split(x, y))
        # Performing the decoding:
        for ind, train_test_ind in enumerate(cv_iter):
            if verbose:
                print_trials_counts("train", y[train_test_ind[0]])
                print_trials_counts("test", y[train_test_ind[1]])
            # Initiate time:
            start = time.time()
            # Train on this split:
            time_gen.fit(X=x[train_test_ind[0]],
                         y=y[train_test_ind[0]])
            if verbose:
                print("Train time={:.2f}sec".format(time.time() - start))
            # Initiate time and CPU counters:
            start = time.time()
            # Test decoder within the same condition:
            scores.append(time_gen.score(X=x[train_test_ind[1]], y=y[train_test_ind[1]]))
            if verbose:
                print("Test time={:.2f}sec".format(time.time() - start))
        # Compute the coefficient regardless:
        try:
            time_gen.fit(x, y)
            coef = get_coef(time_gen, 'coef_', inverse_transform=True)
        except ValueError:
            coef = None
            print("WARNING: The coefficient could not be computed as the classifier was not linear")

        # Saving the scores in dict to keep things consistent between options
        scores = np.array(scores)
    # --------------------------------------------------------------------------------------------------------------
    # Option 2: Cross task generalization:
    # --------------------------------------------------------------------------------------------------------------
    else:
        # Prepare result of test:
        scores = []
        # Getting the data of the condition on which to perform the decoding:
        x_train_condition = x[np.where(groups == train_group)]
        # Get the targets:
        y_train_condition = decoding_target[np.where(groups == train_group)]

        # Get the test data:
        x_generalization = x[np.where(groups == test_group)]
        y_generalization = decoding_target[np.where(groups == test_group)]

        # In case of cross task generalization, there is the option of performing cross fold validation. If this is
        # expected, the data in the train group will be splitted in n_folds to test against the test set. The test
        # set can further be subdived into n folds or not:
        if cross_validation_parameters["n_folds"] is not None:
            # Create the kfold object:
            skf = StratifiedKFold(n_splits=cross_validation_parameters["n_folds"])
            # Get the labels and data of the different splits:
            x_train_condition_train, y_train_condition_train, x_train_condition_test, y_train_condition_test = \
                get_cross_validation_data(skf, x_train_condition, y_train_condition)

            # If the generalization set needs to be split:
            if cross_validation_parameters["split_generalization_set"]:
                _, _, x_generalization_sets, y_generalization_sets = \
                    get_cross_validation_data(
                        skf, x_generalization, y_generalization)

                # Performing the decoding:
                for ind, train_set in enumerate(x_train_condition_train):
                    if verbose:
                        print_trials_counts("train", y_train_condition_train[ind])
                        print_trials_counts("test", y_generalization_sets[ind])
                    # Train on this split:
                    start = time.time()
                    time_gen.fit(X=train_set,
                                 y=y_train_condition_train[ind])
                    if verbose:
                        print("Train time={:.2f}sec".format(time.time() - start))
                    # Test decoder generalization on other condition
                    start = time.time()
                    scores.append(time_gen.score(X=x_generalization_sets[ind],
                                                 y=y_generalization_sets[ind]))
                    if verbose:
                        print("Test time={:.2f}sec".format(time.time() - start))
            else:  # On each cross validation fold, testing the decoder on the entire generalization set:
                for ind, train_set in enumerate(x_train_condition_train):
                    print_trials_counts("train", y_train_condition_train[ind])
                    print_trials_counts("test", y_generalization)
                    # Train on this split:
                    start = time.time()
                    time_gen.fit(X=train_set, y=y_train_condition_train[ind])
                    if verbose:
                        print("Train time={:.2f}sec".format(time.time() - start))
                    start = time.time()
                    scores.append(time_gen.score(X=x_generalization,
                                                 y=y_generalization))
                    if verbose:
                        print("Test time={:.2f}sec".format(time.time() - start))
        else:  # If no cross fold validation is expected
            print_trials_counts("train", y_train_condition)
            print_trials_counts("test", y_generalization)
            # Train on this split:
            start = time.time()
            time_gen.fit(X=x_train_condition, y=y_train_condition)
            if verbose:
                print("Train time={:.2f}sec".format(time.time() - start))
            start = time.time()
            scores.append(time_gen.score(X=x_generalization,
                                         y=y_generalization))
            if verbose:
                print("Test time={:.2f}sec".format(time.time() - start))
        scores = np.array(scores)
        # Compute the coefficient regardless:
        try:
            time_gen.fit(x_train_condition, y_train_condition)
            coef = get_coef(time_gen, 'coef_', inverse_transform=True)
        except ValueError:
            coef = None
            print("WARNING: The coefficient could not be computed as the classifier was not linear")

    # Averaging the scores if necessary. If there were no cross fold validation or if only the first fold was tested
    # there will be only 1 matrix, so no need to average:
    if average_scores and cross_validation_parameters["n_folds"] is not None:
        scores = np.mean(scores, axis=0)

    return scores, coef


# compute variance-corrected ttests for model comparison
# see: Statistical comparison of models using grid search, https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = stats.t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val
