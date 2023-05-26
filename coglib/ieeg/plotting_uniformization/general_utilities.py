import numpy as np
import scipy
import mne
import pandas as pd
from pathlib import Path
from mne.baseline import rescale
from scipy.ndimage import uniform_filter1d


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


def epochs_loader(subjects, epochs_dir, epochs_file, picks, crop_time, ses, conditions=None,
                  baseline_mode="ratio", filtering_parameters=None, baseline_window=None):
    """
    This function loads the subjects epochs picking a specific set of electrodes and extracting conditions of interest
    :param subjects:
    :param epochs_dir:
    :param epochs_file:
    :param picks:
    :param conditions:
    :return:
    """
    if baseline_window is None:
        baseline_window = [-0.35, -0.05]
    epochs = {subject: None for subject in subjects}
    for subject in subjects:
        print("Loading sub-{} data".format(subject))
        sub_epo_dir = epochs_dir.format(subject)
        sub_epo_file = epochs_file.format(subject, ses)
        epo = mne.read_epochs(str(Path(sub_epo_dir, sub_epo_file)), verbose="error", preload=True)
        if conditions is not None:
            epo = epo[conditions]
        # Filter the data according to the filtering parameters:
        if filtering_parameters is not None:
            freqs = np.arange(filtering_parameters["freq_range"][0], filtering_parameters["freq_range"][1],
                              filtering_parameters["step"])
            n_cycles = freqs / filtering_parameters["n_cycle_denom"]
            if filtering_parameters["method"] == "multitaper":
                tfr = mne.time_frequency.tfr_multitaper(
                    epo,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=False,
                    picks=epo.ch_names,
                    time_bandwidth=filtering_parameters["time_bandwidth"],
                    verbose=True)
            elif filtering_parameters["method"] == "wavelet":
                tfr = mne.time_frequency.tfr_morlet(
                    epo,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=False,
                    picks=epo.ch_names,
                    output="power",
                    verbose=True)

            # Do baseline correction:
            if filtering_parameters["baseline_mode"] is not None:
                tfr.apply_baseline(filtering_parameters["baseline_win"], mode=filtering_parameters["baseline_mode"])
            # Extract the data in the frequency band:
            data = np.mean(tfr.data, axis=-2)
            # Adjust the sfreq:
            info = epo.info
            info["sfreq"] = data.shape[-1] / (epo.times[-1] - epo.times[0])
            # Shove it back in the mne epochs object:
            epo = mne.EpochsArray(data, epo.info, tmin=epo.times[0], events=epo.events,
                                  event_id=epo.event_id, on_missing="warning", metadata=epo.metadata)
        else:
            epo.apply_function(rescale, times=epo.times, baseline=(baseline_window[0], baseline_window[1]),
                               mode=baseline_mode)
        epo.crop(crop_time[0], crop_time[1])
        # There is a bug from MNE such that the montage coordinate space is ignore and considered to be head always.
        # Setting the montage to the correct coordinates frame:
        from mne.io.constants import FIFF
        for d in epo.info["dig"]:
            d['coord_frame'] = FIFF.FIFFV_COORD_MRI
        # To spare memory as much as possible, selecting only the relevant channels:
        sub_picks = [pick.split("-")[1] for pick in picks if pick.split("-")[0] == subject]
        if len(sub_picks) > 0:
            epo.pick(sub_picks)
        else:
            del epochs[subject]
            continue
        epochs[subject] = epo

    return epochs


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
    err = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, err


def sum_of_sq(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)


def compute_gfp(data, times):
    """
    This function computes global field power
    :param data:
    :param times:
    :return:
    """
    gfp = np.sum(data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    ci_low, ci_up = mne.stats.bootstrap_confidence_interval(data, random_state=0,
                                                            stat_fun=sum_of_sq)
    return gfp, ci_low, ci_up


def get_channels_labels(epochs, subject, fs_dir, aseg="aparc+aseg"):
    """

    :param epochs:
    :param subject:
    :param fs_dir:
    :param aseg:
    :return:
    """
    # Get the labels of each channel according to the atlas:
    channels_labels = pd.DataFrame()
    # Get the labels of these channels:
    labels, _ = mne.get_montage_volume_labels(
        epochs.get_montage(), "sub-" + subject,
        subjects_dir=fs_dir, aseg=aseg)
    for ind, channel in enumerate(labels.keys()):
        channels_labels = channels_labels.append(
            pd.DataFrame({"subject": subject, "channel": channel, "region": "/".join(labels[channel])}, index=[ind]))

    return channels_labels


def baseline_correction(data, times, baseline=(None, 0)):
    """
    This functions corrects the single trials by the baseline. This is an adaption of mne baseline.rescale for iEEG.
    Instead of computed single trials baseline to apply to the rest of the time series, computing the baseline average
    across trials.
    :param data:
    :param times:
    :param baseline:
    :return:
    """
    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        imin = np.where(times >= bmin)[0]
        if len(imin) == 0:
            raise ValueError('bmin is too large (%s), it exceeds the largest '
                             'time value' % (bmin,))
        imin = int(imin[0])
    if bmax is None:
        imax = len(times)
    else:
        imax = np.where(times <= bmax)[0]
        if len(imax) == 0:
            raise ValueError('bmax is too small (%s), it is smaller than the '
                             'smallest time value' % (bmax,))
        imax = int(imax[-1]) + 1
    if imin >= imax:
        raise ValueError('Bad rescaling slice (%s:%s) from time values %s, %s'
                         % (imin, imax, bmin, bmax))
    # Compute the mean across trials:
    mean = np.mean(data[..., imin:imax], axis=(0, -1))
    data /= mean

    return data


def get_roi_channels(epochs, roi_labels, bids_root, aseg="aparc.a2009s+aseg"):
    """
    This function extracts the channels from a given ROI
    """
    epo_new = {}
    for sub in epochs.keys():
        # Get the labels of this subject's channels:
        subject_labels = get_channels_labels(epochs[sub], sub,
                                             Path(bids_root, "derivatives", "fs"),
                                             aseg=aseg)
        # Get the channels that are within the ROI:
        picks = []
        for ind, row in subject_labels.iterrows():
            ch_lbls = row["region"].split("/")
            for lbl in ch_lbls:
                if lbl in roi_labels:
                    picks.append(row["channel"])
                    continue
        if len(picks) == 0:
            continue
        epo_new[sub] = epochs[sub].pick(list(set(picks)))

    return epo_new


def load_fsaverage_coord(bids_root, subjects_list, ses='V1', laplace_reloc=True):
    """

    """
    fsavg_coord = []
    # Loop through each subject
    for sub in subjects_list:
        if laplace_reloc:
            fsaverage_root = Path(bids_root, 'derivatives', 'preprocessing', 'sub-' + sub, 'ses-' + ses,
                                  'ieeg', 'laplace_reference', 'broadband', 'desbadcharej_notfil_lapref',
                                  'sub-{}_ses-{}_space-fsaverage_electrodes.tsv'.format(sub, ses))
            try:
                sub_coord = pd.read_csv(fsaverage_root, sep='\t')
            except FileNotFoundError:
                fsaverage_root = Path(bids_root, 'sub-' + sub, 'ses-' + ses,
                                      'ieeg', 'sub-{}_ses-{}_space-fsaverage_electrodes.tsv'.format(sub, ses))
                sub_coord = pd.read_csv(fsaverage_root, sep='\t')
        else:
            fsaverage_root = Path(bids_root, 'sub-' + sub, 'ses-' + ses,
                                  'ieeg', 'sub-{}_ses-{}_space-fsaverage_electrodes.tsv'.format(sub, ses))
            sub_coord = pd.read_csv(fsaverage_root, sep='\t')

        sub_coord['name'] = ['-'.join([sub, ch]) for ch in sub_coord['name'].to_list()]
        fsavg_coord.append(sub_coord)

    return pd.concat(fsavg_coord)


def get_ch_rois(bids_root, subjects_list, ses='V1', laplace_reloc=True, aparc="aparc.a2009s+aseg"):
    """

    """

    ch_rois = []
    # Loop through each subject
    for sub in subjects_list:
        if laplace_reloc:
            rois_root = Path(bids_root, 'derivatives', 'preprocessing', 'sub-' + sub, 'ses-' + ses,
                             'ieeg', 'atlas_mapping', 'raw', 'desbadcharej_notfil_lapref',
                             'sub-{}_ses-{}_task-Dur_desc-elecmapping_{}_ieeg.csv'.format(sub, ses, aparc))
            try:
                sub_rois = pd.read_csv(rois_root)
            except FileNotFoundError:
                rois_root = Path(bids_root, 'derivatives', 'preprocessing', 'sub-' + sub, 'ses-' + ses,
                                 'ieeg', 'atlas_mapping', 'raw', 'raw',
                                 'sub-{}_ses-{}_task-Dur_desc-elecmapping_{}_ieeg.csv'.format(sub, ses, aparc))
                sub_rois = pd.read_csv(rois_root)
        else:
            rois_root = Path(bids_root, 'derivatives', 'preprocessing', 'sub-' + sub, 'ses-' + ses,
                             'ieeg', 'atlas_mapping', 'raw', 'raw',
                             'sub-{}_ses-{}_task-Dur_desc-elecmapping_{}_ieeg.csv'.format(sub, ses, aparc))
            sub_rois = pd.read_csv(rois_root)
        # Append subject name to the channel name:
        sub_rois['channel'] = ['-'.join([sub, ch]) for ch in sub_rois['channel'].to_list()]
        # Extracting only the cortical labels:
        sub_rois["region"] = [[region for region in regions.split("/") if "ctx" in region]
                              if regions == regions else [] for regions in sub_rois["region"].to_list()]
        sub_rois["region"] = [region[0].replace("ctx_lh_", '').replace('ctx_rh_', '') if len(region) > 0 else []
                              for region in sub_rois["region"].to_list()]

        ch_rois.append(sub_rois)

    return pd.concat(ch_rois)


def corrected_sem(data, levels):
    """
    Compute the coousineau morey corrected standard error for time series data. The data should be in the format
    conditions x subjects x time

    """
    if not isinstance(levels, np.ndarray):
        if isinstance(levels, int):
            levels = np.array([levels])
        else:
            levels = np.array(levels)
    if levels.shape[0] == 1 and levels[0] == 1:
        raise Exception("The corrected  SEM is only applicable to within subject design with several factors!")
    if isinstance(data, list):
        # Equate the trials if there were inequal numbers for participants that didn't finish the
        # experiment:
        if not all([data[0].shape[0] == mat.shape[0] for mat in data]):
            # Find the smallest set:
            n_trials = min([mat.shape[0] for mat in data])
            # Randomly sample N:
            data = np.array([mat[np.random.choice(np.arange(mat.shape[0]), n_trials)] for mat in data])
        else:
            data = np.array(data)

    try:
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
    except AttributeError:
        print("A")
    # Compute the group average per time point:
    group_mean = np.mean(data, axis=(0, 1))
    # Compute the within group mean:
    subject_mean = np.mean(data, axis=0)
    # Compute the correction factor:
    corr_fact = levels.prod() / (levels.prod() - 1)
    corrected_se = []
    for cond in range(data.shape[0]):
        # Normalize data by removing the within subject average to that condition to each subject's data and adding back
        # the grand mean:
        norm_data = data[cond, :, :] - subject_mean + group_mean
        # Compute the denominator, i.e. the variance corrected by the Cousineau factor:
        denom = (np.sqrt(np.std(norm_data, axis=0) * corr_fact))
        # Compute the denominator
        numer = np.sqrt(data.shape[1])
        corrected_se.append(denom/numer)
    return corrected_se
