"""
Port of Janca, 2014 IED detection algorithm
https://link.springer.com/article/10.1007/s10548-014-0379-1

@author: Simon Henin
simon.henin@nyulangone.org
"""
# %% 
from scipy import signal
import numpy as np
from mne.filter import resample
import pandas as pd


# %%
def local_maxima_detection(envelope, prah_int, fs, polyspike_union_time=0.12):
    marker1 = np.zeros_like(envelope);
    marker1[np.where(envelope > prah_int)[0]] = 1;  # crossing of high threshold

    #
    point1 = np.where(np.diff(np.concatenate((np.array((0,)), marker1), 0)) > 0)[0]  # strat crossing
    point2 = np.where(np.diff(np.concatenate((marker1, np.array((0,))), 0)) < 0)[0]  # end crossing
    point = np.stack((point1, point2), axis=1)

    # switch ti_switch
    #     case 2
    #         envelope=abs(d_decim);
    # end

    marker1 = np.zeros_like(envelope)  # false(size(envelope));
    for k in range(point.shape[0]):  # 1:size(point,1)
        # detection of local maxima in section which crossed threshold curve
        if (point[k, 1] - point[k, 0] > 2):
            seg = envelope[point[k, 0]:point[k, 1] + 1]
            seg_s = np.diff(seg);
            seg_s = np.sign(seg_s);
            seg_s = np.where(np.diff(np.concatenate((np.array((0,)), seg_s))) < 0)[
                0]  # positions of local maxima in the section

            marker1[point[k, 0] + seg_s - 1] = 1
        elif (point[k, 1] - point[k, 1] <= 2):
            seg = envelope[point[k, 0]:point[k, 1] + 1]
            s_max = np.argmax(seg)  # positions of local maxima in the section
            marker1[point[k, 0] + s_max - 1] = 1

    # union of section, where local maxima are close together <(1/f_low + 0.02 sec.)~ 120 ms
    pointer = np.where(marker1 == 1)[0]  # index of local maxima
    state_previous = False
    for k in range(len(pointer)):
        if np.ceil(pointer[k] + polyspike_union_time * fs) > np.size(marker1, 0):
            seg = marker1[pointer[k] + 1:-1];
        else:
            seg = marker1[pointer[k] + 1:int(np.ceil(pointer[k] + polyspike_union_time * fs) + 1)];

        if state_previous:
            if np.sum(seg) > 0:
                state_previous = True;
            else:
                state_previous = False;
                marker1[start:pointer[k]] = 1;
        else:
            if np.sum(seg) > 0:
                state_previous = True
                start = pointer[k]

    # finding of the highes maxima of the section with local maxima
    point1 = np.where(np.diff(np.concatenate((np.array((0,)), marker1), 0)) > 0)[0]  # strat crossing
    point2 = np.where(np.diff(np.concatenate((marker1, np.array((0,))), 0)) < 0)[0]  # end crossing
    point = np.stack((point1, point2), axis=1)

    # local maxima with gradient in souroundings
    for k in range(np.size(point, 0)):
        if point[k, 1] - point[k, 0] > 1:
            lokal_max = pointer[(pointer >= point[k, 0]) & (pointer <= point[k, 1])];  # index of local maxima
            marker1[point[k, 0]:point[k, 1]] = 0;
            lokal_max_val = envelope[lokal_max];  # envelope magnitude in local maxima
            lokal_max_poz = np.where(
                np.diff(np.sign(np.diff(np.concatenate((np.array((0,)), lokal_max_val, np.array((0,)))))) < 0) > 0)[0]
            marker1[lokal_max[lokal_max_poz]] = 1

    return marker1


def detection_union(marker1, envelope, union_samples):
    # do the union
    if np.mod(union_samples, 2) == 0:
        union_samples = union_samples + 1

    MASK = np.ones((union_samples,))
    marker1 = (np.convolve(marker1, MASK, 'same') > 0);  # dilatation
    marker1 = np.logical_not(np.convolve(np.logical_not(marker1), MASK, 'same'))  # erosion

    marker2 = np.zeros_like(marker1)  # false(size(marker1));
    point1 = np.where(np.diff(np.concatenate((np.array((0,)), marker1), 0)) > 0)[0]  # strat crossing
    point2 = np.where(np.diff(np.concatenate((marker1, np.array((0,))), 0)) < 0)[0]  # end crossing
    point = np.stack((point1, point2), axis=1)

    for i in range(np.size(point, 0)):
        maxp = np.argmax(envelope[point[i, 0]:point[i, 1] + 1])
        marker2[point[i, 0] + maxp] = 1

    return marker2


def one_channel_detect(envelope, fs, k1=3.65, k2=3.65, k3=0., winlength=5., overlap=4., polyspike_union_time=0.12):
    """
    IED detection on single channel:
        envelope: hilbert envelope of the signal to analyze
        fs: sampling frequency
        k1: threshold value for obvious spike decision ('-k1 3.65' DEFAULT)
        k2: defines ambiguous spike treshold. Ambiguous 
            spike is accepted, when simultaneous obvious detection is in other 
            channel k1 >= k2 (k1 in DEFAULT)
        k3: decrease the threshold value (0 in DEFAULT) k1*(mode+median)-k3*(mean-mode);
        winlength: size of segment in seconds around spike for background
                        definition (5 seconds DEFAULT)
        overlap: overlap of segment in seconds 
                        (4 seconds DEFAULT)
        polyspike union time: spike in same channel nearest then time will
            be united

        returns:
            markers_high - timecourse of unambigious spikes (1=spike, 0=nospike) (based on k1 threshold)
            markers_low - ambigioius spikes (controlled by k2 threshold)
    """

    winsize = int(winlength * fs)
    noverlap = int(overlap * fs)
    index = np.arange(0, len(envelope) - winsize, winsize - noverlap, dtype=int)
    phat = []
    for ind in index:
        seg_ = envelope[ind:ind + winsize]
        phat.append([np.mean(np.log(seg_)), np.std(np.log(seg_))])
    phat = np.asarray(phat)

    r = envelope.shape[0] / len(index);
    n_average = winsize / fs;

    if round(n_average * fs / r) > 1:
        phat[:, 0] = signal.filtfilt(np.ones((int(np.round(n_average * fs / r)),)) / (np.round(n_average * fs / r)), 1,
                                     phat[:, 0])
        phat[:, 1] = signal.filtfilt(np.ones((int(np.round(n_average * fs / r)),)) / (np.round(n_average * fs / r)), 1,
                                     phat[:, 1])

    # % interpolation of thresholds value to threshold curve (like background)
    x = np.arange(index[0] + np.round(winsize / 2), index[-1] + np.round(winsize / 2))
    if phat.shape[0] > 1:
        phat_int1 = np.interp(x, index + np.round(winsize / 2), phat[:, 0]);
        phat_int2 = np.interp(x, index + np.round(winsize / 2), phat[:, 1]);
    phat_int = np.stack((phat_int1, phat_int2), axis=1)
    phat_int = np.concatenate((np.ones((int(np.floor(winsize / 2)), 2)) * phat_int[0, :], phat_int, np.ones(
        (int(np.size(envelope, 0) - (np.size(phat_int, 0) + np.floor(winsize / 2))), 2)) * phat_int[-1, :]));

    lognormal_mode = np.exp(phat_int[:, 0] - phat_int[:, 1] ** 2);
    lognormal_median = np.exp(phat_int[:, 0]);
    lognormal_mean = np.exp(phat_int[:, 0] + (phat_int[:, 1] ** 2) / 2);

    prah_int = np.zeros_like(phat_int)
    prah_int[:, 0] = k1 * (lognormal_mode + lognormal_median) - k3 * (lognormal_mean - lognormal_mode)
    if k2 != k1:
        prah_int[:, 1] = k2 * (lognormal_mode + lognormal_median) - k3 * (lognormal_mean - lognormal_mode)

    markers_high = local_maxima_detection(envelope, prah_int[:, 0], fs)
    markers_high = detection_union(markers_high, envelope, int(polyspike_union_time * fs))

    if (k2 != k1):
        markers_low = local_maxima_detection(envelope, prah_int[:, 1], fs, polyspike_union_time)
        markers_low = detection_union(markers_low, envelope, polyspike_union_time * fs)
    else:
        markers_low = markers_high

    # first and last second is not analyzed (filter time response etc.) -------
    markers_high[np.arange(fs, dtype=int)] = 0;
    markers_high[np.arange(len(markers_high) - fs, len(markers_high), dtype=int)] = 0
    markers_low[np.arange(fs, dtype=int)] = 0;
    markers_low[np.arange(len(markers_high) - fs, len(markers_high), dtype=int)] = 0

    return markers_high, markers_low


def Janca_IED_Detection(data, fs=None, k1=3.65, k2=3.65, k3=0., winlength=5., overlap=4., polyspike_union_time=0.12,
                        downsample_fs=200):
    """
    main wrapper function for IED detection
    """
    ieds = pd.DataFrame({'time': [], 'chan': []})

    decim = None
    if downsample_fs is not None:
        decim = fs / downsample_fs
        fs = downsample_fs

    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    nchans = np.size(data, 1)
    for ch in range(nchans):
        d = data[:, ch]
        if decim is not None:
            d = resample(d, down=decim)
        envelope = np.abs(signal.hilbert(d))
        spikes = one_channel_detect(envelope, fs, k1=k1, k2=k2, k3=k3, winlength=winlength, overlap=overlap,
                                    polyspike_union_time=polyspike_union_time)

        # store ied times in a dataframe
        ieds_ = np.where(spikes[0])[0] / fs
        if len(ieds_) > 0:
            df = pd.DataFrame({'time': ieds_, 'chan': np.tile(ch, len(ieds_))})
            ieds = pd.concat((ieds, df), ignore_index=True)

    return ieds
