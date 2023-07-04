from mne.time_frequency import psd_multitaper
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def viz_psd(raw):
    # Compute averaged power
    psds, freqs = psd_multitaper(raw, fmin = 1, fmax = 40, picks=['eeg'])
    psds = np.sum(psds, axis = 1)
    psds = 10. * np.log10(psds)
    # Show power spectral density plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    raw.plot_psd(picks = ["eeg"], 
                  fmin = 1, fmax = 40, 
                  ax=ax[0])
    # Normalize (z-score) channel-specific average power values 
    psd = {}
    psd_zscore = zscore(psds)
    for i in range(len(psd_zscore)):
        psd["EEG%03d"%(i+1)] = psd_zscore[i]
    # Plot chennels ordered by power
    ax[1].bar(sorted(psd, key=psd.get, reverse = True), sorted(psd.values(), reverse = True), width = 0.5)
    labels = sorted(psd, key=psd.get, reverse = True)
    ax[1].set_xticklabels(labels, rotation=90)
    ax[1].annotate("Average power: %.2e dB"%(np.average(psds)), (27, np.max(psd_zscore)*0.9), fontsize = 'x-large')
    return fig