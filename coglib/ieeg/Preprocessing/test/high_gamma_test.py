from general_helper_functions.pathHelperFunctions import find_files
from Preprocessing.PreprocessingHelperFunctions import frequency_bands_computations
import mne
import matplotlib.pyplot as plt
import os
import numpy as np

plt.ioff()


def run(file_root=None, save_root=None, njobs=1):
    # Loading the data:
    data_file = find_files(file_root,
                           naming_pattern="*-raw", extension=".fif")
    raw = mne.io.read_raw_fif(data_file[-1],
                              verbose='error', preload=True)

    # Computing the high gamma in filter banks:
    raw_filtbank_norm = frequency_bands_computations(raw.copy(), frequency_range=[70, 150], njobs=njobs,
                                                     bands_width=10,
                                                     channel_types=None, method="filter_bank",
                                                     do_baseline_normalization=True)
    # Plotting the psd:
    raw_filtbank_norm.plot_psd(show=False)
    # Saving the fig:
    plt.savefig(os.path.join(save_root, "PSD_filtbank_norm.png"))
    plt.close()

    # Computing the high gamma in filter banks without normalization:
    raw_filtbank_nonorm = frequency_bands_computations(raw.copy(), frequency_range=[70, 150], njobs=njobs,
                                                       bands_width=10,
                                                       channel_types=None, method="filter_bank",
                                                       do_baseline_normalization=False)
    # Plotting the psd:
    raw_filtbank_nonorm.plot_psd(show=False)
    # Saving the fig:
    plt.savefig(os.path.join(save_root, "PSD_filtbank_nonorm.png"))
    plt.close()

    # Computing the high gamma in band_pass:
    raw_banddpass_norm = frequency_bands_computations(raw.copy(), frequency_range=[70, 150], njobs=njobs,
                                                      bands_width=10,
                                                      channel_types=None, method="band_pass_filter",
                                                      do_baseline_normalization=True)
    # Plotting the psd:
    raw_banddpass_norm.plot_psd(show=False)
    # Saving the fig:
    plt.savefig(os.path.join(save_root, "PSD_banddpass_norm.png"))
    plt.close()

    # Computing the high gamma in band_pass without baseline normalization:
    raw_banddpass_nonorm = frequency_bands_computations(raw.copy(), frequency_range=[70, 150], njobs=njobs,
                                                        bands_width=10,
                                                        channel_types=None, method="band_pass_filter",
                                                        do_baseline_normalization=False)
    # Plotting the psd:
    raw_banddpass_nonorm.plot_psd(show=False)
    # Saving the fig:
    plt.savefig(os.path.join(save_root, "PSD_banddpass_nonorm.png"))
    plt.close()


if __name__ == "__main__":
    run(
        file_root="/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/preprocessing/sub-CF109/ses-V1/ieeg/manual_bad_channels_rejection/broadband/manbadcharej_notfil_manbadcharej_car_manbadcharej",
        save_root="/home/alexander.lepauvre/Desktop/temp",
        njobs=8)
