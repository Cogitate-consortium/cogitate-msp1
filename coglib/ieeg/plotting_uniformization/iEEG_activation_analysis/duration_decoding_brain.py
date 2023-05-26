import pandas as pd
from pathlib import Path
import config
import os
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from general_utilities import epochs_loader
from plotters import plot_rasters
from scipy.ndimage import uniform_filter1d


def sort_metadata(metadata, conditions_order):
    """

    """
    dfs = []
    for condition in conditions_order.keys():
        for cond in conditions_order[condition]:
            dfs.append(metadata.loc[metadata[condition] == cond])
        metadata = pd.concat(dfs)
        dfs = []
    return metadata


param = config.param
# Set up the colors:
accuracy_cmap = plt.get_cmap("Reds")
norm = mpl.colors.Normalize(vmin=0, vmax=1)
accuracy_scalar_map = cm.ScalarMappable(norm=norm, cmap=accuracy_cmap)

bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
save_dir = "/hpc/users/alexander.lepauvre/plotting_test/activation_analysis/high_gamma_duration_decoding"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Set the path to the files:
face_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
               "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_face_ti_500ms/desbadcharej_notfil_lapref/sub" \
               "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
object_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_object_ti_500ms/desbadcharej_notfil_lapref/sub" \
                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
letter_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_letter_ti_500ms/desbadcharej_notfil_lapref/sub" \
                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
false_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_false_ti_500ms/desbadcharej_notfil_lapref/sub" \
                "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
face_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
               "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_face_tr_500ms/desbadcharej_notfil_lapref/sub" \
               "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
object_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_object_tr_500ms/desbadcharej_notfil_lapref/sub" \
                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
letter_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_letter_tr_500ms/desbadcharej_notfil_lapref/sub" \
                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
false_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_false_tr_500ms/desbadcharej_notfil_lapref/sub" \
                "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
# Load the results:
face_ti_results = pd.read_csv(face_ti_file)
object_ti_results = pd.read_csv(object_ti_file)
letter_ti_results = pd.read_csv(letter_ti_file)
false_ti_results = pd.read_csv(false_ti_file)
face_tr_results = pd.read_csv(face_tr_file)
object_tr_results = pd.read_csv(object_tr_file)
letter_tr_results = pd.read_csv(letter_tr_file)
false_tr_results = pd.read_csv(false_tr_file)

# Extract the list of channels:
channels = face_ti_results["channel"].to_list()
duration_decoding_channels = []
# Loop through each channel:
for ch in channels:
    # Locate the channel results in each table:
    face_res_ti = face_ti_results.loc[face_ti_results["channel"] == ch, "p-value"].item()
    object_res_ti = object_ti_results.loc[object_ti_results["channel"] == ch, "p-value"].item()
    letter_res_ti = letter_ti_results.loc[letter_ti_results["channel"] == ch, "p-value"].item()
    false_res_ti = false_ti_results.loc[false_ti_results["channel"] == ch, "p-value"].item()
    face_res_tr = face_tr_results.loc[face_tr_results["channel"] == ch, "p-value"].item()
    object_res_tr = object_tr_results.loc[object_tr_results["channel"] == ch, "p-value"].item()
    letter_res_tr = letter_tr_results.loc[letter_tr_results["channel"] == ch, "p-value"].item()
    false_res_tr = false_tr_results.loc[false_tr_results["channel"] == ch, "p-value"].item()

    # Make the conjunction:
    if face_res_ti < 0.05 and face_res_tr < 0.05:
        face_decoding = True
    else:
        face_decoding = False
    if object_res_ti < 0.05 and object_res_tr < 0.05:
        object_decoding = True
    else:
        object_decoding = False
        object_decoding_accuracy = 0
    if letter_res_ti < 0.05 and letter_res_tr < 0.05:
        letter_decoding = True
    else:
        letter_decoding = False
    if false_res_ti < 0.05 and false_res_tr < 0.05:
        false_decoding = True
    else:
        false_decoding = False

    # Check whether there are any conditions for which we have decoding
    if not any([face_decoding, object_decoding, letter_decoding, false_decoding]):
        continue
    # Add to the table:
    duration_decoding_channels.append(ch)

# Load the data:
subs = list(set([pick.split("-")[0] for pick in duration_decoding_channels]))
epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                   "ses-V1", "ieeg", "epoching", "high_gamma", "desbadcharej_notfil_lapref"))
epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
epochs = \
    epochs_loader(subs, epo_dir, epo_file, duration_decoding_channels, [-0.3, 2.0], "V1",
                  conditions=["stimulus onset/Relevant non-target", "stimulus onset/Irrelevant"],
                  filtering_parameters=None,
                  baseline_window=(None, -0.05))

smooth_time_ms = 50
sfreq = 512
# Convert to samples:
smooth_samp = int(smooth_time_ms * (sfreq / 1000))

for subject in epochs.keys():
    epo = epochs[subject].copy()
    metadata = epo.metadata
    metadata["order"] = range(0, len(metadata))
    metadata = sort_metadata(metadata, {
        "task_relevance": ["Relevant non-target", "Irrelevant"],
        "duration": ["1500ms", "1000ms", "500ms"],
        "category": ["face", "object", "letter", "false"]
    })
    conds = metadata["category"].to_list()
    t0 = epo.times[0]
    tend = epo.times[-1]
    for ch in epo.ch_names:
        # Create the file name:
        file_name = Path(save_dir, "sub-{}_ch-{}.png".format(subject, ch))
        data = np.squeeze(epo.get_data(picks=ch))
        data = data[metadata["order"], :]
        data = uniform_filter1d(data, smooth_samp, axis=-1)
        plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=None, midpoint=1.0, transparency=1.0,
                     xlabel="Time (s)", ylabel="Trials", cbar_label="HGP (norm.)", filename=file_name,
                     vlines=[0, 0.5, 1.0, 1.5], title=None, square_fig=False, conditions=conds,
                     cond_order=["face", "object", "letter", "false"], crop_cbar=False)
        plt.close()
