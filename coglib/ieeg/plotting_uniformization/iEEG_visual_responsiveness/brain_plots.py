import mne
import os
import json
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from mne.baseline import rescale
import numpy as np
import pandas as pd
from pathlib import Path
from mne import get_montage_volume_labels
from general_utilities import load_fsaverage_coord, epochs_loader, get_channels_labels, baseline_correction
from ecog_plotters import plot_brain
import config

param = config.param
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "visual_responsiveness"
sub = "super"
ses = "V1"
data_type = "ieeg"
results_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "results")
figures_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "figure")
category_order = ["face", "object", "letter", "false"]
duration_order = ["500ms", "1000ms", "1500ms"]
vlines = [0, 0.5, 1.0, 1.5]
ylim = None
crop_time = [-0.3, 2.0]


def get_channels_rois(epochs, bids_root, aseg="aparc.a2009s+aseg"):
    """

    """
    channels_rois = []
    # Loop through each subject:
    for sub in epochs.keys():
        # Get the labels of this subject's channels:
        labels = get_montage_volume_labels(epochs[sub].get_montage(), "sub-" + sub,
                                           Path(bids_root, "derivatives", "fs"),
                                           aseg=aseg)
        # Extract one label per channel:
        ch_rois = {}
        for channel in labels[0].keys():
            lbl = [l for l in labels[0][channel] if
                   not "White" in l and not "WM" in l and not l == "Unknown" and not "unknown" in l]
            # Remove non cortical labels:
            lbl = [l for l in lbl if "ctx" in l]
            if len(lbl) == 0:
                continue
            ch_rois["-".join([sub, channel])] = lbl[0].replace("ctx_lh_", '').replace('ctx_rh_', '')
        # Convert to a dataframe:
        channels_rois.append(pd.DataFrame.from_dict(ch_rois, orient="index", columns=["label"]))
    # Concatenate all the channels:
    return pd.concat(channels_rois)


def brain_plots_handler(subjects, folder, save_root="", cond_to_plot="both"):
    """

    :param subjects:
    :param folder:
    :param save_root:
    :param cond_to_plot:
    :return:
    """
    # ================================================================================================
    # Load the data:
    results_path = Path(results_root, folder)
    subdirs = [x for x in results_path.iterdir() if x.is_dir()]
    confg_files = []
    for file in glob.glob(str(Path(subdirs[0], '*.json'))):
        confg_files.append(file)
    results_files = []
    for file in glob.glob(str(Path(subdirs[0], '*all_results.csv'))):
        results_files.append(file)
    vis_resp_results = pd.read_csv(results_files[0])
    with open(confg_files[0], 'r') as f:
        vis_resp_param = json.load(f)
    if subjects is None:
        subjects = list(vis_resp_results["subject"].unique())
    # Get the channels localization:
    fsaverage_coord = load_fsaverage_coord(bids_root, subjects, ses='V1', laplace_reloc=True)

    # ================================================================================================
    # Plot the channels latencies and activation strength:
    # Extract the significant channels:
    vis_resp_results_sig = vis_resp_results.loc[vis_resp_results["reject"] == True]

    # Combining the different info:
    coords = pd.DataFrame()
    colors = pd.DataFrame()
    activation = pd.DataFrame()
    latencies = pd.DataFrame()
    for ch in list(vis_resp_results_sig["channel"].unique()):
        # Extract the coordinates of this channel:
        coords = coords.append(pd.DataFrame({
            "channel": ch,
            "x": fsaverage_coord.loc[fsaverage_coord["name"] == ch, "x"].item(),
            "y": fsaverage_coord.loc[fsaverage_coord["name"] == ch, "y"].item(),
            "z": fsaverage_coord.loc[fsaverage_coord["name"] == ch, "z"].item()
        }, index=[0]))
        colors = colors.append(pd.DataFrame({
            "channel": ch,
            "r": 255/255,
            "g": 127/255,
            "b": 80/255
        }, index=[0]))
        activation = activation.append(pd.DataFrame({
            "channel": ch,
            "activation_ti": vis_resp_results_sig.loc[vis_resp_results_sig["channel"] == ch,
                                                      "effect_strength-stimulus onset/Irrelevant"].item(),
            "activation_tr": vis_resp_results_sig.loc[vis_resp_results_sig["channel"] == ch,
                                                      "effect_strength-stimulus onset/Relevant non-target"].item()
        }, index=[0]))
        latencies = latencies.append(pd.DataFrame({
            "channel": ch,
            "latencies_ti": vis_resp_results_sig.loc[vis_resp_results_sig["channel"] == ch,
                                                     "latency-stimulus onset/Irrelevant"].item(),
            "latencies_tr": vis_resp_results_sig.loc[vis_resp_results_sig["channel"] == ch,
                                                     "latency-stimulus onset/Relevant non-target"].item()
        }, index=[0]))
    # Reset indices:
    coords = coords.reset_index(drop=True)
    colors = colors.reset_index(drop=True)
    activation = activation.reset_index(drop=True)
    latencies = latencies.reset_index(drop=True)
    # Save to csvs:
    save_dir = Path(save_root, folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coords.to_csv(Path(save_dir, "coords.csv"))
    colors.to_csv(Path(save_dir, "coords_colors.csv"))
    activation.to_csv(Path(save_dir, "coords_activation.csv"))
    latencies.to_csv(Path(save_dir, "coords_latencies.csv"))

    # ================================================================================================
    # Plot the channels counts per ROI:
    results_path = Path(results_root, folder[0])
    subdirs = [x for x in results_path.iterdir() if x.is_dir()]
    results_files = []
    for file in glob.glob(str(Path(subdirs[0], '*all_results.csv'))):
        results_files.append(file)
    confg_files = []
    for file in glob.glob(str(Path(subdirs[0], '*.json'))):
        confg_files.append(file)

    # ==========================================================================
    # Plot the channels counts per ROI:
    # Load the epochs:
    epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                       "ses-" + ses, "ieeg", vis_resp_param["preprocessing_folder"],
                       vis_resp_param["analysis_parameters"][folder]["signal"],
                       vis_resp_param["preprocess_steps"]))
    epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
    epochs = epochs_loader(subjects, epo_dir, epo_file, vis_resp_results["channel"].to_list(), crop_time, ses,
                           conditions=vis_resp_param["analysis_parameters"][folder]["conditions"])

    # Extract which ROI the channels belong to:
    channels_roi = get_channels_rois(epochs, bids_root, aseg="aparc.a2009s+aseg")

    # Count the # channels per ROI:
    roi_cts = dict(tuple(channels_roi.groupby(["label"])))
    roi_cts = {label: len(roi_cts[label]) for label in roi_cts.keys()}

    # Plot the brain surfaces:
    save_file = Path(save_dir, "channel_counts.png")
    plot_brain(subject='fsaverage', surface='inflated', hemi='lh', sulc_map='curv', parc='aparc.a2009s',
               roi_map=roi_cts, views=['lateral', 'medial'],
               cmap='Reds', colorbar=True, colorbar_title='# channels', vmin=None, vmax=None,
               overlay_method='overlay',
               brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6),
               save_file=save_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    subjects_list = None
    analysis_folder = "high_gamma_wilcoxon_onset_two_tailed"
    brain_plots_handler(subjects_list, analysis_folder,
                        save_root="/hpc/users/alexander.lepauvre/plotting_test/brain_plots",
                        cond_to_plot="both")
