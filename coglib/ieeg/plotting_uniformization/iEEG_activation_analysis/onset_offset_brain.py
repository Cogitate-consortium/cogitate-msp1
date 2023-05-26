import pandas as pd
from pathlib import Path
import config
import theories_rois
import glob
import os
import subprocess
import json
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from general_utilities import mean_confidence_interval, epochs_loader, get_roi_channels, get_ch_rois, \
    load_fsaverage_coord, corrected_sem
import seaborn as sns
from plotters import plot_time_series, plot_rasters, mm2inch
from ecog_plotters import plot_electrodes
from scipy.stats import sem
from scipy.ndimage import uniform_filter1d

param = config.param
# Set up the colors:
onset_color = "YlGn"
offset_color = "YlOrRd"
onset_cmap = plt.get_cmap(onset_color)
offset_cmap = plt.get_cmap(offset_color)
norm = mpl.colors.Normalize(vmin=1, vmax=4)
onset_scalar_map = cm.ScalarMappable(norm=norm, cmap=onset_cmap)
offset_scalar_map = cm.ScalarMappable(norm=norm, cmap=offset_cmap)

bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
save_dir = "/hpc/users/alexander.lepauvre/plotting_test/activation_analysis/onset_offset_high_gamma"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Set the path to the files:
face_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
               "-V1/ieeg/results/onset_offset_high_gamma_gnw_face_ti/desbadcharej_notfil_lapref/sub-super_ses-V1_task" \
               "-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
object_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/onset_offset_high_gamma_gnw_object_ti/desbadcharej_notfil_lapref/sub-super_ses" \
                 "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
letter_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/onset_offset_high_gamma_gnw_letter_ti/desbadcharej_notfil_lapref/sub-super_ses" \
                 "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
false_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                "-V1/ieeg/results/onset_offset_high_gamma_gnw_false_ti/desbadcharej_notfil_lapref/sub-super_ses" \
                "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
face_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
               "-V1/ieeg/results/onset_offset_high_gamma_gnw_face_tr/desbadcharej_notfil_lapref/sub-super_ses-V1_task" \
               "-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
object_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/onset_offset_high_gamma_gnw_object_tr/desbadcharej_notfil_lapref/sub-super_ses" \
                 "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
letter_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                 "-V1/ieeg/results/onset_offset_high_gamma_gnw_letter_tr/desbadcharej_notfil_lapref/sub-super_ses" \
                 "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
false_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                "-V1/ieeg/results/onset_offset_high_gamma_gnw_false_tr/desbadcharej_notfil_lapref/sub-super_ses" \
                "-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
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
subjects = list(set([ch.split("-")[0] for ch in channels]))
# Get the ROI and position of those channels:
ch_coords = load_fsaverage_coord(bids_root, subjects, ses='V1', laplace_reloc=True)
ch_rois = get_ch_rois(bids_root, subjects, ses='V1', laplace_reloc=True)

channels_loc = pd.DataFrame()
channels_colors = pd.DataFrame()
# Loop through each channel:
for ch in channels:
    # Locate the channel results in each table:
    face_res_ti = face_ti_results.loc[face_ti_results["channel"] == ch, "condition"].item()
    object_res_ti = object_ti_results.loc[object_ti_results["channel"] == ch, "condition"].item()
    letter_res_ti = letter_ti_results.loc[letter_ti_results["channel"] == ch, "condition"].item()
    false_res_ti = false_ti_results.loc[false_ti_results["channel"] == ch, "condition"].item()
    face_res_tr = face_tr_results.loc[face_tr_results["channel"] == ch, "condition"].item()
    object_res_tr = object_tr_results.loc[object_tr_results["channel"] == ch, "condition"].item()
    letter_res_tr = letter_tr_results.loc[letter_tr_results["channel"] == ch, "condition"].item()
    false_res_tr = false_tr_results.loc[false_tr_results["channel"] == ch, "condition"].item()

    # Make the conjunction:
    if face_res_ti == face_res_tr:
        if 'stimulus onset' in face_res_ti:
            face_onset = True
            face_offset = False
        elif 'stimulus offset' in face_res_ti:
            face_onset = False
            face_offset = True
        else:
            face_onset = False
            face_offset = False
    else:
        face_onset = False
        face_offset = False

    if object_res_ti == object_res_tr:
        if 'stimulus onset' in object_res_ti:
            object_onset = True
            object_offset = False
        elif 'stimulus offset' in object_res_ti:
            object_onset = False
            object_offset = True
        else:
            object_onset = False
            object_offset = False
    else:
        object_onset = False
        object_offset = False
    if letter_res_ti == letter_res_tr:
        if 'stimulus onset' in letter_res_ti:
            letter_onset = True
            letter_offset = False
        elif 'stimulus offset' in letter_res_ti:
            letter_onset = False
            letter_offset = True
        else:
            letter_onset = False
            letter_offset = False
    else:
        letter_onset = False
        letter_offset = False
    if false_res_ti == false_res_tr:
        if 'stimulus onset' in false_res_ti:
            false_onset = True
            false_offset = False
        elif 'stimulus offset' in false_res_ti:
            false_onset = False
            false_offset = True
        else:
            false_onset = False
            false_offset = False
    else:
        false_onset = False
        false_offset = False

    # Sum the categories for which we have onset or offset:
    onset_sum = np.sum([face_onset, object_onset, letter_onset, false_onset])
    offset_sum = np.sum([face_offset, object_offset, letter_offset, false_offset])

    # Getting the color:
    if offset_sum > 0:
        c = onset_scalar_map.to_rgba(offset_sum)
    elif onset_sum > 0:
        c = offset_scalar_map.to_rgba(offset_sum)
    else:
        continue

    # Append to the table:
    channels_loc = channels_loc.append(pd.DataFrame({
        "channel": ch,
        "x": ch_coords.loc[ch_coords["name"] == ch, "x"].item(),
        "y": ch_coords.loc[ch_coords["name"] == ch, "y"].item(),
        "z": ch_coords.loc[ch_coords["name"] == ch, "z"].item(),
        "radius": 2
    }, index=[0]), ignore_index=True)
    channels_colors = channels_colors.append(pd.DataFrame({
        "channel": ch,
        "r": c[0],
        "g": c[1],
        "b": c[2],
        "radius": 2
    }, index=[0]), ignore_index=True)

# Save the data:
if len(channels_loc) > 0:
    ch_rois = ch_rois.loc[ch_rois["channel"].isin(channels_colors["channel"].to_list())]
    channels_loc.to_csv(Path(save_dir, "coords.csv"))
    channels_colors.to_csv(Path(save_dir, "coords_colors.csv"))
    ch_rois.to_csv(Path(save_dir, "coords_rois.csv"))
    # Save the ROIs as well:
    rois_colors = pd.DataFrame({roi.replace("ctx_rh_", "").replace("ctx_lh_", ""): np.array(param["colors"]["gnw"])
                                for roi in theories_rois.rois["gnw"]},
                               index=['r', 'g', 'b']).T
    rois_colors['roi'] = rois_colors.index
    rois_colors = rois_colors.reset_index(drop=True)
    rois_colors.to_csv(Path(save_dir, "rois_dict.csv"))

# Plot the onset and offset color bars:
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=onset_color, norm=norm)
cb.set_ticks([1, 2, 3, 4])
cb.set_ticklabels([1, 2, 3, 4])
plt.savefig(Path(save_dir, "onset_color.png"), bbox_inches='tight', transparent=True)
plt.savefig(Path(save_dir, "onset_color.svg"), bbox_inches='tight', transparent=True)
plt.close()

# Plot the onset and offset color bars:
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=offset_color, norm=norm)
cb.set_ticks([1, 2, 3, 4])
cb.set_ticklabels([1, 2, 3, 4])
plt.savefig(Path(save_dir, "offset_color.png"), bbox_inches='tight', transparent=True)
plt.savefig(Path(save_dir, "offset_color.svg"), bbox_inches='tight', transparent=True)
plt.close()
