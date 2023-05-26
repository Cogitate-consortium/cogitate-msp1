"""
This script plots the outcome of hte visual responsiveness
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import mne
import matplotlib

from general_helper_functions.plotters import sort_epochs, MidpointNormalize
from general_helper_functions.pathHelperFunctions import path_generator, find_files
from general_helper_functions.data_general_utilities import stack_evoked, mean_confidence_interval

from visual_responsiveness_analysis.visual_responsivness_parameters_class import VisualResponsivnessAnalysisParameters
from visual_responsiveness_analysis.visual_responsiveness_helper_functions import load_epochs
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
fig_size = [20, 15]
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
cmap = "RdYlBu_r"
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi


def plot_single_electrodes(configs, save_folder="super"):
    # ==================================================================================================================
    # Fetching all the config files if none were passed:
    if configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")

    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = \
            VisualResponsivnessAnalysisParameters(
                config, sub_id=save_folder)

        # Looping through the different analysis performed in the visual responsivness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create the path to where the data should be saved:
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)
            results_path = path_generator(param.save_root,
                                          analysis=analysis_name,
                                          preprocessing_steps=param.preprocess_steps,
                                          fig=False, stats=True)

            # Load all the results:
            results_file = find_files(results_path, naming_pattern="*all_results")[0]
            results = pd.read_csv(results_file)

            # Load every subject's data:
            epochs = {}
            mni_coords = []
            with open(Path(save_path_fig, 'subjects_list.txt'), 'w') as f:
                for subject in list(results["subject"].unique()):
                    f.write(f"{subject}\n")
            # Load the epochs of every subject:
            for subject in results["subject"].unique():
                epochs[subject], mni_coordinates = load_epochs(param.BIDS_root, analysis_parameters["signal"],
                                                               subject,
                                                               session=param.session,
                                                               task_name=param.task_name,
                                                               preprocess_folder=param.preprocessing_folder,
                                                               preprocess_steps=param.preprocess_steps,
                                                               channel_types={"seeg": True, "ecog": True},
                                                               condition=analysis_parameters["conditions"],
                                                               baseline_method=analysis_parameters[
                                                                   "baseline_correction"],
                                                               baseline_time=analysis_parameters["baseline_time"],
                                                               crop_time=analysis_parameters["crop_time"],
                                                               select_vis_resp=False,
                                                               vis_resp_folder=None,
                                                               aseg=param.aseg,
                                                               montage_space="T1",
                                                               get_mni_coord=True,
                                                               picks_roi=None,
                                                               filtering_parameters=analysis_parameters[
                                                                   "multitaper_parameters"])
                mni_coords.append(mni_coordinates)
            mni_coords = pd.concat(mni_coords, ignore_index=True)
            channels_info = pd.DataFrame()
            # Looping through the results to append info to the mni coordinates:
            for ind, channel in enumerate(results["channel"].to_list()):
                # Get the channel mni coordinate:
                ch_mni_coord = mni_coords.loc[mni_coords["channels"] == channel]
                ch_results = results.loc[results["channel"] == channel]
                # Create the table:
                channels_info = channels_info.append(pd.DataFrame({
                    "subject_id": channel.split("-")[0],
                    "channel": channel,
                    "ch_types": ch_mni_coord["ch_types"].item(),
                    "x": ch_mni_coord["x"].item(),
                    "y": ch_mni_coord["y"].item(),
                    "z": ch_mni_coord["z"].item(),
                    "condition": ch_results["condition"].item(),
                    "reject": ch_results["reject"].item(),
                    "onset": ch_results["onset"].item(),
                    "offset": ch_results["offset"].item(),
                    "effect_strength": ch_results["effect_strength"].item()
                }, index=[ind]))
            # Saving the results to file:
            channels_info.to_csv(Path(save_path_fig, param.files_prefix + "channels_info.csv"))

            # Get the labels of each channel according to the atlas:
            channels_labels = pd.DataFrame()
            for subject in epochs.keys():
                # Get the labels of these channels:
                labels, _ = mne.get_montage_volume_labels(
                    epochs[subject].get_montage(), "sub-" + subject,
                    subjects_dir=Path(param.BIDS_root, "derivatives", "fs"), aseg=param.aseg)
                # Convert the labels to a dataframe:
                subjects_label_df = pd.DataFrame()
                for ind, channel in enumerate(labels.keys()):
                    subjects_label_df = subjects_label_df.append(
                        pd.DataFrame({"channel": channel, "region": "/".join(labels[channel])}, index=[ind]))
                channels_labels = channels_labels.append(subjects_label_df, ignore_index=True)

            # Generate one directory per ROI:
            roi_dirs = {roi: None for roi in param.rois.keys()}
            for roi in roi_dirs.keys():
                roi_dirs[roi] = Path(save_path_fig, roi)
                if not os.path.isdir(roi_dirs[roi]):
                    os.makedirs(roi_dirs[roi])
            # Adding the path to whatever isn't in the ROIs:
            roi_dirs["other"] = Path(save_path_fig, "other")
            if not os.path.isdir(roi_dirs["other"]):
                os.makedirs(roi_dirs["other"])

            # ==========================================================================================================
            # Plotting single electrodes evoked response:
            colors = sns.color_palette("colorblind")
            print("=" * 40)
            print("Plotting single electrodes evoked responses")
            # Looping through each subject:
            for subject in epochs.keys():
                print("Plotting ")
                # Figure out whether the results were significant for this one subject:
                subject_results = results.loc[results["subject"] == subject]
                # Get the epochs of the said subject:
                sub_epochs = epochs[subject]
                # Looping through each channel of this subject:
                for channel in epochs[subject].ch_names:
                    print("Plotting {} evoked response".format(channel))
                    # Get the data of the said subject:
                    ch_epochs = sub_epochs.copy().pick(channel)
                    # Check whether this electrode is significant:
                    rej = subject_results.loc[subject_results["channel"] == channel,
                                              "reject"].item()
                    # Find in which ROI this channel is:
                    channel_roi = None
                    for roi in param.rois.keys():
                        if channel_roi is None:
                            roi_list = param.rois[roi]
                            # Looping through the channel labels:
                            ch_label = channels_labels.loc[channels_labels["channel"] == channel, "region"].item()
                            for label in ch_label.split("/"):
                                if label in roi_list:
                                    channel_roi = roi
                    if channel_roi is None:
                        channel_roi = "other"
                    if rej is True:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "sig-" + param.files_prefix + channel + "_evoked.png")
                    else:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "non_sig-" + param.files_prefix + channel + "_evoked.png")
                    fig, ax = plt.subplots(len(analysis_parameters["evoked_parameters"]["conds_2"]),
                                           len(analysis_parameters["evoked_parameters"]["conds_1"]),
                                           figsize=fig_size, sharey=True, sharex=True)
                    for ind_1, cond_1 in enumerate(analysis_parameters["evoked_parameters"]["conds_1"]):
                        # Get the data of the said condition and channel:
                        epochs_cond_1 = ch_epochs[cond_1]
                        for ind_2, cond_2 in enumerate(analysis_parameters["evoked_parameters"]["conds_2"]):
                            epochs_cond_2 = epochs_cond_1[cond_2]
                            for ind_3, cond_3 in enumerate(analysis_parameters["evoked_parameters"]["conds_3"]):
                                # Compute the mean and ci:
                                avg, low_ci, high_ci = \
                                    mean_confidence_interval(np.squeeze(epochs_cond_2[cond_3].get_data()),
                                                             confidence=0.95)
                                # Plotting the average with the confidence interval:
                                ax[ind_2, ind_1].plot(ch_epochs.times,
                                                      avg * analysis_parameters["raster_parameters"]["scaling"],
                                                      color=colors[ind_3],
                                                      label=cond_3)
                                ax[ind_2, ind_1].fill_between(ch_epochs.times,
                                                              low_ci * analysis_parameters["raster_parameters"][
                                                                  "scaling"],
                                                              high_ci * analysis_parameters["raster_parameters"][
                                                                  "scaling"],
                                                              color=colors[ind_3], alpha=.2)
                            # Add the vertical lines:
                            # Adding the vertical lines;
                            ax[ind_2, ind_1].vlines(analysis_parameters["evoked_parameters"]["v_lines"],
                                                    ax[ind_2, ind_1].get_ylim()[0], ax[ind_2, ind_1].get_ylim()[1],
                                                    linestyles='dashed', linewidth=1.5, colors='k')
                            # Set only the relevant x ticks:
                            ax[ind_2, ind_1].set_xticks(analysis_parameters["evoked_parameters"]["v_lines"])
                            ax[ind_2, ind_1].set_xticklabels([str(val)
                                                              for val in
                                                              analysis_parameters["evoked_parameters"]["v_lines"]])
                            # Setting the axis limit:
                            ax[ind_2, ind_1].set_xlim(ch_epochs.times[0], ch_epochs.times[-1])
                            # Add the axis title:
                            if ind_2 == 0:
                                ax[ind_2, ind_1].set_title(cond_1 + "\n" + cond_2)
                            else:
                                ax[ind_2, ind_1].set_title(cond_2)
                            # Add the legend:
                            if ind_1 == len(analysis_parameters["evoked_parameters"]["conds_1"]) - 1 \
                                    and ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                ax[ind_2, ind_1].legend()
                            # Add the x axis legend only if we are on the last row:
                            if ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                ax[ind_2, ind_1].set_xlabel("Time (s)")
                            else:
                                ax[ind_2, ind_1].set_xticks([])
                            # Set the y label:
                            ax[ind_2, ind_1].set_ylabel(analysis_parameters["evoked_parameters"]["ylabel"])
                    # Set the super title and adjust to page:
                    plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal")
                    plt.tight_layout()
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()

            # ==========================================================================================================
            # Plot combination of evoked and raster:
            print("=" * 40)
            print("Plotting single electrodes rasters and evoked")
            for subject in epochs.keys():
                subject_results = results.loc[results["subject"] == subject]
                sub_epochs = epochs[subject]
                for channel in epochs[subject].ch_names:
                    print("Plotting {} raster".format(channel))
                    ch_epochs = sub_epochs.copy().pick(channel)
                    # Check whether this electrode is significant:
                    rej = subject_results.loc[subject_results["channel"] == channel,
                                              "reject"].item()
                    # Find in which ROI this channel is:
                    channel_roi = None
                    for roi in param.rois.keys():
                        if channel_roi is None:
                            roi_list = param.rois[roi]
                            # Looping through the channel labels:
                            ch_label = channels_labels.loc[channels_labels["channel"] == channel, "region"].item()
                            for label in ch_label.split("/"):
                                if label in roi_list:
                                    channel_roi = roi
                    if channel_roi is None:
                        channel_roi = "other"
                    if rej is True:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "sig-" + param.files_prefix + channel + "_evoked_raster.png")
                    else:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "non_sig-" + param.files_prefix + channel + "_raster.png")
                    fig, (ax_rast, ax_evo) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                                          figsize=fig_size, sharex=True)
                    # Get the extremes values
                    extremes = [np.percentile(ch_epochs.get_data(), 5), np.percentile(ch_epochs.get_data(), 95)]
                    # Sort the epochs:
                    order, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels = \
                        sort_epochs(ch_epochs.metadata,
                                    analysis_parameters["evoked_raster_parameters"]["sort_conditions"],
                                    order=analysis_parameters["evoked_raster_parameters"]["order"])
                    if analysis_parameters["raster_parameters"]["sigma"] > 0:
                        img = gaussian_filter1d(np.squeeze(ch_epochs.get_data())[order],
                                                sigma=analysis_parameters["evoked_raster_parameters"]["sigma"],
                                                axis=-1, mode="nearest")
                    else:
                        img = np.squeeze(ch_epochs.get_data())[order]
                    # Generate a cmap that will be centered on what we want:
                    norm = MidpointNormalize(vmin=analysis_parameters["evoked_raster_parameters"]["vmin"],
                                             midpoint=analysis_parameters[
                                                 "raster_parameters"]["cmap_center"],
                                             vmax=analysis_parameters["evoked_raster_parameters"]["vmax"])
                    # Plot a heat map of the sorted trials:
                    im = ax_rast.imshow(img * analysis_parameters["raster_parameters"]["scaling"],
                                        cmap=cmap, norm=norm,
                                        extent=[ch_epochs.times[0], ch_epochs.times[-1], 0,
                                                len(order)],
                                        aspect="auto", origin='lower')
                    # Set the axis and color bars:
                    # Add the y labels:
                    y_labels = list(
                        cond_y_ticks_labels[analysis_parameters["evoked_raster_parameters"]["sort_conditions"][0]])
                    labels_loc = \
                        cond_y_ticks_pos[analysis_parameters["evoked_raster_parameters"]["sort_conditions"][0]]
                    ax_rast.set_yticks(labels_loc)
                    ax_rast.set_yticklabels(y_labels * int((len(labels_loc) / len(y_labels))))
                    ax_rast.set_xticks(analysis_parameters["raster_parameters"]["v_lines"])
                    ax_rast.set_xticklabels([str(val)
                                             for val in
                                             analysis_parameters["raster_parameters"]["v_lines"]])
                    # Adding the vertical lines;
                    ax_rast.vlines(analysis_parameters["evoked_raster_parameters"]["v_lines"],
                                   ax_rast.get_ylim()[0], ax_rast.get_ylim()[1],
                                   linestyles='dashed', linewidth=1.5, colors='k')
                    fig.subplots_adjust(right=0.9)
                    cbar_ax = fig.add_axes([0.92, 0.355, 0.05, 0.525])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(analysis_parameters["evoked_raster_parameters"]["ylabel"])
                    cbar.ax.yaxis.set_label_position('left')
                    # Compute the average and ci:
                    avg, low_ci, high_ci = mean_confidence_interval(img, confidence=0.95)
                    # Plotting the average with the confidence interval:
                    ax_evo.plot(ch_epochs.times, avg * analysis_parameters["raster_parameters"]["scaling"],
                                color="k")
                    ax_evo.fill_between(ch_epochs.times,
                                        low_ci * analysis_parameters["raster_parameters"]["scaling"],
                                        high_ci * analysis_parameters["raster_parameters"]["scaling"],
                                        color="k", alpha=.2)
                    ax_evo.set_xticks(analysis_parameters["raster_parameters"]["v_lines"])
                    ax_evo.set_xticklabels([str(val)
                                            for val in
                                            analysis_parameters["evoked_raster_parameters"]["v_lines"]])
                    ax_evo.vlines(analysis_parameters["evoked_raster_parameters"]["v_lines"],
                                  ax_evo.get_ylim()[0], ax_evo.get_ylim()[1],
                                  linestyles='dashed', linewidth=1.5, colors='k')
                    ax_evo.set_xlabel("Time (s)")
                    ax_evo.set_ylabel(analysis_parameters["evoked_raster_parameters"]["ylabel"])
                    plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal")
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()

            # ==========================================================================================================
            # Plotting single electrodes rasters:
            print("=" * 40)
            print("Plotting single electrodes rasters")
            for subject in epochs.keys():
                subject_results = results.loc[results["subject"] == subject]
                sub_epochs = epochs[subject]
                for channel in epochs[subject].ch_names:
                    print("Plotting {} raster".format(channel))
                    ch_epochs = sub_epochs.copy().pick(channel)
                    # Check whether this electrode is significant:
                    rej = subject_results.loc[subject_results["channel"] == channel,
                                              "reject"].item()
                    # Find in which ROI this channel is:
                    channel_roi = None
                    for roi in param.rois.keys():
                        if channel_roi is None:
                            roi_list = param.rois[roi]
                            # Looping through the channel labels:
                            ch_label = channels_labels.loc[channels_labels["channel"] == channel, "region"].item()
                            for label in ch_label.split("/"):
                                if label in roi_list:
                                    channel_roi = roi
                    if channel_roi is None:
                        channel_roi = "other"

                    if rej is True:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "sig-" + param.files_prefix + channel + "_" + "_raster.png")
                    else:
                        file_prefix = Path(roi_dirs[channel_roi],
                                           "non_sig-" + param.files_prefix + channel + "_" + "_raster.png")
                    fig, ax = plt.subplots(len(analysis_parameters["raster_parameters"]["conds_2"]),
                                           len(analysis_parameters["raster_parameters"]["conds_1"]),
                                           figsize=fig_size)
                    # Getting vmin and vmax:
                    vmin, vmax = np.percentile(ch_epochs.get_data(), 5), np.percentile(ch_epochs.get_data(), 95)
                    for ind_1, cond_1 in enumerate(analysis_parameters["raster_parameters"]["conds_1"]):
                        # Get the data of the said condition and channel:
                        epochs_cond_1 = ch_epochs[cond_1]
                        for ind_2, cond_2 in enumerate(analysis_parameters["raster_parameters"]["conds_2"]):
                            epochs_cond_2 = epochs_cond_1[cond_2]
                            # Get the epochs order:
                            order, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels = \
                                sort_epochs(epochs_cond_2.metadata,
                                            analysis_parameters["raster_parameters"]["sort_conditions"],
                                            order=analysis_parameters["raster_parameters"]["order"])
                            if analysis_parameters["raster_parameters"]["sigma"] > 0:
                                img = gaussian_filter1d(np.squeeze(epochs_cond_2.get_data())[order],
                                                        sigma=analysis_parameters["raster_parameters"]["sigma"],
                                                        axis=-1, mode="nearest")
                            else:
                                img = np.squeeze(epochs_cond_2.get_data())[order]
                            # Generate a cmap that will be centered on what we want:
                            norm = MidpointNormalize(
                                vmin=analysis_parameters["evoked_raster_parameters"]["vmin"],
                                midpoint=analysis_parameters[
                                    "raster_parameters"]["cmap_center"],
                                vmax=analysis_parameters[
                                    "evoked_raster_parameters"]["vmax"])
                            # Plot a heat map of the sorted trials:
                            im = ax[ind_2, ind_1].imshow(img * analysis_parameters["raster_parameters"]["scaling"],
                                                         cmap=cmap, norm=norm,
                                                         extent=[epochs_cond_2.times[0], epochs_cond_2.times[-1], 0,
                                                                 len(order)],
                                                         aspect="auto", origin='lower')
                            # Add the y labels:
                            y_labels = list(
                                cond_y_ticks_labels[analysis_parameters["raster_parameters"]["sort_conditions"][0]])
                            labels_loc = \
                                cond_y_ticks_pos[analysis_parameters["raster_parameters"]["sort_conditions"][0]]
                            ax[ind_2, ind_1].set_yticks(labels_loc)
                            ax[ind_2, ind_1].set_yticklabels(y_labels * int((len(labels_loc) / len(y_labels))))
                            # Adding the vertical lines;
                            ax[ind_2, ind_1].vlines(analysis_parameters["raster_parameters"]["v_lines"],
                                                    ax[ind_2, ind_1].get_ylim()[0], ax[ind_2, ind_1].get_ylim()[1],
                                                    linestyles='dashed', linewidth=1.5, colors='k')
                            # Set only the relevant x ticks:
                            if ind_2 == 0:
                                ax[ind_2, ind_1].set_xticks(analysis_parameters["raster_parameters"]["v_lines"])
                                ax[ind_2, ind_1].set_xticklabels([str(val)
                                                                  for val in
                                                                  analysis_parameters["raster_parameters"]["v_lines"]])
                            # Add the x axis legend only if we are on the last row:
                            if ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                ax[ind_2, ind_1].set_xlabel("Time (s)")
                            else:
                                ax[ind_2, ind_1].set_xticks([])
                            # Adding the title:
                            if ind_2 == 0:
                                ax[ind_2, ind_1].set_title(cond_1 + "\n" + cond_2)
                            else:
                                ax[ind_2, ind_1].set_title(cond_2)
                    # Add the color bar:
                    # Add the super title:
                    plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal")
                    plt.tight_layout()
                    fig.subplots_adjust(right=0.9)
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(analysis_parameters["raster_parameters"]["cbar_label"])
                    cbar.ax.yaxis.set_label_position('left')
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()


def plot_roi_evoked(configs, save_folder="super"):
    # ==================================================================================================================
    # Fetching all the config files if none were passed:
    if configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")

    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = \
            VisualResponsivnessAnalysisParameters(
                config, sub_id=save_folder)

        # Looping through the different analysis performed in the visual responsivness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create the path to where the data should be saved:
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)
            results_path = path_generator(param.save_root,
                                          analysis=analysis_name,
                                          preprocessing_steps=param.preprocess_steps,
                                          fig=False, stats=True)

            # Load all the results:
            results_file = find_files(results_path, naming_pattern="*all_results")[0]
            results = pd.read_csv(results_file)

            # Generate one directory per ROI:
            roi_dirs = {roi: None for roi in param.rois.keys()}
            for roi in roi_dirs.keys():
                roi_dirs[roi] = Path(save_path_fig, roi)
                if not os.path.isdir(roi_dirs[roi]):
                    os.makedirs(roi_dirs[roi])
            # Adding the path to whatever isn't in the ROIs:
            roi_dirs["other"] = Path(save_path_fig, "other")
            if not os.path.isdir(roi_dirs["other"]):
                os.makedirs(roi_dirs["other"])

            # Prepare the plot:
            fig_all_roi, ax_all_roi = plt.subplots(figsize=fig_size)
            # Looping through each ROI:
            for roi in param.rois.keys():
                # Load every subject's data:
                epochs = {}
                # Load the epochs of every subject:
                for subject in results["subject"].unique():
                    epochs[subject], _ = load_epochs(param.BIDS_root, analysis_parameters["signal"],
                                                     subject,
                                                     session=param.session,
                                                     task_name=param.task_name,
                                                     preprocess_folder=param.preprocessing_folder,
                                                     preprocess_steps=param.preprocess_steps,
                                                     channel_types={"seeg": True, "ecog": True},
                                                     condition=analysis_parameters["conditions"],
                                                     baseline_method=analysis_parameters[
                                                         "baseline_correction"],
                                                     baseline_time=analysis_parameters["baseline_time"],
                                                     crop_time=analysis_parameters["crop_time"],
                                                     select_vis_resp=False,
                                                     vis_resp_folder=None,
                                                     aseg=param.aseg,
                                                     montage_space="T1",
                                                     get_mni_coord=False,
                                                     picks_roi=param.rois[roi],
                                                     filtering_parameters=analysis_parameters["multitaper_parameters"])
                    if epochs[subject] is None:
                        del epochs[subject]

                # ======================================================================================================
                # Plotting the average in separate conditions in this specific ROI:
                colors = sns.color_palette("colorblind")
                fig_single_roi, ax_single_roi = plt.subplots(len(analysis_parameters["evoked_parameters"]["conds_2"]),
                                                             len(analysis_parameters["evoked_parameters"]["conds_1"]),
                                                             figsize=fig_size, sharey=True, sharex=True)
                # Computing the evoked of the onset responsive electrodes:
                evoked = {
                    cond_1:
                        {
                            cond_2:
                                {
                                    cond_3: []
                                    for cond_3 in analysis_parameters["evoked_parameters"]["conds_3"]}
                            for cond_2 in analysis_parameters["evoked_parameters"]["conds_2"]}
                    for cond_1 in analysis_parameters["evoked_parameters"]["conds_1"]
                }
                times = []
                # Looping through each subject:
                for subject in epochs.keys():
                    # Get the results of this subject:
                    sub_results = results.loc[results["subject"] == subject]
                    # Fetching the onset responsive electrodes of this subject:
                    resp_channels = sub_results.loc[sub_results["reject"] == True, "channel"].to_list()
                    # Compute the evoked of only those:
                    try:
                        resp_epochs = epochs[subject].copy().pick(resp_channels)
                        for ind_1, cond_1 in enumerate(analysis_parameters["evoked_parameters"]["conds_1"]):
                            for ind_2, cond_2 in enumerate(analysis_parameters["evoked_parameters"]["conds_2"]):
                                for ind_3, cond_3 in enumerate(analysis_parameters["evoked_parameters"]["conds_3"]):
                                    evoked[cond_1][cond_2][cond_3].append(resp_epochs.copy()[[cond_1, cond_2, cond_3]].
                                                                          average())
                                    times.append(resp_epochs.copy()[[cond_1, cond_2, cond_3]].times)
                    except ValueError:
                        print("WARNING: No responsive channels found for sub-{}".format(subject))
                # Concatenating the evoked objects:
                evoked = {
                    cond_1:
                        {
                            cond_2:
                                {
                                    cond_3: stack_evoked(evoked[cond_1][cond_2][cond_3])
                                    for cond_3 in analysis_parameters["evoked_parameters"]["conds_3"]}
                            for cond_2 in analysis_parameters["evoked_parameters"]["conds_2"]}
                    for cond_1 in analysis_parameters["evoked_parameters"]["conds_1"]
                }
                for ind_1, cond_1 in enumerate(evoked.keys()):
                    for ind_2, cond_2 in enumerate(evoked[cond_1].keys()):
                        for ind_3, cond_3 in enumerate(evoked[cond_1][cond_2].keys()):
                            # Compute the mean and ci:
                            avg, low_ci, high_ci = \
                                mean_confidence_interval(np.squeeze(evoked[cond_1][cond_2][cond_3].get_data()),
                                                         confidence=0.95)
                            # Plotting the average with the confidence interval:
                            ax_single_roi[ind_2, ind_1].plot(evoked[cond_1][cond_2][cond_3].times,
                                                             avg * analysis_parameters["raster_parameters"]["scaling"],
                                                             color=colors[ind_3],
                                                             label=cond_3)
                            ax_single_roi[ind_2, ind_1].fill_between(evoked[cond_1][cond_2][cond_3].times,
                                                                     low_ci * analysis_parameters["raster_parameters"][
                                                                         "scaling"],
                                                                     high_ci * analysis_parameters["raster_parameters"][
                                                                         "scaling"],
                                                                     color=colors[ind_3], alpha=.2)
                        # Add the vertical lines:
                        # Adding the vertical lines;
                        ax_single_roi[ind_2, ind_1].vlines(analysis_parameters["evoked_parameters"]["v_lines"],
                                                           ax_single_roi[ind_2, ind_1].get_ylim()[0],
                                                           ax_single_roi[ind_2, ind_1].get_ylim()[1],
                                                           linestyles='dashed', linewidth=1.5, colors='k')
                        # Set only the relevant x ticks:
                        ax_single_roi[ind_2, ind_1].set_xticks(analysis_parameters["evoked_parameters"]["v_lines"])
                        ax_single_roi[ind_2, ind_1].set_xticklabels([str(val)
                                                                     for val in
                                                                     analysis_parameters["evoked_parameters"][
                                                                         "v_lines"]])
                        # Setting the axis limit:
                        ax_single_roi[ind_2, ind_1].set_xlim(times[0][0], times[0][-1])
                        # Add the axis title:
                        if ind_2 == 0:
                            ax_single_roi[ind_2, ind_1].set_title(cond_1 + "\n" + cond_2)
                        else:
                            ax_single_roi[ind_2, ind_1].set_title(cond_2)
                        # Add the legend:
                        if ind_1 == len(analysis_parameters["evoked_parameters"]["conds_1"]) - 1 \
                                and ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                            ax_single_roi[ind_2, ind_1].legend()
                        # Add the x axis legend only if we are on the last row:
                        if ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                            ax_single_roi[ind_2, ind_1].set_xlabel("Time (s)")
                        else:
                            ax_single_roi[ind_2, ind_1].set_xticks([])
                        # Set the y label:
                        ax_single_roi[ind_2, ind_1].set_ylabel(analysis_parameters["evoked_parameters"]["ylabel"])
                # Set the super title and adjust to page:
                plt.suptitle(roi + " " + analysis_parameters["signal"] + " signal")
                plt.tight_layout()
                # Save the figure:
                file_prefix = Path(roi_dirs[roi], param.files_prefix + "_" + roi + "_grand_average.png")
                plt.savefig(file_prefix, transparent=True)
                plt.close()
                # ======================================================================================================
                # Plotting average per ROI:
                print("=" * 40)
                print("Plotting ROI evoked responses")
                subjects_dir = Path(mne.datasets.sample.data_path(), 'subjects')
                mgz_fname = Path(subjects_dir, "fsaverage", "mri", param.aseg + ".mgz")
                vol_labels, label_colors = mne.get_volume_labels_from_aseg(mgz_fname, return_colors=True)
                # Computing the evoked of the onset responsive electrodes:
                evoked = []
                # Looping through each subject:
                for subject in epochs.keys():
                    # Get the results of this subject:
                    sub_results = results.loc[results["subject"] == subject]
                    # Fetching the onset responsive electrodes of this subject:
                    resp_channels = sub_results.loc[sub_results["reject"] == True, "channel"].to_list()
                    # Compute the evoked of only those:
                    try:
                        evoked.append(epochs[subject].copy().pick(resp_channels).average())
                    except ValueError:
                        print("WARNING: No responsive channels found for sub-{}".format(subject))
                # Now, stacking all these evoked together:
                evoked = stack_evoked(evoked)
                # Compute average and confidence interval for this ROI:
                avg, low_ci, high_ci = mean_confidence_interval(evoked.data, confidence=0.95)
                # Plotting the average with the confidence interval:
                ax_all_roi.plot(evoked.times, avg * analysis_parameters["raster_parameters"]["scaling"],
                                label=roi + " (# {})".format(len(evoked.ch_names)))
                ax_all_roi.fill_between(evoked.times, low_ci * analysis_parameters["raster_parameters"]["scaling"],
                                        high_ci * analysis_parameters["raster_parameters"]["scaling"], alpha=.2)
            # Add axis decorations:
            title = "Evoked responses per ROIs"
            x_label = "Time (s)"
            y_label = "gain" if analysis_parameters["signal"] == "high_gamma" else "amplitude (microV)"
            ax_all_roi.set_title(title)
            ax_all_roi.legend()
            ax_all_roi.set_ylabel(y_label)
            ax_all_roi.set_xlabel(x_label)
            plt.tight_layout()
            # Save to a file:
            file_prefix = Path(save_path_fig, "sig-" + param.files_prefix + "_rois_evoked.png")
            plt.savefig(file_prefix, transparent=True)
            plt.close()


if __name__ == "__main__":
    configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")
    plot_roi_evoked(configs, save_folder="super")
    plot_single_electrodes(configs, save_folder="super")
