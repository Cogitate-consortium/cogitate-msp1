"""
This scripts loops through all single config to perform the category selectivity as slurm jobs
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import ptitprince as pt
from general_helper_functions.plotters import sort_epochs, MidpointNormalize

from general_helper_functions.pathHelperFunctions import path_generator

from category_selectivity_analysis.category_selectivity_parameters_class import CategorySelectivityAnalysisParameters

from general_helper_functions.pathHelperFunctions import find_files

import mne

from general_helper_functions.data_general_utilities import stack_evoked, mean_confidence_interval, load_epochs

import matplotlib.pyplot as plt

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
colors = sns.color_palette("colorblind")
cmap = "RdYlBu_r"
fig_size = [20, 15]
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi


def plot_category_selectivity(configs, save_folder="super"):
    # ==================================================================================================================
    # Fetching all the config files if none were passed:
    if configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")

    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = \
            CategorySelectivityAnalysisParameters(
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
            data_path = path_generator(param.save_root,
                                       analysis=analysis_name,
                                       preprocessing_steps=param.preprocess_steps,
                                       fig=False, data=True)

            # Load all the results:
            results_file = find_files(results_path, naming_pattern="*all_results")[0]
            results = pd.read_csv(results_file)
            # Loading the data table that was used for the tests:
            # Get the data file:
            data_file = find_files(data_path, "data", extension=".csv")
            assert len(data_file) == 1, "More than one file was found for rsa results!"
            # Load the file:
            data_df = pd.read_csv(data_file[0])

            # Load every subject's data:
            epochs = {}
            mni_coords = []

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
                                                               select_vis_resp=analysis_parameters["select_vis_resp"],
                                                               vis_resp_folder=analysis_parameters["vis_resp_folder"],
                                                               aseg=param.aseg,
                                                               montage_space="T1", get_mni_coord=True,
                                                               picks_roi=None,
                                                               filtering_parameters=analysis_parameters[
                                                                   "multitaper_parameters"]
                                                               )
                mni_coords.append(mni_coordinates)

            # Generate the channel info file to plot in matlab:
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

            # ==========================================================================================================
            # Plot category differences vizualization:
            # Create the save root:
            save_root = Path(save_path_fig, "cate_diff")
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            # Generate one directory per ROI:
            roi_dirs = {roi: None for roi in param.rois.keys()}
            for roi in roi_dirs.keys():
                roi_dirs[roi] = Path(save_root, roi)
                if not os.path.isdir(roi_dirs[roi]):
                    os.makedirs(roi_dirs[roi])
            # Adding the path to whatever isn't in the ROIs:
            roi_dirs["other"] = Path(save_root, "other")
            if not os.path.isdir(roi_dirs["other"]):
                os.makedirs(roi_dirs["other"])

            # Looping through each electrode:
            print("=" * 40)
            print("Plotting single electrodes categories evoked responses:")
            for subject in epochs.keys():
                subject_results = results.loc[results["subject"] == subject]
                sub_epochs = epochs[subject]
                for channel in epochs[subject].ch_names:
                    # Check whether this electrode is significant:
                    rej = subject_results.loc[subject_results["channel"] == channel,
                                              "reject"].item()
                    if rej is not True:
                        continue

                    print("Plotting {} evoked".format(channel))
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

                    ch_epochs = sub_epochs.copy().pick(channel)
                    # Get the favorite condition:
                    fav_cond = subject_results.loc[subject_results["channel"] == channel, "condition"].item()
                    # Get the dprime:
                    dprime = subject_results.loc[subject_results["channel"] == channel, "effect_strength"].item()
                    # Create file name:
                    file_prefix = Path(roi_dirs[channel_roi],
                                       fav_cond + "-" + param.files_prefix + channel +
                                       "_evoked.png")

                    fig, ax = plt.subplots(figsize=fig_size)
                    for ind, cond in enumerate(analysis_parameters["evoked_parameters"]["conditions"]):
                        # Compute the mean and ci:
                        avg, low_ci, high_ci = \
                            mean_confidence_interval(np.squeeze(ch_epochs[cond].get_data()),
                                                     confidence=0.95)
                        # Plotting the average with the confidence interval:
                        ax.plot(ch_epochs.times,
                                avg * analysis_parameters["evoked_parameters"]["scaling"],
                                color=colors[ind],
                                label=cond)
                        ax.fill_between(ch_epochs.times,
                                        low_ci * analysis_parameters["evoked_parameters"][
                                            "scaling"],
                                        high_ci * analysis_parameters["evoked_parameters"][
                                            "scaling"],
                                        color=colors[ind], alpha=.2)

                    # Adding vlines:
                    ax.vlines(analysis_parameters["evoked_parameters"]["v_lines"],
                              ax.get_ylim()[0], ax.get_ylim()[1],
                              linestyles='dashed', linewidth=1)
                    # Set only the relevant x ticks:
                    ax.set_xticks(analysis_parameters["evoked_parameters"]["v_lines"])
                    ax.set_xticklabels([str(val)
                                        for val in
                                        analysis_parameters["evoked_parameters"]["v_lines"]])
                    # Setting the axis limit:
                    ax.set_xlim(ch_epochs.times[0], ch_epochs.times[-1])
                    # Set axis and titles text:
                    ax.set_title("{}, dprime={}".format(channel, dprime))
                    ax.legend()
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel(analysis_parameters["evoked_parameters"]["ylabel"])

                    # Set the super title and adjust to page:
                    plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal")
                    plt.tight_layout()
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()

                    # Plotting boxplot:
                    print("Plotting {} boxplot".format(channel))
                    # Extract the data of this channel:
                    ch_data = data_df.loc[data_df["channel"] == channel]
                    # Create file name:
                    file_prefix = Path(roi_dirs[channel_roi],
                                       fav_cond + "-" + param.files_prefix + channel +
                                       "_boxplot.png")

                    fig, ax = plt.subplots(figsize=fig_size)
                    # Plot half violin:
                    pt.half_violinplot(x=analysis_parameters["boxplot_parameters"]["boxes_condition"],
                                       y="value", data=ch_data, bw=.2, cut=0.,
                                       scale="area", width=.6, inner=None, ax=ax,
                                       order=analysis_parameters["boxplot_parameters"]["boxes_order"])
                    # Plot boxplot:
                    sns.boxplot(x=analysis_parameters["boxplot_parameters"]["boxes_condition"],
                                y="value", data=ch_data, ax=ax, width=0.2, color="white",
                                order=analysis_parameters["boxplot_parameters"]["boxes_order"])
                    ax.set_title("{} boxplot, dprime={}".format(channel, dprime))
                    ax.set_ylabel(analysis_parameters["boxplot_parameters"]["ylabel"])
                    ax.set_xlabel(analysis_parameters["boxplot_parameters"]["xlabel"])
                    plt.tight_layout()
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()

            # ==========================================================================================================
            # Plotting single electrodes rasters:
            save_root = Path(save_path_fig, "response_tuning")
            if not os.path.isdir(save_root):
                os.makedirs(save_root)
            # Generate one directory per ROI:
            roi_dirs = {roi: None for roi in param.rois.keys()}
            for roi in roi_dirs.keys():
                roi_dirs[roi] = Path(save_root, roi)
                if not os.path.isdir(roi_dirs[roi]):
                    os.makedirs(roi_dirs[roi])
            # Adding the path to whatever isn't in the ROIs:
            roi_dirs["other"] = Path(save_root, "other")
            if not os.path.isdir(roi_dirs["other"]):
                os.makedirs(roi_dirs["other"])

            print("=" * 40)
            print("Plotting category tuning raster")
            for subject in epochs.keys():
                subject_results = results.loc[results["subject"] == subject]
                sub_epochs = epochs[subject]
                for channel in epochs[subject].ch_names:
                    # Check whether this electrode is significant:
                    rej = subject_results.loc[subject_results["channel"] == channel,
                                              "reject"].item()
                    if rej is not True:
                        continue

                    print("Plotting {} raster".format(channel))

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

                    ch_epochs = sub_epochs.copy().pick(channel)
                    # Get the favorite condition:
                    fav_cond = subject_results.loc[subject_results["channel"] == channel, "condition"].item()
                    # Get the dprime:
                    dprime = subject_results.loc[subject_results["channel"] == channel, "effect_strength"].item()
                    # Create file name:
                    file_prefix = Path(roi_dirs[channel_roi],
                                       fav_cond + "-" + param.files_prefix + channel +
                                       "_raster.png")

                    # Open figure:
                    fig, ax = plt.subplots(len(analysis_parameters["raster_parameters"]["subplots_cond"]),
                                           figsize=[20, 15])
                    # Get the data of the favorite condition and channel:
                    epochs_fav_cond = ch_epochs[fav_cond]
                    # First, compute the vmin and vmax of each sub conditions to set the limits:
                    extremes = [[np.percentile(epochs_fav_cond[cond].get_data(), 10),
                                 np.percentile(epochs_fav_cond[cond].get_data(), 90)]
                                for cond in analysis_parameters["raster_parameters"]["subplots_cond"]]
                    vmin, vmax = np.min(extremes), np.max(extremes)
                    # Looping through each condition:
                    for ind, cond in enumerate(analysis_parameters["raster_parameters"]["subplots_cond"]):
                        subplot_epochs = epochs_fav_cond[cond]
                        # Get the epochs order:
                        order, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels = \
                            sort_epochs(subplot_epochs.metadata,
                                        analysis_parameters["raster_parameters"]["sort_conditions"])
                        norm = MidpointNormalize(vmin=vmin,
                                                 midpoint=analysis_parameters[
                                                     "raster_parameters"]["midpoint"],
                                                 vmax=vmax)
                        # Plot a heat map of the sorted trials:
                        im = ax[ind].imshow(np.squeeze(subplot_epochs.get_data())[order], cmap=cmap, norm=norm,
                                            extent=[subplot_epochs.times[0], subplot_epochs.times[-1], 0,
                                                    len(order)],
                                            aspect="auto", origin='lower')
                        # Add the y labels:
                        y_labels = list(
                            cond_y_ticks_labels[analysis_parameters["raster_parameters"]["sort_conditions"][0]])
                        labels_loc = \
                            cond_y_ticks_pos[analysis_parameters["raster_parameters"]["sort_conditions"][0]]
                        ax[ind].set_yticks(labels_loc)
                        ax[ind].set_yticklabels(y_labels * int((len(labels_loc) / len(y_labels))), fontsize=9)
                        # Adding the vertical lines;
                        ax[ind].vlines(analysis_parameters["raster_parameters"]["v_lines"],
                                       ax[ind].get_ylim()[0], ax[ind].get_ylim()[1],
                                       linestyles='dashed', linewidth=1)
                        # Adding horizontal lines marking break between identities:
                        ax[ind].hlines(hline_pos[analysis_parameters["raster_parameters"]["sort_conditions"][0]],
                                       color="k", linewidth=1, linestyle=":", xmin=subplot_epochs.times[0],
                                       xmax=subplot_epochs.times[-1])
                        # Set only the relevant x ticks:
                        ax[ind].set_xticks(analysis_parameters["raster_parameters"]["v_lines"])
                        ax[ind].set_xticklabels([str(val)
                                                 for val in
                                                 analysis_parameters["raster_parameters"]["v_lines"]])
                        # Adding the title:
                        ax[ind].set_title(cond)

                    plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n" + " " +
                                 fav_cond + " dprime: {:.2f}".format(dprime))
                    plt.tight_layout()
                    fig.subplots_adjust(right=0.9)
                    # Add the color bar:
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label(analysis_parameters["raster_parameters"]["cbar_label"])
                    cbar.ax.yaxis.set_label_position('left')
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()

                    # Finally, plotting a histogram:
                    print("Plotting {} {} tuning histogram".format(channel, fav_cond))
                    id_mean_resp = {identity: None
                                    for identity in
                                    list(epochs_fav_cond.metadata[
                                             analysis_parameters["barplot_parameters"]["group_condition"]].unique())}
                    for identity in list(epochs_fav_cond.metadata[
                                             analysis_parameters["barplot_parameters"]["group_condition"]].unique()):
                        # Compute evoked to this one category:
                        id_mean_resp[identity] = \
                            np.mean(epochs_fav_cond[identity].average().crop(
                                tmin=analysis_parameters["barplot_parameters"]["time_win"][0],
                                tmax=analysis_parameters["barplot_parameters"]["time_win"][1]).get_data())
                    # Sorting the resulting dictionary:
                    id_mean_resp = {k: v for k, v in sorted(id_mean_resp.items(), key=lambda item: item[1],
                                                            reverse=True)}
                    # Plotting the data:
                    fig, ax = plt.subplots(figsize=[20, 15])
                    ax.bar(range(len(id_mean_resp)), list(id_mean_resp.values()), align='center')
                    ax.set_xticks(range(len(id_mean_resp)))
                    ax.set_xticklabels(list(id_mean_resp.keys()), fontsize=10)
                    ax.set_ylabel(analysis_parameters["barplot_parameters"]["ylabel"])
                    ax.set_title("{} mean activation to {} from {} to {}sec".
                                 format(channel, fav_cond,
                                        analysis_parameters["barplot_parameters"]["time_win"][0],
                                        analysis_parameters["barplot_parameters"]["time_win"][1]))
                    # Save the results:
                    file_prefix = Path(roi_dirs[channel_roi],
                                       fav_cond + "-" + param.files_prefix + channel + "_" +
                                       "_bar_plot.png")
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()


if __name__ == "__main__":
    configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")
    plot_category_selectivity(configs, save_folder="super")
