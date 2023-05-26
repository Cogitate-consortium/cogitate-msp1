"""
This script plots the results of the duration tracking analysis
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from general_helper_functions.data_general_utilities import mean_confidence_interval
from general_helper_functions.plotters import sort_epochs, MidpointNormalize
from Experiment1ActivationAnalysis.activation_analysis_helper_function import *
from general_helper_functions.pathHelperFunctions import find_files, path_generator
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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


def plotting_duration_tracking_results(configs, save_folder="super"):
    """
    This function plots the results of the duration tracking analysis.
    :param configs: (list of strings) list of the config files to use to plot
    :return:
    """
    # Looping through the configs:
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = DurationAnalysisParameters(config, sub_id=save_folder)
        # Looping through the different analysis performed in the visual responsiveness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create an evoked object to append to:
            load_path_results = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=False, stats=True)
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)

            # Looping through each ROI:
            for roi in param.rois:
                # Finding the results file:
                results_file = find_files(load_path_results, "*" + roi + "_tracking_proportion_stats",
                                          extension=".csv")
                assert len(results_file) == 1, "More than one file was found for rsa results!"
                # Load the file:
                duration_tracking_results = pd.read_csv(results_file[0])
                # Load the epochs of each subject:
                sub_epochs = {}
                for subject in duration_tracking_results["subject"].unique():
                    sub_epochs[subject], _ = \
                        load_epochs(param.BIDS_root,
                                    analysis_parameters["signal"],
                                    subject,
                                    session=param.session,
                                    task_name=param.task_name,
                                    preprocess_folder=param.preprocessing_folder,
                                    preprocess_steps=param.preprocess_steps,
                                    channel_types={"seeg": True, "ecog": True},
                                    condition=analysis_parameters["conditions"],
                                    baseline_method=analysis_parameters[
                                        "baseline_correction"],
                                    baseline_time=analysis_parameters[
                                        "baseline_time"],
                                    crop_time=analysis_parameters["crop_time"],
                                    aseg=param.aseg,
                                    montage_space="T1", get_mni_coord=True,
                                    picks_roi=param.rois[roi],
                                    filtering_parameters=
                                    analysis_parameters["multitaper_parameters"])
                    # Smoothing the data if needed:
                    if analysis_parameters["moving_average_ms"] is not None:
                        sub_epochs[subject] = epochs_mvavg(sub_epochs[subject],
                                                           analysis_parameters["moving_average_ms"])

                # Now looping through each channel:
                for channel in list(duration_tracking_results["channel"].unique()):
                    # Extracting the results of that model only:
                    ch_duration_tracking_results = \
                        duration_tracking_results.loc[duration_tracking_results["channel"] == channel]
                    # Generating the path to the save dir of that specific results:
                    if ch_duration_tracking_results["reject"].item():
                        fig_path = Path(save_path_fig, "duration_tracking")
                        if not os.path.isdir(fig_path):
                            os.makedirs(fig_path)
                    else:
                        fig_path = Path(save_path_fig, "no_tracking")
                        if not os.path.isdir(fig_path):
                            os.makedirs(fig_path)

                    # Get the participant ID:
                    sub_id = channel.split("-")[0]
                    # ==============================================================================================
                    # Plotting raster:
                    # Get the data of that channel:
                    print("Plotting {} raster".format(channel))
                    ch_epochs = sub_epochs[sub_id].copy().pick(channel)
                    # Generate the file prefix:
                    file_prefix = Path(fig_path,
                                       "sig-" + param.files_prefix + channel + "_raster.png")
                    # Start creating the plot:
                    fig, ax = plt.subplots(len(analysis_parameters["raster_parameters"]["conds_2"]),
                                           len(analysis_parameters["raster_parameters"]["conds_1"]),
                                           figsize=[20, 15])
                    # Getting vmin and vmax:
                    extremes = []
                    for cond_1 in analysis_parameters["raster_parameters"]["conds_1"]:
                        for cond_2 in analysis_parameters["raster_parameters"]["conds_2"]:
                            extremes.append([np.percentile(ch_epochs[cond_1][cond_2].get_data(), 5),
                                             np.percentile(ch_epochs[cond_1][cond_2].get_data(), 95)])
                    vmin, vmax = np.min(extremes), np.max(extremes)

                    # Plotting the conditions:
                    for ind_1, cond_1 in enumerate(analysis_parameters["raster_parameters"]["conds_1"]):
                        # Get the data of the said condition and channel:
                        epochs_cond_1 = ch_epochs[cond_1]
                        for ind_2, cond_2 in enumerate(analysis_parameters["raster_parameters"]["conds_2"]):
                            epochs_cond_2 = epochs_cond_1[cond_2]
                            # Get the epochs order:
                            if analysis_parameters["raster_parameters"]["sort_conditions"] is not None:
                                order, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels = \
                                    sort_epochs(epochs_cond_2.metadata,
                                                analysis_parameters["raster_parameters"]["sort_conditions"],
                                                order=analysis_parameters["raster_parameters"]["order"])
                                norm = MidpointNormalize(vmin=vmin,
                                                         midpoint=analysis_parameters[
                                                             "raster_parameters"]["cmap_center"],
                                                         vmax=vmax)
                                # Plot a heat map of the sorted trials:
                                im = ax[ind_2].imshow(np.squeeze(epochs_cond_2.get_data())[order],
                                                      cmap=cmap, norm=norm,
                                                      extent=[epochs_cond_2.times[0], epochs_cond_2.times[-1], 0,
                                                              len(order)],
                                                      aspect="auto", origin='lower')
                                # Add the y labels:
                                y_labels = list(
                                    cond_y_ticks_labels[analysis_parameters["raster_parameters"]["sort_conditions"][0]])
                                labels_loc = \
                                    cond_y_ticks_pos[analysis_parameters["raster_parameters"]["sort_conditions"][0]]
                                ax[ind_2].set_yticks(labels_loc)
                                ax[ind_2].set_yticklabels(y_labels * int((len(labels_loc) / len(y_labels))))
                                # Adding the vertical lines;
                                ax[ind_2].vlines(analysis_parameters["raster_parameters"]["v_lines"],
                                                 ax[ind_2].get_ylim()[0], ax[ind_2].get_ylim()[1],
                                                 linestyles='dashed', linewidth=1.5, colors='k')
                            else:
                                norm = MidpointNormalize(vmin=vmin,
                                                         midpoint=analysis_parameters[
                                                             "raster_parameters"]["cmap_center"],
                                                         vmax=vmax)
                                data = np.squeeze(epochs_cond_2.get_data())
                                im = ax[ind_2].imshow(data,
                                                      cmap="RdYlBu_r", norm=norm,
                                                      extent=[epochs_cond_2.times[0], epochs_cond_2.times[-1], 0,
                                                              data.shape[0]],
                                                      aspect="auto", origin='lower')
                            # Set only the relevant x ticks:
                            if ind_2 == 0:
                                ax[ind_2].set_xticks(analysis_parameters["raster_parameters"]["v_lines"])
                                ax[ind_2].set_xticklabels([str(val)
                                                           for val in
                                                           analysis_parameters["raster_parameters"][
                                                               "v_lines"]])
                            # Add the x axis legend only if we are on the last row:
                            if ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                ax[ind_2].set_xlabel("Time (s)")
                            else:
                                ax[ind_2].set_xticks([])
                            # Adding the title:
                            if ind_2 == 0:
                                ax[ind_2].set_title(cond_1 + "\n" + cond_2)
                            else:
                                ax[ind_2].set_title(cond_2)
                    # Add the super title:
                    if ch_duration_tracking_results["reject"].item():
                        plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n"
                                     + "duration tracking accuracy: {:.2f}".format(
                            ch_duration_tracking_results["tracking_accuracy"].item()))
                    else:
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

                    # ==============================================================================================
                    # Plotting evoked response:
                    colors = sns.color_palette("colorblind")
                    if  analysis_parameters["evoked_parameters"]["conds_3"] is not None:
                        print("=" * 40)
                        print("Plotting single electrodes evoked responses")
                        file_prefix = Path(fig_path, param.files_prefix + channel +
                                           "_evoked.png")
                        fig, ax = plt.subplots(len(analysis_parameters["evoked_parameters"]["conds_2"]),
                                               len(analysis_parameters["evoked_parameters"]["conds_1"]),
                                               figsize=[20, 15], sharey=True, sharex=True)
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
                                    ax[ind_2].plot(ch_epochs.times, avg, color=colors[ind_3],
                                                   label=cond_3)
                                    ax[ind_2].fill_between(ch_epochs.times, low_ci, high_ci,
                                                           color=colors[ind_3], alpha=.2)
                                # Add the vertical lines:
                                # Adding the vertical lines;
                                ax[ind_2].vlines(analysis_parameters["evoked_parameters"]["v_lines"],
                                                 ax[ind_2].get_ylim()[0], ax[ind_2].get_ylim()[1],
                                                 linestyles='dashed', linewidth=1.5, colors='k')
                                # Set only the relevant x ticks:
                                ax[ind_2].set_xticks(analysis_parameters["evoked_parameters"]["v_lines"])
                                ax[ind_2].set_xticklabels([str(val)
                                                           for val in
                                                           analysis_parameters["evoked_parameters"]["v_lines"]])
                                # Add the axis title:
                                if ind_2 == 0:
                                    ax[ind_2].set_title(cond_1 + "\n" + cond_2)
                                else:
                                    ax[ind_2].set_title(cond_2)
                                # Add the legend:
                                if ind_1 == len(analysis_parameters["evoked_parameters"]["conds_1"]) - 1 \
                                        and ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                    ax[ind_2].legend()
                                # Add the x axis legend only if we are on the last row:
                                if ind_2 == len(analysis_parameters["evoked_parameters"]["conds_2"]) - 1:
                                    ax[ind_2].set_xlabel("Time (s)")
                                else:
                                    ax[ind_2].set_xticks([])
                                # Set the y label:
                                ax[ind_2].set_ylabel(analysis_parameters["evoked_parameters"]["ylabel"])
                        if ch_duration_tracking_results["reject"].item():
                            plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n"
                                         + "duration tracking accuracy: {:.2f}".format(
                                ch_duration_tracking_results["tracking_accuracy"].item()))
                        else:
                            plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal")
                        plt.tight_layout()
                        # Save the figure:
                        plt.savefig(file_prefix, transparent=True)
                        plt.close()

    print("Done!")


if __name__ == "__main__":
    # Fetching all the config files:
    configs = find_files(Path(os.getcwd(), "duration_tracking_config"), naming_pattern="*", extension=".json")
    plotting_duration_tracking_results(configs, save_folder="super")
