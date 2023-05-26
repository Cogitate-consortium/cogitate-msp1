"""
This script plots the results of the linear mixed models
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
from scipy.ndimage import gaussian_filter1d

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


def plotting_lmm_results(configs, save_folder="super"):
    """
    This function plots the results of the linear mixed model analysis.
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
            load_path_data = path_generator(param.save_root,
                                            analysis=analysis_name,
                                            preprocessing_steps=param.preprocess_steps,
                                            fig=False, data=True)
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)

            # Looping through each ROI:
            for roi in param.rois:
                # Finding the results file:
                lmm_results_file = find_files(load_path_results, "*" + roi + "_best_lmm_results", extension=".csv")
                assert len(lmm_results_file) == 1, "More than one file was found for rsa results!"
                # Get the data file:
                data_file = find_files(load_path_data, "*" + roi + "_lmm_data", extension=".csv")
                assert len(data_file) == 1, "More than one file was found for rsa results!"
                # Load the file:
                lmm_results = pd.read_csv(lmm_results_file[0])
                lmm_data = pd.read_csv(data_file[0])
                # Load the epochs of each subject:
                sub_epochs = {}
                for subject in lmm_results["subject"].unique():
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
                                    montage_space=param.montage_space,
                                    get_mni_coord=True,
                                    picks_roi=param.rois[roi],
                                    select_vis_resp=analysis_parameters["select_vis_resp"],
                                    vis_resp_folder=analysis_parameters["vis_resp_folder"],
                                    filtering_parameters=
                                    analysis_parameters["multitaper_parameters"]
                                    )
                    # Smoothing the data if needed:
                    if analysis_parameters["moving_average_ms"] is not None:
                        sub_epochs[subject] = epochs_mvavg(sub_epochs[subject],
                                                           analysis_parameters["moving_average_ms"])

                # Now looping through each model:
                for model in list(lmm_results["model"].unique()):
                    # Extracting the results of that model only:
                    model_lmm_results = lmm_results.loc[lmm_results["model"] == model]
                    # Generating the path to the save dir of that specific model:
                    model_fig_path = Path(save_path_fig, model)
                    if not os.path.isdir(model_fig_path):
                        os.makedirs(model_fig_path)
                    # ==============================================================================================
                    # Plotting model grand average:
                    colors = sns.color_palette("colorblind")
                    print("=" * 40)
                    print("Plotting single electrodes evoked responses")
                    file_prefix = Path(model_fig_path, "sig-" + param.files_prefix + model +
                                       "_grand_avg.png")
                    fig, ax = plt.subplots(len(analysis_parameters["evoked_parameters"]["conds_2"]),
                                           len(analysis_parameters["evoked_parameters"]["conds_1"]),
                                           figsize=[20, 15], sharey=True, sharex=True)
                    # Looping through each electrode of this model:
                    data = {cond_1: {cond_2: {cond_3: []
                                              for cond_3 in analysis_parameters["evoked_parameters"]["conds_3"]}
                                     for cond_2 in analysis_parameters["evoked_parameters"]["conds_2"]}
                            for cond_1 in analysis_parameters["evoked_parameters"]["conds_1"]}
                    for channel in list(model_lmm_results["group"].unique()):
                        # Get the participant ID:
                        sub_id = channel.split("-")[0]
                        ch_epochs = sub_epochs[sub_id].copy().pick(channel)
                        for ind_1, cond_1 in enumerate(analysis_parameters["evoked_parameters"]["conds_1"]):
                            # Get the data of the said condition and channel:
                            epochs_cond_1 = ch_epochs[cond_1]
                            for ind_2, cond_2 in enumerate(analysis_parameters["evoked_parameters"]["conds_2"]):
                                epochs_cond_2 = epochs_cond_1[cond_2]
                                for ind_3, cond_3 in enumerate(analysis_parameters["evoked_parameters"]["conds_3"]):
                                    data[cond_1][cond_2][cond_3].append(
                                        np.squeeze(epochs_cond_2[cond_3].average().get_data()))
                    for ind_1, cond_1 in enumerate(analysis_parameters["evoked_parameters"]["conds_1"]):
                        for ind_2, cond_2 in enumerate(analysis_parameters["evoked_parameters"]["conds_2"]):
                            for ind_3, cond_3 in enumerate(analysis_parameters["evoked_parameters"]["conds_3"]):
                                avg, low_ci, high_ci = \
                                    mean_confidence_interval(np.array(data[cond_1][cond_2][cond_3]),
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
                            ax[ind_2].set_xlim(ch_epochs.times[0], ch_epochs.times[-1])
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
                    # Set the super title and adjust to page:
                    plt.suptitle("Model grand average" + analysis_parameters["signal"] + " signal" + "\n"
                                 + "# channels=" + str(len(list(model_lmm_results["group"].unique()))))
                    plt.tight_layout()
                    # Save the figure:
                    plt.savefig(file_prefix, transparent=True)
                    plt.close()
                # Now looping through each model:
                for model in list(lmm_results["model"].unique()):
                    # Extracting the results of that model only:
                    model_lmm_results = lmm_results.loc[lmm_results["model"] == model]
                    # Generating the path to the save dir of that specific model:
                    model_fig_path = Path(save_path_fig, model)
                    # Looping through each channel of that model:
                    for channel in list(model_lmm_results["group"].unique()):
                        # Get the participant ID:
                        sub_id = channel.split("-")[0]

                        # ==============================================================================================
                        # Plotting evoked response across categories:
                        colors = sns.color_palette("colorblind")
                        print("=" * 40)
                        print("Plotting single electrodes evoked responses")
                        file_prefix = Path(model_fig_path, "sig-" + param.files_prefix + channel + "_" + model +
                                           "_across_cate_evoked.png")
                        fig, ax = plt.subplots(figsize=[20, 15], sharey=True, sharex=True)
                        for ind_1, cond_1 in enumerate(analysis_parameters["evoked_parameters"]["conds_1"]):
                            # Get the data of the said condition and channel:
                            epochs_cond_1 = ch_epochs[cond_1]
                            for ind_2, cond_2 in enumerate(analysis_parameters["evoked_parameters"]["conds_2"]):
                                epochs_cond_2 = epochs_cond_1[cond_2]
                                # Compute the mean and ci:
                                avg, low_ci, high_ci = \
                                    mean_confidence_interval(np.squeeze(epochs_cond_2.get_data()),
                                                             confidence=0.95)
                                # Plotting the average with the confidence interval:
                                ax.plot(ch_epochs.times, avg, color=colors[ind_2],
                                        label=cond_2)
                                ax.fill_between(ch_epochs.times, low_ci, high_ci,
                                                color=colors[ind_2], alpha=.2)
                            # Add the vertical lines:
                            # Adding the vertical lines;
                            ax.vlines(analysis_parameters["evoked_parameters"]["v_lines"],
                                      ax.get_ylim()[0], ax.get_ylim()[1],
                                      linestyles='dashed', linewidth=1.5, colors='k')
                            # Set only the relevant x ticks:
                            ax.set_xticks(analysis_parameters["evoked_parameters"]["v_lines"])
                            ax.set_xticklabels([str(val)
                                                for val in
                                                analysis_parameters["evoked_parameters"]["v_lines"]])
                            # Add the axis title:
                            ax.set_title(cond_1)
                            ax.legend()
                            ax.set_xlabel("Time (s)")
                            ax.set_xticks([])
                            # Set the y label:
                            ax.set_ylabel(analysis_parameters["evoked_parameters"]["ylabel"])
                        # Set the super title and adjust to page:
                        plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n" + model)
                        plt.tight_layout()
                        # Save the figure:
                        plt.savefig(file_prefix, transparent=True)
                        plt.close()

                        # ==============================================================================================
                        # Plotting raster:
                        # Get the data of that channel:
                        print("Plotting {} raster".format(channel))
                        ch_epochs = sub_epochs[sub_id].copy().pick(channel)
                        # Generate the file prefix:
                        file_prefix = Path(model_fig_path,
                                           "sig-" + param.files_prefix + channel + "_" + model + "_raster.png")
                        # Start creating the plot:
                        fig, ax = plt.subplots(len(analysis_parameters["raster_parameters"]["conds_2"]),
                                               len(analysis_parameters["raster_parameters"]["conds_1"]),
                                               figsize=[20, 15])
                        # Getting vmin and vmax:
                        extremes = []
                        for cond_1 in analysis_parameters["raster_parameters"]["conds_1"]:
                            for cond_2 in analysis_parameters["raster_parameters"]["conds_2"]:
                                extremes.append([np.percentile(ch_epochs[cond_1][cond_2].get_data(), 5),
                                                 np.percentile(ch_epochs[cond_1][cond_2].get_data(), 99)])
                        vmin, vmax = np.min(extremes), np.max(extremes)

                        # Plotting the conditions:
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
                                                            sigma=analysis_parameters["raster_parameters"][
                                                                "sigma"],
                                                            axis=-1, mode="nearest")
                                else:
                                    img = np.squeeze(epochs_cond_2.get_data())[order]
                                if vmax > analysis_parameters["raster_parameters"]["cmap_center"]:
                                    # Generate a cmap that will be centered on what we want:
                                    norm = MidpointNormalize(vmin=vmin,
                                                             midpoint=analysis_parameters[
                                                                 "raster_parameters"]["cmap_center"],
                                                             vmax=vmax)
                                else:
                                    # Generate a cmap that will be centered on what we want:
                                    norm = MidpointNormalize(vmin=vmin,
                                                             midpoint=vmax,
                                                             vmax=vmax)
                                # Plot a heat map of the sorted trials:
                                im = ax[ind_2].imshow(img,
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
                        # Add the color bar:
                        # Add the super title:
                        plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n" + model)
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
                        print("=" * 40)
                        print("Plotting single electrodes evoked responses")
                        file_prefix = Path(model_fig_path, "sig-" + param.files_prefix + channel + "_" + model +
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
                        # Set the super title and adjust to page:
                        plt.suptitle(channel + " " + analysis_parameters["signal"] + " signal" + "\n" + model)
                        plt.tight_layout()
                        # Save the figure:
                        plt.savefig(file_prefix, transparent=True)
                        plt.close()

    print("Done!")


if __name__ == "__main__":
    # Fetching all the config files:
    configs = find_files(Path(os.getcwd(), "lmm_configs"), naming_pattern="*", extension=".json")
    configs = [config for config in configs if "all_brain" not in config]
    plotting_lmm_results(configs, save_folder="super")
