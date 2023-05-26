import os
import json
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import config
from scipy.stats import sem
import numpy as np
import pandas as pd
from pathlib import Path
from general_utilities import mean_confidence_interval, epochs_loader, get_channels_labels, corrected_sem
from plotters import plot_time_series, plot_rasters, mm2inch

param = config.param
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "visual_responsiveness"
sub = "super"
ses = "V1"
data_type = "ieeg"
results_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "results")
category_order = ["face", "object", "letter", "false"]
duration_order = ["500ms", "1000ms", "1500ms"]
vlines = [0, 0.5, 1.0, 1.5]
ylim = None
crop_time = [-0.05, 0.5]
smooth_time_ms = 50
sfreq = 512
# Convert to samples:
smooth_samp = int(smooth_time_ms * (sfreq / 1000))

anat_rois = {
    "Occ": ["ctx-lh-lateraloccipital",
            "ctx-lh-cuneus",
            "ctx-lh-pericalcarine",
            "ctx-rh-lateraloccipital",
            "ctx-rh-cuneus",
            "ctx-rh-pericalcarine"],
    "Par": ["ctx-lh-isthmuscingulate",
            "ctx-lh-precuneus",
            "ctx-lh-inferiorparietal",
            "ctx-lh-superiorparietal",
            "ctx-lh-supramarginal",
            "ctx-rh-isthmuscingulate",
            "ctx-rh-precuneus",
            "ctx-rh-inferiorparietal",
            "ctx-rh-superiorparietal",
            "ctx-rh-supramarginal"],
    "VT": ["ctx-lh-inferiortemporal",
           "ctx-lh-lingual",
           "ctx-lh-fusiform",
           "ctx-lh-parahippocampal",
           "ctx-lh-entorhinal",
           "ctx-rh-inferiortemporal",
           "ctx-rh-lingual",
           "ctx-rh-fusiform",
           "ctx-rh-parahippocampal",
           "ctx-rh-entorhinal"],
    "LT": ["ctx-lh-middletemporal",
           "ctx-lh-bankssts",
           "ctx-lh-transversetemporal",
           "ctx-lh-superiortemporal",
           "ctx-lh-temporalpole",
           "ctx-rh-middletemporal",
           "ctx-rh-bankssts",
           "ctx-rh-transversetemporal",
           "ctx-rh-superiortemporal",
           "ctx-rh-temporalpole"],
    "PFC": ["ctx-lh-caudalmiddlefrontal",
            "ctx-lh-superiorfrontal",
            "ctx-lh-parsopercularis",
            "ctx-lh-rostralmiddlefrontal",
            "ctx-lh-parstriangularis",
            "ctx-lh-parsorbitalis",
            "ctx-lh-lateralorbitofrontal",
            "ctx-lh-medialorbitofrontal",
            "ctx-lh-orbitofrontal",
            "ctx-lh-frontalpole",
            "ctx-lh-medialorbitofrontal",
            "ctx-lh-rostralanteriorcingulate",
            "ctx-lh-caudalanteriorcingulate",
            "ctx-rh-caudalmiddlefrontal",
            "ctx-rh-superiorfrontal",
            "ctx-rh-parsopercularis",
            "ctx-rh-rostralmiddlefrontal",
            "ctx-rh-parstriangularis",
            "ctx-rh-parsorbitalis",
            "ctx-rh-lateralorbitofrontal",
            "ctx-rh-medialorbitofrontal",
            "ctx-rh-orbitofrontal",
            "ctx-rh-frontalpole",
            "ctx-rh-medialorbitofrontal",
            "ctx-rh-rostralanteriorcingulate",
            "ctx-rh-caudalanteriorcingulate"],
    "SM": [
        "ctx-rh-precentral",
        "ctx-rh-postcentral",
        "ctx-rh-paracentral",
        "ctx-lh-precentral",
        "ctx-lh-postcentral",
        "ctx-lh-paracentral"
    ]
}

roi_colors = {
    "Occ": [0.6313725490196078, 0.788235294117647, 0.9568627450980393],
    "Par": [1.0, 0.7058823529411765, 0.5098039215686274],
    "VT": [0.5529411764705883, 0.8980392156862745, 0.6313725490196078],
    "LT": [0.8156862745098039, 0.7333333333333333, 1.0],
    "PFC": [255/255, 233/255, 0],
    "SM": [0.8705882352941177, 0.7333333333333333, 0.6078431372549019]
}
ylabel = "HGP (norm.)"
fig_size = param["figure_size_mm"]


def vis_resp_raster(epochs, cond_1, cond_2, file_name, conditions="category"):
    """

    :param epochs:
    :param cond_1:
    :param cond_2:
    :param file_name:
    :param conditions:
    :return:
    """
    print("=" * 40)
    print("Plotting rasters")
    for subject in epochs.keys():
        epoch = epochs[subject]
        for cond1 in cond_1:
            for cond2 in cond_2:
                epo = epoch.copy()["/".join([cond1, cond2])]
                metadata = epo.metadata
                conds = metadata[conditions].to_list()
                t0 = epo.times[0]
                tend = epo.times[-1]
                for channel in epo.ch_names:
                    data = np.squeeze(epo.get_data(picks=channel))
                    data = uniform_filter1d(np.array(data), smooth_samp, axis=-1)
                    filename = file_name.format("{}-{}".format(subject, channel), "raster", cond1, cond2)
                    plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=ylim, midpoint=1.0, transparency=1.0,
                                 xlabel="Time (s)", ylabel="Trials", cbar_label=ylabel, filename=filename,
                                 vlines=vlines, title=None, square_fig=False, conditions=conds,
                                 cond_order=category_order, crop_cbar=False)
                    plt.close()
        # Plot the entire raster:
        for channel in epoch.ch_names:
            data = np.squeeze(epoch.get_data(picks=channel))
            data = uniform_filter1d(np.array(data), smooth_samp, axis=-1)
            metadata = epoch.metadata
            conds = metadata[conditions].to_list()
            filename = file_name.format("{}-{}".format(subject, channel), "raster", "all", "all")
            plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=ylim, midpoint=1.0, transparency=1.0,
                         xlabel="Time (s)", ylabel="Trials", cbar_label=ylabel, filename=filename,
                         vlines=vlines, title=None, square_fig=False, conditions=conds,
                         cond_order=category_order, crop_cbar=False)
            plt.close()

    return None


def vis_resp_time_series(epochs, cond_1, cond_2, file_name, patches=None, conditions="category"):
    """

    :param epochs:
    :param cond_1:
    :param cond_2:
    :param file_name:
    :param patches:
    :param conditions:
    :return:
    """
    print("=" * 40)
    print("Plotting time series")
    for subject in epochs.keys():
        epoch = epochs[subject]
        for cond1 in cond_1:
            for cond2 in cond_2:
                epo = epoch.copy()["/".join([cond1, cond2])]
                metadata = epo.metadata
                t0 = epo.times[0]
                tend = epo.times[-1]
                for channel in epo.ch_names:
                    # Compute the mean and confidence interval for each category:
                    data = np.squeeze(epo.get_data(picks=channel))
                    avgs = []
                    data_sem = []
                    conds = []
                    for cond in list(metadata[conditions].unique()):
                        data_sem.append(data[np.where(metadata[conditions] == cond)[0]])
                        avg, error = mean_confidence_interval(data[np.where(metadata[conditions] == cond)[0]],
                                                              confidence=0.95, axis=0)
                        avgs.append(avg)
                        conds.append(cond)
                    # Compute the cousineau morey within subject SEM:
                    errors = corrected_sem(data_sem, len(data_sem))
                    # Get the colors:
                    colors = [param["colors"][cond] for cond in conds]
                    filename = file_name.format("{}-{}".format(subject, channel), "", cond1, cond2)
                    # Apply smoothing:
                    avgs = uniform_filter1d(np.array(avgs), smooth_samp, axis=-1)
                    errors = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errors]
                    plot_time_series(avgs, t0, tend, ax=None, err=errors, colors=colors, vlines=vlines,
                                     ylim=None, xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2,
                                     filename=filename, title=None, square_fig=False, conditions=conds, do_legend=False,
                                     patches=patches, patch_color="r", patch_transparency=0.2)
                    plt.close()
    return None


def roi_averages(epochs, rois, result_df, conds, file_name, aseg="aparc+aseg"):
    """
    This function averages the onset responsive electrodes per ROI
    :param epochs:
    :param rois:
    :param file_name:
    :param aseg:
    :return:
    """
    print("=" * 40)
    print("Plotting roi averages")
    for cond in conds:
        # Loop through each ROI:
        roi_names = []
        avgs = []
        errors = []
        t0 = None
        tend = None
        channels_roi = {}
        for roi in rois.keys():
            roi_labels = rois[roi]
            roi_evks = []
            for subject in epochs.keys():
                # Extract only the channels we are interested about for this subject:
                subject_channels = [ch.split("-")[1] for ch in
                                    result_df.loc[result_df["subject"] == subject, "channel"].to_list()]
                if len(subject_channels) == 0:
                    continue
                sub_epo = epochs[subject].copy().pick(subject_channels)
                # Get the labels of this subject's channels:
                subject_labels = get_channels_labels(sub_epo, subject,
                                                     Path(bids_root, "derivatives", "fs"),
                                                     aseg=aseg)
                # Get the channels that are within the ROI:
                picks = []
                for ind, row in subject_labels.iterrows():
                    ch_lbls = row["region"].split("/")
                    for lbl in ch_lbls:
                        if lbl in roi_labels:
                            picks.append(row["channel"])
                            channels_roi["-".join([subject, row["channel"]])] = roi
                            continue
                # Loop through each channel:
                for channel in picks:
                    roi_evks.append(sub_epo.copy()[cond].average(picks=channel).get_data())
                t0 = sub_epo.times[0]
                tend = sub_epo.times[-1]
            if len(roi_evks) == 0:  # If no channels were found in this ROI
                continue
            # Compute the average in this ROI and the error:
            avg, error = mean_confidence_interval(np.array(roi_evks),
                                                  confidence=0.95, axis=0)
            # Compute the cousineau morey within subject SEM to remove the between electrodes variance:
            error = sem(np.squeeze(np.array(roi_evks)), axis=0)
            avgs.append(np.squeeze(avg))
            errors.append(np.squeeze(error))
            roi_names.append(roi)
        # Convert the dict to a dataframe for easy handling:
        channels_roi = pd.DataFrame.from_dict(channels_roi, orient="index",
                                              columns=["roi"]).rename_axis("channel").reset_index()
        # Now, get the latency in each ROIs:
        lats = pd.DataFrame()
        for roi in channels_roi["roi"].unique():
            # Get a list of channels:
            channels = channels_roi.loc[channels_roi["roi"] == roi, "channel"].to_list()
            # Now extract the latencies:
            lats = lats.append(pd.DataFrame({
                "roi": [roi] * len(result_df.loc[result_df["channel"].isin(channels),
                                                 "latency-stimulus onset/" + cond].to_list()),
                "latency": result_df.loc[result_df["channel"].isin(channels),
                                         "latency-stimulus onset/" + cond].to_list()
            }))
        colors = [roi_colors[roi] for roi in roi_names]
        # Finally, plotting:
        fig, ax = plt.subplots(2, 1, figsize=[mm2inch(fig_size[0]) * 2, mm2inch(fig_size[0])* 2],
                               gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        # height_ratios=[2, 1], sharex=True)
        # Apply smoothing:
        avgs = uniform_filter1d(np.array(avgs), smooth_samp, axis=-1)
        errors = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errors]
        plot_time_series(avgs, t0, tend, ax=ax[0], err=errors, colors=colors, vlines=vlines, ylim=None,
                         xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=None,
                         title=None, square_fig=False, conditions=roi_names, do_legend=False,
                         patches=None, patch_color="r", patch_transparency=0.2)
        # Add the boxplot:
        ax[1].set_ylabel("")
        ax[1].set_xlabel("Time (s)")
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        sns.boxplot(x="latency", y="roi", data=lats, ax=ax[1], palette=colors, order=roi_names)
        plt.tight_layout()
        plt.savefig(file_name.format(cond.split("/")[-1]), transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        filename, file_extension = os.path.splitext(file_name.format(cond.split("/")[-1]))
        plt.savefig(filename + ".svg", transparent=True)
        plt.close()
    return None


def vis_resp_plot_handler(subjects, folders_list, save_root="", cond_to_plot="both"):
    """

    :param subjects:
    :param folders_list:
    :param save_root:
    :param cond_to_plot:
    :return:
    """
    # Loop through each subfolders:
    for folder in folders_list:
        results_path = Path(results_root, folder)
        # Loop through the subdirectories:
        subdirs = [x for x in results_path.iterdir() if x.is_dir()]
        for subdir in subdirs:
            # Find the files relevant for the plotting:
            results_files = []
            for file in glob.glob(str(Path(subdir, '*all_results.csv'))):
                results_files.append(file)
            confg_files = []
            for file in glob.glob(str(Path(subdir, '*.json'))):
                confg_files.append(file)
            if len(results_files) > 1:
                raise Exception("More than one file for visual responsiveness results!")

            # ==========================================================================
            # Load the data:
            vis_resp_results = pd.read_csv(results_files[0])
            with open(confg_files[0], 'r') as f:
                vis_resp_param = json.load(f)
            if subjects is None:
                subjects = list(vis_resp_results["subject"].unique())
            # Extract the relevant infos:
            vis_resp_results = vis_resp_results.loc[vis_resp_results["subject"].isin(subjects)]
            if cond_to_plot is not None:
                vis_resp_results = vis_resp_results.loc[vis_resp_results["condition"] == cond_to_plot]

            # Handle the epochs:
            epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                               "ses-" + ses, "ieeg", vis_resp_param["preprocessing_folder"],
                               vis_resp_param["analysis_parameters"][folder]["signal"],
                               vis_resp_param["preprocess_steps"]))
            epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
            epochs = epochs_loader(subjects, epo_dir, epo_file, vis_resp_results["channel"].to_list(), crop_time, ses,
                                   conditions=vis_resp_param["analysis_parameters"][folder]["conditions"])

            # Extract a couple extra info relevant for the plotting:
            time_windows = [vis_resp_param["analysis_parameters"][folder]["baseline_window"],
                            vis_resp_param["analysis_parameters"][folder]["test_window"]]
            cond_1 = vis_resp_param["analysis_parameters"][folder]["raster_parameters"]["conds_1"]
            cond_2 = vis_resp_param["analysis_parameters"][folder]["raster_parameters"]["conds_2"]

            # Create the save root:
            save_dir = Path(save_root, folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gen_file_name = "sub-{}_desc-{}_cond1-{}_cond2-{}.png"
            # Plot separately the task relevant and irrelevant:
            activated_results = vis_resp_results.loc[vis_resp_results["effect_strength-stimulus onset/Irrelevant"] > 0]
            roi_averages(epochs, anat_rois, activated_results, cond_1,
                         str(Path(save_dir, "grand_average_task-{}_activated.png")),
                         aseg="aparc+aseg")
            deactivated_results = vis_resp_results.loc[
                vis_resp_results["effect_strength-stimulus onset/Irrelevant"] < 0]
            roi_averages(epochs, anat_rois, deactivated_results, cond_1,
                         str(Path(save_dir, "grand_average_task-{}_deactivated.png")),
                         aseg="aparc+aseg")
            # Plot the visual responsiveness raster:
            time_series_dir = Path(save_dir, "time_series")
            if not os.path.exists(time_series_dir):
                os.makedirs(time_series_dir)
            rasters_dir = Path(save_dir, "raster")
            if not os.path.exists(rasters_dir):
                os.makedirs(rasters_dir)
            # vis_resp_raster(epochs, cond_1, cond_2, str(Path(rasters_dir, gen_file_name)), conditions="category")
            # Plot the time series:
            # vis_resp_time_series(epochs, cond_1, cond_2, str(Path(time_series_dir, gen_file_name)),
            #                      patches=time_windows,
            #                      conditions="category")


if __name__ == "__main__":
    subjects_list = None
    subfolders_list = ["high_gamma_wilcoxon_onset_two_tailed"]
    vis_resp_plot_handler(subjects_list, subfolders_list,
                          save_root="/hpc/users/alexander.lepauvre/plotting_test/vis_resp",
                          cond_to_plot="both")
