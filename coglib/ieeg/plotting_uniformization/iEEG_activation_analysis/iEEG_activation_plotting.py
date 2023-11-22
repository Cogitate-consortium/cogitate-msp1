import pandas as pd
from pathlib import Path
import config
import theories_rois
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from general_utilities import epochs_loader, get_roi_channels, get_ch_rois, \
    load_fsaverage_coord
import seaborn as sns
from plotters import plot_time_series, plot_rasters, mm2inch
from scipy.stats import sem
from scipy.ndimage import uniform_filter1d

# get the parameters dictionary
param = config.param
fig_size = param["figure_size_mm"]
model_colors = {
    "time_win_dur_iit": [0.52, 0.86, 1],
    "time_win_dur_gnw": [
        0,
        1,
        0
    ],
    "time_win_dur_cate_iit": [
        0,
        0,
        1
    ],
    "time_win_dur_cate_gnw": [0.42, 1, 0.86],
}

cate_colors = {
    "face": [
        1,
        0,
        1
    ],
    "object": [
        173 / 255,
        80 / 255,
        29 / 255
    ],
    "letter": [
        57 / 255,
        115 / 255,
        132 / 255
    ],
    "false": [
        97 / 255,
        15 / 255,
        0 / 255
    ],
}

rois = theories_rois.rois
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "activation_analysis"
sub = "super"
ses = "V1"
data_type = "ieeg"
results_root = Path(bids_root, "derivatives", "activation_analysis", "sub-" + sub, "ses-" + ses, data_type, "results")
figures_root = Path(bids_root, "derivatives", "activation_analysis", "sub-" + sub, "ses-" + ses, data_type, "figure")
crop_time = [-0.5, 2.2]
smooth_time_ms = 50
sfreq = 512
# Convert to samples:
smooth_samp = int(smooth_time_ms * (sfreq / 1000))
category_order = ["face", "object", "letter", "false"]
duration_order_sec = ["1.5", "1.0", "0.5"]
duration_order_msec = ["1500ms", "1000ms", "500ms"]
dflt_ylim = None
vlines = [0., 0.5, 1., 1.5]
patches = [[0.8, 1.0], [1.3, 1.5], [1.8, 2.0]]


def plot_avg(epochs, cond_1, cond_2, file_name, colors, ylim=None, smoothing=0, img_width_fact=1,
             ylabel=""):

    colors = [colors[cond] for cond in cond_2]
    # Looping through each categories:
    for cond1 in cond_1:
        evks = []
        errs = []
        # Looping through each durations:
        for cond2 in cond_2:
            cond2_evk = []
            # Now looping through each subject:
            for subject in epochs.keys():
                # Compute the average for that particular subject:
                cond2_evk.append(epochs[subject].copy()["/".join([cond1, cond2])].average().get_data())
            # Now, compute the mean and confidence intervals:
            evks.append(np.mean(np.concatenate(cond2_evk), axis=0))
            errs.append(sem(np.concatenate(cond2_evk), axis=0))
        # Now name the file for condition 1:
        filename = file_name.format("all", "avg_ts", cond1)
        # Opend a figure:
        fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0] * img_width_fact),
                                        mm2inch(fig_size[1])])
        if smoothing is not None and smoothing > 0:
            evks = uniform_filter1d(np.array(evks), smoothing, axis=-1)
            errs = [uniform_filter1d(error, smoothing, axis=-1) for error in errs]
        plot_time_series(np.array(evks), crop_time[0], crop_time[1], err=errs, colors=colors, vlines=vlines, ylim=ylim,
                         ax=ax,
                         xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                         title=None, square_fig=False, conditions=cond_2, do_legend=False,
                         patches=None, patch_color="r", patch_transparency=0.2, x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
        plt.close()

    # Additionally, plotting across all categories:
    evks = []
    errs = []
    # Looping through each durations:
    for cond2 in cond_2:
        cond2_evk = []
        # Now looping through each subject:
        for subject in epochs.keys():
            # Compute the average for that particular subject:
            cond2_evk.append(epochs[subject].copy()[cond2].average().get_data())
        # Now, compute the mean and confidence intervals:
        evks.append(np.mean(np.concatenate(cond2_evk), axis=0))
        errs.append(sem(np.concatenate(cond2_evk), axis=0))
    # Now name the file for condition 1:
    filename = file_name.format("all", "avg_ts", "all")
    # Opend a figure:
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0] * img_width_fact),
                                    mm2inch(fig_size[1])])
    if smoothing is not None and smoothing > 0:
        evks = uniform_filter1d(np.array(evks), smoothing, axis=-1)
        errs = [uniform_filter1d(error, smoothing, axis=-1) for error in errs]
    plot_time_series(np.array(evks), crop_time[0], crop_time[1], err=errs, colors=colors, vlines=vlines, ylim=ylim,
                     ax=ax,
                     xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=cond_2, do_legend=False,
                     patches=None, patch_color="r", patch_transparency=0.2, x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    plt.close()

    # Now name the file for condition 1:
    filename = file_name.format("all", "avg_ts", "all_legend")
    # Open a figure:
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0] * img_width_fact),
                                    mm2inch(fig_size[1])])
    plot_time_series(np.array(evks), crop_time[0], crop_time[1], err=errs, colors=colors, vlines=vlines, ylim=ylim,
                     ax=ax,
                     xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=cond_2, do_legend=True,
                     patches=None, patch_color="r", patch_transparency=0.2)
    plt.close()

    return None


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


def plot_lmm_raster(epochs, file_name, colors, conditions_order=None, ylim=None, smoothing=0, crop_cb=False,
                    scaling=1, ylabel="", midpoint=1.0):
    """

    :param epochs:
    :param cond_1:
    :param file_name:
    :param conditions:
    :return:
    """

    if conditions_order is None:
        conditions_order = {"duration": ["1500ms", "1000ms", "500ms"],
                            "category": ["face", "object", "letter", "false"],
                            }
    print("=" * 40)
    print("Plotting rasters")
    for subject in epochs.keys():
        epoch = epochs[subject]
        metadata = epoch.metadata
        metadata["order"] = range(0, len(metadata))
        # Sort the data accordingly:
        metadata = sort_metadata(metadata, conditions_order)
        conds = metadata[list(conditions_order.keys())[1]].to_list()
        t0 = epoch.times[0]
        tend = epoch.times[-1]
        for channel in epoch.ch_names:
            if subject == "CE107" and channel == "O2PH16":
                ylim = [0.5, 2.0]
            else:
                ylim = ylim
            data = np.squeeze(epoch.get_data(picks=channel)) * scaling
            data = data[metadata["order"], :]
            if smoothing is not None and smoothing > 0:
                data = uniform_filter1d(data, smoothing, axis=-1)
            filename = file_name.format("{}-{}".format(subject, channel), "raster", "all")
            plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=ylim, midpoint=midpoint, transparency=1.0,
                         xlabel="Time (s)", ylabel="Trials", cbar_label=ylabel, filename=filename,
                         vlines=vlines, title=None, square_fig=False, conditions=conds,
                         cond_order=duration_order_msec, crop_cbar=crop_cb)
            plt.close()
            filename = file_name.format("{}-{}".format(subject, channel), "raster", "all_no_cb")
            plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=ylim, midpoint=midpoint, transparency=1.0,
                         xlabel="Time (s)", ylabel="Trials", cbar_label=ylabel, filename=filename,
                         vlines=vlines, title=None, square_fig=False, conditions=conds,
                         cond_order=duration_order_msec, do_cbar=False, add_cond_ticks=False, crop_cbar=crop_cb)
            plt.close()
            # Plot the average time series:
            avg = []
            err = []
            for dur in ["1500ms", "1000ms", "500ms"]:
                avg.append(np.squeeze(epoch.copy()[dur].average().get_data(picks=channel)) * scaling)
                err.append(sem(np.squeeze(epoch.copy()[dur].get_data(picks=channel) * scaling)))
            avg = np.array(avg)
            if smoothing is not None and smoothing > 0:
                avg = uniform_filter1d(avg, smoothing, axis=-1)
                err = [uniform_filter1d(val, smoothing, axis=-1) for val in err]
            col = [colors[cond] for cond in ["1500ms", "1000ms", "500ms"]]
            filename = file_name.format("{}-{}".format(subject, channel), "time_series", "no_leg")
            plot_time_series(avg, crop_time[0], crop_time[1], err=err, colors=col, vlines=vlines,
                             ylim=ylim,
                             xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                             title=None, square_fig=False, conditions=["1500ms", "1000ms", "500ms"], do_legend=False,
                             patches=None, patch_color="r", patch_transparency=0.2)
            plt.close()
            filename = file_name.format("{}-{}".format(subject, channel), "time_series", "leg")
            plot_time_series(avg, crop_time[0], crop_time[1], err=err, colors=col, vlines=vlines,
                             ylim=ylim,
                             xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                             title=None, square_fig=False, conditions=["1500ms", "1000ms", "500ms"], do_legend=True,
                             patches=None, patch_color="r", patch_transparency=0.2)
            plt.close()

    return None


def cousineau_sem(data, axis=0):

    # Compute the group average per time point:
    group_mean = np.mean(data, axis=axis)
    # Compute the within group mean:
    subject_mean = np.mean(data, axis=0)
    corrected_se = []
    for cond in range(data.shape[0]):
        # Normalize data by removing the within subject average to that condition to each subject's data and adding back
        # the grand mean:
        norm_data = data[cond, :, :] - subject_mean + group_mean
        # Compute the SEM
        corrected_se.append(sem(norm_data))

def reformat_lmm_results(df):
    """

    """
    # Get each unique model:
    mdls = list(df["model"].unique())
    return {mdl: list(df.loc[df["model"] == mdl, "group"].unique())
            for mdl in mdls}


def activation_plot_handler(folders_list, models, cat_sel_file, save_root=""):
    """

    :param models:
    :param folders_list:
    :param save_root:
    :return:
    """
    # Load the category selectivity:
    cate_sel_results_ti = pd.read_csv(cat_sel_file[0])
    cate_sel_results_tr = pd.read_csv(cat_sel_file[1])

    # Identify the channels that are category selective across both task relevance conditions:
    selective_channels = []
    for ch in cate_sel_results_ti["channel"].to_list():
        ti_sel = cate_sel_results_ti.loc[cate_sel_results_ti["channel"] == ch, "condition"].item()
        if ch in cate_sel_results_tr["channel"].to_list() and \
                cate_sel_results_tr.loc[cate_sel_results_tr["channel"] == ch, "condition"].item() == ti_sel:
            selective_channels.append(ch)

    cate_sel_results = cate_sel_results_ti.loc[cate_sel_results_ti["channel"].isin(selective_channels)]

    # Loop through each subfolders:
    for ind, folder in enumerate(folders_list):
        results_path = Path(results_root, folder)
        figures_path = Path(figures_root, folder)
        # Loop through the subdirectories:
        subdirs = [x for x in results_path.iterdir() if x.is_dir()]
        subdirs_fig = [x for x in figures_path.iterdir() if x.is_dir()]
        if "erp" in folder:
            baseline_mode = "mean"
            scaling = 10 ** 6
            ylabel = "ERP \u03BCV (corr.)"
            baseline_window = [-0.2, 0]
            midpoint = 0
        elif "gamma" in folder:
            baseline_mode = "ratio"
            scaling = 1
            ylabel = "HGP (norm.)"
            midpoint = 1
            baseline_window = [-0.5, -0.05]
        elif "alpha" in folder:
            baseline_mode = "ratio"
            scaling = 1
            ylabel = "Alpha (norm.)"
            baseline_window = [-0.5, -0.05]
            midpoint = 1
        else:
            raise Exception("The analyzed signal isn't recognized!")
        for ind2, subdir in enumerate(subdirs):
            # Find the files relevant for the plotting:
            results_files = []
            for file in glob.glob(str(Path(subdir, '*best_lmm_results.csv'))):
                results_files.append(file)
            confg_files = []
            for file in glob.glob(str(Path(subdir, '*.json'))):
                confg_files.append(file)
            with open(confg_files[0], 'r') as f:
                lmm_param = json.load(f)
            if len(results_files) > 1:
                raise Exception("More than one file for activation analysis results!")
            # Load the channels localization:
            channels_info_files = []
            for file in glob.glob(str(Path(subdirs_fig[ind2], '*channels_info.csv'))):
                channels_info_files.append(file)

            # ==========================================================================
            # Generate tables based on the channels models to plot with matlab:

            if not os.path.exists(Path(save_root, folder)):
                os.makedirs(Path(save_root, folder))
            # ===========================
            # Load the channels info:
            lmm_results = pd.read_csv(results_files[0])
            channels = list(lmm_results["group"].unique())
            channels_models = pd.DataFrame()
            for ch in channels:
                channels_models = channels_models.append(pd.DataFrame({
                    "channel": ch,
                    "model": lmm_results.loc[lmm_results["group"] == ch, "model"].to_list()[0]
                }, index=[0]), ignore_index=True)
            # Extract those for which we had the relevant models:
            channels_models = channels_models.loc[channels_models["model"].isin(models)]
            # Get a list of the channels:
            channels_list = channels_models["channel"].to_list()
            # Get a list of the subjects:
            subjects_list = list(set([ch.split("-")[0] for ch in channels_list]))
            if len(subjects_list) == 0:
                continue
            # Load the fsaverage coordinates:
            ch_coords = load_fsaverage_coord(bids_root, subjects_list, ses='V1', laplace_reloc=True)
            ch_coords = ch_coords.loc[ch_coords["name"].isin(channels_list)]
            ch_coords = ch_coords.rename(columns={"name": "channel"})
            ch_coords["radius"] = 2
            # Load the channels ROIs:
            ch_rois = get_ch_rois(bids_root, subjects_list, ses='V1', laplace_reloc=True)
            ch_rois = ch_rois.loc[ch_rois["channel"].isin(channels_list)]
            ch_rois = ch_rois.rename(columns={"region": "roi"})
            # Create another table storing the colors depending on the model:
            # Get the channels colors:
            ch_colors = pd.DataFrame({
                "channel": channels_list,
                "r": [model_colors[mdl][0] for mdl in channels_models["model"].to_list()],
                "g": [model_colors[mdl][1] for mdl in channels_models["model"].to_list()],
                "b": [model_colors[mdl][2] for mdl in channels_models["model"].to_list()],
            })
            for ch in ch_colors["channel"].to_list():
                if "cate" in channels_models.loc[channels_models["channel"] == ch, "model"].item():
                    if ch in cate_sel_results["channel"].to_list():
                        ch_colors.loc[ch_colors["channel"] == ch, ["r", "g", "b"]] = cate_colors[
                            cate_sel_results.loc[cate_sel_results["channel"] == ch, "condition"].item()
                        ]
                if ch == "CF107-O1" or ch == "CF113-RIT1" or ch == "CE107-O2PH16" or ch == "CF109-G45":
                    ch_coords.loc[ch_coords["channel"] == ch, "radius"] = 3

            # Save the data:
            ch_coords.to_csv(Path(save_root, folder, "coords.csv"))
            ch_colors.to_csv(Path(save_root, folder, "coords_colors.csv"))
            ch_rois.to_csv(Path(save_root, folder, "coords_rois.csv"))
            # Save the ROIs as well:
            rois_colors = pd.DataFrame({roi.replace("ctx_rh_", "").replace("ctx_lh_", ""):
                                            np.array(param["colors"][list(lmm_param["rois"].keys())[0]])
                                        for roi in lmm_param["rois"][list(lmm_param["rois"].keys())[0]]},
                                       index=['r', 'g', 'b']).T
            rois_colors['roi'] = rois_colors.index
            rois_colors = rois_colors.reset_index(drop=True)
            # Set a different color for the fusiform:
            rois_colors.loc[rois_colors["roi"] == "G_oc-temp_lat-fusifor", ["r", "g", "b"]] = list(
                np.array(param["colors"][list(lmm_param["rois"].keys())[0]]) * 1.8)
            rois_colors.to_csv(Path(save_root, folder, "rois_dict.csv"))

            # ==========================================================================
            # Load the data:
            # Extract the relevant infos and reformat:
            ch_model_dict = reformat_lmm_results(lmm_results.loc[lmm_results["model"].isin(models)])

            # Looping through each model:
            for mdl in ch_model_dict.keys():
                print(mdl)
                picks = ch_model_dict[mdl]  # Get each channels we have for this model
                # Extract each subject:
                subs = list(set([pick.split("-")[0] for pick in picks]))
                epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                                   "ses-" + ses, "ieeg", lmm_param["preprocessing_folder"],
                                   lmm_param["analysis_parameters"][folder]["signal"],
                                   lmm_param["preprocess_steps"]))
                epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
                epochs = \
                    epochs_loader(subs, epo_dir, epo_file, picks, crop_time, ses,
                                  conditions=lmm_param["analysis_parameters"][folder]["conditions"],
                                  filtering_parameters=lmm_param["analysis_parameters"][folder][
                                      "multitaper_parameters"], baseline_window=baseline_window,
                                  baseline_mode=baseline_mode)

                # =======================================================================
                # Plotting:
                # Create the save root:
                save_dir = Path(save_root, folder, mdl)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                gen_file_name = "sub-{}_desc-{}_cond-{}.png"
                # Set the colors:
                if "iit" in mdl:
                    colors = {
                        "500ms": [5 / 255, 53 / 255, 82 / 255],
                        "1000ms": [65 / 255, 153 / 255, 212 / 255],
                        "1500ms": [178 / 255, 226 / 255, 249 / 255]
                    }
                    if "alpha" in folder:
                        ylim = None
                        ts_lim = None
                    elif "gamma" in folder:
                        ylim = [0.5, 2.5]
                        ts_lim = [0.8, 2.5]
                    else:
                        ts_lim = None
                        ylim = None
                elif "gnw" in mdl:
                    colors = {
                        "500ms": [3 / 255, 81 / 255, 59 / 255],
                        "1000ms": [105 / 255, 195 / 255, 156 / 255],
                        "1500ms": [144 / 255, 210 / 255, 193 / 255]
                    }
                    if "alpha" in folder:
                        ts_lim = None
                        ylim = None
                    elif "gamma" in folder:
                        ts_lim = [0.8, 1.5]
                        ylim = [0.5, 1.5]
                    else:
                        ts_lim = None
                        ylim = None
                else:
                    continue
                # Plot the raster:
                plot_lmm_raster(epochs, str(Path(save_dir, gen_file_name)), colors,
                                conditions_order={
                                    "category": ["face", "object", "letter", "false"],
                                    "duration": ["1500ms", "1000ms", "500ms"],
                                },
                                ylim=ylim, smoothing=smooth_samp, crop_cb=True, scaling=scaling, ylabel=ylabel,
                                midpoint=midpoint)

                # Plot the data aggregated across channels for this model:
                plot_avg(epochs, category_order, duration_order_msec, str(Path(save_dir, gen_file_name)),
                         colors, ylim=ts_lim, smoothing=smooth_samp, ylabel=ylabel)

                # PLot separately per category selective electrodes:
                mdl_cat_sel = cate_sel_results.loc[cate_sel_results["channel"].isin(picks)]
                for cate in list(mdl_cat_sel["condition"].unique()):
                    if cate != cate:
                        continue
                    # Extract only the channels
                    cat_picks = mdl_cat_sel.loc[mdl_cat_sel["condition"] == cate, "channel"]
                    print(len(cat_picks))
                    # Create the subjects list:
                    subs = list(set([pick.split("-")[0] for pick in cat_picks]))
                    cat_sel_epo = {}
                    for s in subs:
                        sub_cat_picks = [pick.split("-")[1] for pick in cat_picks if s in pick]
                        cat_sel_epo[s] = epochs[s].copy().pick(sub_cat_picks)
                    # Extract the subject:
                    plot_avg(cat_sel_epo, category_order, duration_order_msec,
                             str(Path(save_dir, "sub-{}_desc-{}_cond-{}" + "_sel-{}.png".format(cate))),
                             colors, ylim=ts_lim, smoothing=smooth_samp, ylabel=ylabel)
                    # Plot the average per duration for the selective category with the average across duration for the
                    # other categories:
                    other_cate = [cat for cat in category_order if cat != cate]
                    evks = []
                    errs = []
                    # Looping through each durations:
                    for cond2 in duration_order_msec:
                        cond2_evk = []
                        # Now looping through each subject:
                        for subject in epochs.keys():
                            # Compute the average for that particular subject:
                            cond2_evk.append(
                                epochs[subject].copy()["/".join([cate, cond2])].average().get_data())
                        # Now, compute the mean and confidence intervals:
                        evks.append(np.mean(np.concatenate(cond2_evk), axis=0))
                        errs.append(sem(np.concatenate(cond2_evk), axis=0))
                    for cond_1 in other_cate:
                        cond_1_evk = []
                        # Now looping through each subject:
                        for subject in epochs.keys():
                            # Compute the average for that particular subject:
                            cond_1_evk.append(
                                epochs[subject].copy()[cond_1].average().get_data())
                        evks.append(np.mean(np.concatenate(cond_1_evk), axis=0))
                        errs.append(sem(np.concatenate(cond_1_evk), axis=0))
                    # Now name the file for condition 1:
                    filename = str(Path(save_dir, "sub-all_desc-_cond-all_sel-{}.png".format(cate)))
                    # Opend a figure:
                    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                                    mm2inch(fig_size[1])])
                    if smooth_samp is not None and smooth_samp > 0:
                        evks = uniform_filter1d(np.array(evks), smooth_samp, axis=-1)
                        errs = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errs]
                    c = [colors[key] for key in colors]
                    c.append([0.1, 0.1, 0.1])
                    c.append([0.4, 0.4, 0.4])
                    c.append([0.7, 0.7, 0.7])
                    conditions = ["500ms", "1000ms", "1500ms", *other_cate]
                    plot_time_series(np.array(evks), crop_time[0], crop_time[1], err=errs, colors=c, vlines=vlines,
                                     ylim=ts_lim,
                                     ax=ax,
                                     xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2, filename=filename,
                                     title=None, square_fig=False, conditions=conditions, do_legend=False,
                                     patches=None, patch_color="r", patch_transparency=0.2,
                                     x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
                    plt.close()

                    # Generate boxplots, showing the average amplitude in different time windows:
                    df = pd.DataFrame()
                    for s in cat_sel_epo.keys():
                        s_epo = cat_sel_epo[s]
                        for cat in category_order:
                            s_cat_epo = s_epo.copy()[cat]
                            s_cat_dur_epo = s_cat_epo.copy()["1500ms"]
                            for time_win in patches:
                                # Crop the data in this particular time window:
                                s_cat_dur_t_epo = \
                                    s_cat_dur_epo.copy().crop(tmin=time_win[0], tmax=time_win[1]).get_data()
                                avg = np.mean(s_cat_dur_t_epo, axis=(0, -1))
                                df = df.append(pd.DataFrame({
                                    "channel": s_epo.ch_names,
                                    "cate": [cat] * avg.shape[0],
                                    "dur": ["1500ms"] * avg.shape[0],
                                    "Time window": ["_".join([str(time_win[0]), str(time_win[1]) +
                                                              " sec"])] * avg.shape[0],
                                    ylabel: avg
                                }), ignore_index=True)

                    # Create bar plot:
                    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                                    mm2inch(fig_size[1])])
                    ax = sns.barplot(data=df, x="Time window", y=ylabel, hue="cate", ax=ax,
                                     palette=[param["colors"][cate] for cate in category_order])
                    ax.legend_.remove()
                    plt.tight_layout()
                    filename = str(Path(save_dir, "sub-all_desc-{}_selective_barplot.png".format(cate)))
                    plt.savefig(filename)
                    filename = str(Path(save_dir, "sub-all_desc-{}_selective_barplot.svg".format(cate)))
                    plt.savefig(filename)
                    plt.close()

                    # Create bar plot:
                    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                                    mm2inch(fig_size[1])])
                    sns.barplot(data=df, x="Time window", y=ylabel, hue="cate", ax=ax,
                                palette=[param["colors"][cate] for cate in category_order])
                    plt.tight_layout()
                    filename = str(Path(save_dir, "sub-all_desc-{}_selective_barplot_legend.png".format(cate)))
                    plt.savefig(filename)
                    filename = str(Path(save_dir, "sub-all_desc-{}_selective_barplot_legend.svg".format(cate)))
                    plt.savefig(filename)
                    plt.close()

    # Plot the control for GNW: plotting onset responsive channels
    vis_resp_results = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/" \
                       "visual_responsiveness/" \
                       "sub-super/ses-V1/ieeg/results/high_gamma_wilcoxon_onset_two_tailed/" \
                       "desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_analysis-vis_resp_sig_results.csv"
    vis_resp_results = pd.read_csv(vis_resp_results)
    # Further filter the results by extracting only the channels showing increase of responsiveness:
    vis_resp_results = vis_resp_results.loc[vis_resp_results["effect_strength-stimulus onset/Irrelevant"] > 0]
    # Get each channels showing onset responsiveness:
    picks = vis_resp_results["channel"].to_list()
    # Remove those that are category selective:
    picks = [pick for pick in picks if pick not in cate_sel_results["channel"].to_list()]
    # Further remove any channels showing category selectivity:
    subs = list(set([pick.split("-")[0] for pick in picks]))
    epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                       "ses-" + ses, "ieeg", "epoching",
                       "high_gamma", "desbadcharej_notfil_lapref"))
    epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
    epochs = epochs_loader(subs, epo_dir, epo_file, picks, crop_time, ses,
                           conditions="stimulus onset/Irrelevant",
                           filtering_parameters=None, baseline_window=[-0.5, -0.05],
                           baseline_mode="ratio")
    # Extract the channel in the GNW ROI:
    epochs = get_roi_channels(epochs, rois["gnw"], bids_root, aseg="aparc.a2009s+aseg")
    # Plot the data aggregated across channels for this model:
    save_dir = Path(save_root, "onset_resp_gnw")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colors = {
        "500ms": [0.1, 0.1, 0.1],
        "1000ms": [0.4, 0.4, 0.4],
        "1500ms": [0.7, 0.7, 0.7]
    }
    ts_lim = None

    # Plot the grand average:
    avgs = {dur: [] for dur in ["1500ms", "1000ms", "500ms"]}
    for subject in epochs.keys():
        # Loop through each channel:
        for ch in epochs[subject].ch_names:
            # Loop through each condition:
            for duration in ["1500ms", "1000ms", "500ms"]:
                # Compute the average within this condition:
                avgs[duration].append(np.squeeze(epochs[subject][duration].average(picks=ch).get_data()))
    # Compute the grand average:
    grand_avg = np.array([np.mean(np.array(avgs[cond]), axis=0) for cond in avgs.keys()])
    # Compute the errors:
    errors = []
    for cond in avgs.keys():
        errors.append(sem(avgs[cond], axis=0))
    grand_avg = uniform_filter1d(np.array(grand_avg), smooth_samp, axis=-1)
    errors = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errors]
    filename = Path(save_dir, "non_selective_grand_average.png")
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                    mm2inch(fig_size[1])])
    plot_time_series(grand_avg, crop_time[0], crop_time[1], err=errors,
                     colors=[colors[key] for key in colors.keys()], vlines=vlines, ylim=[0.95, 1.25],
                     ax=ax,
                     xlabel="Time (s)", ylabel="HGP (norm.)", err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=duration_order_msec, do_legend=False,
                     patches=None, patch_color="r", patch_transparency=0.2,
                     x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    plt.close()

    # Plot the average time series of the one electrodes from the GNW ROI that shows onset offset response:
    epo = epochs["CF104"]
    # Extract the one channel we are interested in:
    ch_epo = epo.pick("G12")
    dur_evk = []
    errs = []
    for dur in colors.keys():
        dur_evk.append(np.squeeze(ch_epo.copy()[dur].average().get_data()))
        errs.append(sem(np.squeeze(ch_epo.copy()[dur].get_data()), axis=0))

    # Apply smoothing:
    evks = uniform_filter1d(np.array(dur_evk), smooth_samp, axis=-1)
    errs = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errs]
    # Opend a figure:
    filename = Path(save_dir, "onset_offset_CF104-G12.png")
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                    mm2inch(fig_size[1])])
    plot_time_series(evks, crop_time[0], crop_time[1], err=errs,
                     colors=[colors[key] for key in colors.keys()], vlines=vlines, ylim=ts_lim,
                     ax=ax,
                     xlabel="Time (s)", ylabel="HGP (norm.)", err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=duration_order_msec, do_legend=False,
                     patches=None, patch_color="r", patch_transparency=0.2,
                     x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    plt.close()

    # Now name the file for condition 1:
    filename = Path(save_dir, "onset_offset_CF104-G12_lgd.png")
    # Opend a figure:
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                    mm2inch(fig_size[1])])
    plot_time_series(evks, crop_time[0], crop_time[1], err=errs,
                     colors=[colors[key] for key in colors.keys()], vlines=vlines, ylim=None,
                     ax=ax,
                     xlabel="Time (s)", ylabel="HGP (norm.)", err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=duration_order_msec, do_legend=True,
                     patches=None, patch_color="r", patch_transparency=0.2)
    plt.close()

    # Plot the average across face selective electrodes:
    face_selective_picks = cate_sel_results.loc[cate_sel_results["condition"] == "face", "channel"]
    # Further remove any channels showing category selectivity:
    subs = list(set([pick.split("-")[0] for pick in face_selective_picks]))
    # Load these channels:
    epochs = epochs_loader(subs, epo_dir, epo_file, face_selective_picks, crop_time, ses,
                           conditions="stimulus onset/Irrelevant",
                           filtering_parameters=None, baseline_window=[-0.5, -0.05],
                           baseline_mode="ratio")
    # Extract the channel in the GNW ROI:
    epochs = get_roi_channels(epochs, rois["gnw"], bids_root, aseg="aparc.a2009s+aseg")
    # Plot the grand average:
    avgs = {dur: [] for dur in ["1500ms", "1000ms", "500ms"]}
    for subject in epochs.keys():
        # Loop through each channel:
        for ch in epochs[subject].ch_names:
            # Loop through each condition:
            for duration in ["1500ms", "1000ms", "500ms"]:
                # Compute the average within this condition:
                avgs[duration].append(np.squeeze(epochs[subject][duration].average(picks=ch).get_data()))
    # Compute the grand average:
    grand_avg = np.array([np.mean(np.array(avgs[cond]), axis=0) for cond in avgs.keys()])
    # Compute the errors:
    errors = []
    for cond in avgs.keys():
        errors.append(sem(avgs[cond], axis=0))
    grand_avg = uniform_filter1d(np.array(grand_avg), smooth_samp, axis=-1)
    errors = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errors]
    filename = Path(save_dir, "face_selective_grand_average.png")
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                    mm2inch(fig_size[1])])
    plot_time_series(grand_avg, crop_time[0], crop_time[1], err=errors,
                     colors=[colors[key] for key in colors.keys()], vlines=vlines, ylim=ts_lim,
                     ax=ax,
                     xlabel="Time (s)", ylabel="HGP (norm.)", err_transparency=0.2, filename=filename,
                     title=None, square_fig=False, conditions=duration_order_msec, do_legend=False,
                     patches=None, patch_color="r", patch_transparency=0.2,
                     x_ticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    plt.close()


if __name__ == "__main__":
    subfolders_list = [
    ]
    models = ["time_win_dur_iit", "time_win_dur_gnw", "time_win_dur_cate_iit", "time_win_dur_cate_gnw"]
    category_selectivity_file = ["/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/" \
                                "category_selectivity/sub-super/ses-V1/ieeg/results/high_gamma_dprime_test_ti/" \
                                "desbadcharej_notfil_lapref/" \
                                "sub-super_ses-V1_task-Dur_analysis-category_selectivity_sig_results.csv",
                                 "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/" \
                                 "category_selectivity/sub-super/ses-V1/ieeg/results/high_gamma_dprime_test_tr/" \
                                 "desbadcharej_notfil_lapref/" \
                                 "sub-super_ses-V1_task-Dur_analysis-category_selectivity_sig_results.csv"
                                 ]

    activation_plot_handler(subfolders_list, models, category_selectivity_file,
                            save_root="/hpc/users/alexander.lepauvre/plotting_test/activation_analysis_newest")
