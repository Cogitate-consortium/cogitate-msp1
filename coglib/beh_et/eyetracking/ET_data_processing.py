import pickle
import numpy as np
import os
import pandas as pd
import gc
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import math
import DataParser
import ET_param_manager
import ET_data_extraction
import itertools

"""
The following list is a list of participans whose ET data quality was so low we decided not to analyze their ET data.
This is mainly due to issues with the tracker onsite, that are also reported in those subjects' CRFs. 
"""
INVALID_LIST = ["SD119", "SE115", "SD156", "SG101", "SG102", "SG104", "SG105"]

TRIAL_INFO = "trial_info"
ET_DATA_DICT = "et_data_dict"
PARAMS = "params"
TIME_IN_EPOCH = "timeInEpoch"
CENTER_DIST_DEGS = "CenterDistDegs"
CENTER_DIST_PIX = "CenterDistPixels"
SIGNED = "Signed"
SUBJECT = "sub"
MODALITY = "mod"
LAB = "Lab"
BIN_PROPORTION = "bin_proportion"
RANGE_START_RAD = "range_starts_rad"
STIM_TYPES = ["object", "falseFont", "letter", "face"]
SACC_DIST_TO_CENTER = "saccade_dist_to_center"
STIMULUS_DURATIONS = [0.5, 1, 1.5]
WINDOWS = [DataParser.TRIAL, DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]

FACE = "face"
OBJ = "object"
LETTER = "letter"
FALF = "falseFont"

STIM_HUE_DICT = {FACE: "#003544", OBJ: "#ad501d", LETTER: "#397384", FALF: "#601f00"}
STIM_TITLE_DICT = {FACE: FACE.title(), OBJ: OBJ.title(), LETTER: LETTER.title(), FALF: "False Font"}
TASK_TITLE_DICT = {True: "Task Relevant", False: "Task Irrelevant"}

"""
We assume that regardless of the actual screen size, the visual stimulus was THE SAME SIZE IN VISUAL ANGLES
for ALL subjects (i.e., larger screens --> subjects sat farther). 
For fixation density, we are interested in a histogram showing the fixation distances (in VA) 
from the screen center and from points of interest in the screen (which all should be the SAME DISTANCE IN VA 
across subjects). Thus, when aggregating across labs with different screen sizes, we assume that all screens spanned 
the same VA SIZE. And so, for density, we want to BIN ALL SCREENS to the SAME NUMBER OF BINS across screens. 
Because each bin spans THE SAME VISUAL ANGLE. 
"""
FIXATION_DENSITY_XBINS = 96  # number of bins on the X axis to be calculated (width)
FIXATION_DENSITY_YBINS = 54  # number of bins on the Y axis to be calculated (height)
EXP1_STIM_VA = 6  # this is the AVERAGE size of a stimulus in experiment 1 in VISUAL ANGLES as per the registered report

# for the saccade amplitude over time - we smooth by calculating a rolling avg over window
ROLLING_SIZE_MS = 50  # in milliseconds!

# plotting
STIMULUS_DURATION_MAP = {0.5: "#1f968b", 1: "#33638d", 1.5: "#440154"}
STIMULUS_DURATION_NAME_MAP = {0.5: "Short", 1: "Medium", 1.5: "Long"}
# colors from https://github.com/Cogitate-consortium/plotting_uniformization/blob/main/config.py
STIMULUS_RELEVANCE_MAP = {True: (0.8352941176470589, 0.3686274509803922, 0.0), False: (0.5450980392156862, 0.16862745098039217, 0.8862745098039215)}
STIMULUS_RELEVANCE_NAME_MAP = {True: "Task Relevant", False: "Task Irrelevant"}

# fonts
TITLE_SIZE = 20
AXIS_SIZE = 19
TICK_SIZE = 17
LABEL_PAD = 8


def polar_plotter(data_df, hue_col, hue_col_name, hue_names_map, hue_color_map, title_name,
                  save_path, save_name, r_range=[0, 0.2, 0.05]):
    """
    Plot a polar DISTRIBUTION plot (radians).
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')

    """
    Theta should be range_start_rad + range_end_rads / 2 --> This is the MIDDLE between bins in RADIANS
    (we chose the middle between the bin start and end)
    r should be bin_proportion
    """
    for dur in list(data_df[hue_col].unique()):
        dur_df = data_df.loc[data_df[hue_col] == dur, :]
        range_starts_rad = list(dur_df["range_starts_rad"].values)
        range_end_rads = list(dur_df["range_end_rads"].values)
        # matplotlib's polarplot theta is in RADIANS, which is just like our data
        theta = [(range_starts_rad[ind] + range_end_rads[ind]) / 2 for ind in range(len(range_starts_rad))]
        r = list(dur_df["bin_proportion"].values)
        ax.plot(theta, r, c=hue_color_map[dur], linewidth=4, label=hue_names_map[dur])
        ax.grid(linewidth=3)

    ax.set_rmax(r_range[1])
    ax.set_rticks(np.arange(r_range[0], r_range[1], r_range[2]))
    plt.title(f"{title_name}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=hue_col_name)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)
    # free memory
    del figure
    gc.collect()
    return


def polar_plot_per_mod(data, title, hue_col, hue_col_name, hue_names_map, hue_color_map, save_path, save_name):
    modality_list = [ET_param_manager.ECOG, ET_param_manager.FMRI, ET_param_manager.MEG]

    for mod in modality_list:
        df = data[data[MODALITY] == mod]
        df.to_csv(os.path.join(save_path, f"{save_name}_{mod}.csv"), index=False)
        polar_plotter(data_df=df, hue_col=hue_col, hue_col_name=hue_col_name,
                      hue_names_map=hue_names_map, hue_color_map=hue_color_map,
                      title_name=f"{title} {mod}", save_path=save_path, save_name=f"{save_name}_{mod}",
                      r_range=[0, 0.2, 0.05])

    return


def line_plotter_ci(df, x_col, x_col_name, y_col, y_col_name, hue_col, hue_col_name, ci_col, title, save_path, save_name,
                 hue_names_map=None, palette=None, y_max=None, y_min=None, black_vertical_x=None, gray_vertical_x=None, y_ticks=None):
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    hue_groups = df[hue_col].unique()
    for group in hue_groups:
        df_group = df[df[hue_col] == group]
        color = palette[group]
        plt.plot(df_group[x_col], df_group[y_col], color=color)
        plt.fill_between(df_group[x_col], df_group[y_col]-df_group[ci_col], df_group[y_col]+df_group[ci_col],
                         alpha=0.2, edgecolor=color, facecolor=color)
    # add vertical lines
    if not(black_vertical_x is None):
        if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
            mi = y_min
            ma = y_max
        else:
            mi = df_group[y_col].min()
            ma = df_group[y_col].max()
        plt.vlines(x=black_vertical_x, ymin=mi, ymax=ma, colors='black', ls='--', lw=1)
    if not(gray_vertical_x is None):
        if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
            mi = y_min
            ma = y_max
        else:
            mi = df_group[y_col].min()
            ma = df_group[y_col].max()
        plt.vlines(x=gray_vertical_x, ymin=mi, ymax=ma, colors='gray', ls='--', lw=1)

    plt.title(f"{title}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.xticks(fontsize=TICK_SIZE)
    if y_max is not None and y_min is not None and not np.isnan(y_max) and not np.isnan(y_min):
        #plt.ylim(0.0,  0.5)
        plt.ylim(y_min, y_max)

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=TICK_SIZE)

    plt.locator_params(axis='y', nbins=6)
    plt.xlabel(x_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.ylabel(y_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)

    # markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    # new_labels = [hue_names_map[label] for label in palette]
    # legend = plt.legend(markers, new_labels, title=hue_col_name, markerscale=1, fontsize=TICK_SIZE - 2)
    # plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    plt.legend('', frameon=False)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)

    # now free some memory
    del figure
    plt.close()
    gc.collect()
    return


def line_plotter(df, x_col, x_col_name, y_col, y_col_name, hue_col, hue_col_name, title, save_path, save_name,
                 hue_names_map=None, palette=None, y_max=None, y_min=None):
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    if palette is not None:
        ax = sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar="se", palette=palette)  # STANDARD ERROR pad
    else:
        ax = sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar="se")  # STANDARD ERROR pad
    plt.title(f"{title}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.xticks(fontsize=TICK_SIZE)
    if y_max is not None and y_min is not None:
        plt.ylim(y_min, y_max)
    plt.yticks(fontsize=TICK_SIZE)
    plt.xlabel(x_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.ylabel(y_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    #markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    #new_labels = [hue_names_map[label] for label in palette]
    #legend = plt.legend(markers, new_labels, title=hue_col_name, markerscale=1, fontsize=TICK_SIZE - 2)
    #plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    plt.legend('', frameon=False)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)
    # now free some memory
    del figure
    plt.close()
    gc.collect()

    return


def calculate_within_ci(df, groupby_cols, hue_col, y_col):
    """
    Based of https://github.com/Cogitate-consortium/plotting_uniformization/blob/meg_oscar/MEG_activation/plot_spectral_activation.py,
    mean_ci_group function
    """
    for hue_value in df[hue_col].unique():
        df_hue = df.loc[df[hue_col] == hue_value, :]
        means_without_subs = df_hue.groupby(groupby_cols).mean().reset_index()
        means_with_subs = df_hue.groupby(groupby_cols + [SUBJECT]).mean().reset_index()
        for sub in df_hue[SUBJECT].unique():
            sub_y_col_mean = means_with_subs.loc[means_with_subs[SUBJECT] == sub, y_col]
            sbj_mean = sub_y_col_mean.mean()
            group_mean = means_without_subs[y_col].mean()
            within_y_col = sub_y_col_mean - sbj_mean + group_mean
            df_hue.loc[df_hue[SUBJECT] == sub, "y_col_tag"] = list(within_y_col)
        df.loc[df[hue_col] == hue_value, "y_col_tag"] = list(df_hue["y_col_tag"])

    df_ci = df.groupby(groupby_cols)["y_col_tag"].sem() * 1.96

    return df_ci


def line_plot_per_mod(data, x_col, x_col_name, y_col, y_col_name, hue_col, hue_col_name, title_name,
                      save_path, save_name, hue_names_map=None, palette=None, global_y_min=None, epsilon=None,
                      global_y_max=True, black_vertical_x=None, gray_vertical_x=None, y_lim_dict=None, y_ticks=None):
    modality_list = list(data[MODALITY].unique())
    """ DEPRECATED - FOR LINE_PLOTTER
    y_max = 0
    y_min = 100000000

    df_mean = data.groupby([MODALITY, hue_col, x_col]).mean(numeric_only=True).reset_index()
    for mod in modality_list:
        min_max_df = df_mean[df_mean[MODALITY] == mod]
        maxy = min_max_df[y_col].max()
        miny = min_max_df[y_col].min()
        if maxy > y_max:
            y_max = maxy
        if miny < y_min:
            y_min = miny

    if epsilon is not None:
        y_max += epsilon

    if global_y_min is not None:
        if y_min > global_y_min:
            y_min = global_y_min
    """
    for mod in modality_list:
        df = data[data[MODALITY] == mod]
        df_ci = calculate_within_ci(df, [x_col, hue_col], hue_col, y_col)
        df_mean = df.groupby([x_col, hue_col]).mean().reset_index()
        df_mean[f"{y_col}_CI_SE"] = list(df_ci)

        if y_lim_dict is None:
            y_max = max(df_mean[y_col] + df_mean[f"{y_col}_CI_SE"])
            y_min = min(df_mean[y_col] - df_mean[f"{y_col}_CI_SE"])
        else:
            y_min, y_max = y_lim_dict[mod]
            y_ticks_tag = y_ticks[mod]

        line_plotter_ci(df=df_mean, x_col=x_col, x_col_name=x_col_name, y_col=y_col, y_col_name=y_col_name,
                        hue_col=hue_col, hue_col_name=hue_col_name, ci_col=f"{y_col}_CI_SE",
                        title=f"{title_name} {mod}", save_path=save_path, save_name=f"{save_name}_{mod}",
                        hue_names_map=hue_names_map, palette=palette, y_max=y_max, y_min=y_min,
                        black_vertical_x=black_vertical_x, gray_vertical_x=gray_vertical_x, y_ticks=y_ticks_tag)

        df.to_csv(os.path.join(save_path, f"{save_name}_{mod}.csv"), index=False)
        """
        if not global_y_max:
            y_max = df[y_col].max()
        line_plotter(df=df, x_col=x_col, x_col_name=x_col_name, y_col=y_col, y_col_name=y_col_name,
                     hue_col=hue_col, hue_col_name=hue_col_name, title=f"{title_name} {mod}",
                     save_path=save_path, save_name=f"{save_name}_{mod}",
                     hue_names_map=hue_names_map, palette=palette, y_max=y_max, y_min=y_min)
        """

        # plot per lab

        """
        # take the modality's max / min
        min_max_df = df_mean[df_mean[MODALITY] == mod]
        maxy = min_max_df[y_col].max()
        miny = min_max_df[y_col].min()
        """
        """
        DEPRECATED
        for lab in df[LAB].unique():
            print(f"------LAB: {lab} -------")
            df_lab = df[df[LAB] == lab]
            df_ci = calculate_within_ci(df_lab, [x_col, hue_col], hue_col, y_col)
            df_mean = df_lab.groupby([x_col, hue_col]).mean().reset_index()
            df_mean[f"{y_col}_CI_SE"] = list(df_ci)
            df_lab.to_csv(os.path.join(save_path, f"{save_name}_{mod}_{lab}.csv"), index=False)
            y_max = max(df_mean[y_col] + df_mean[f"{y_col}_CI_SE"])
            y_min = min(df_mean[y_col] - df_mean[f"{y_col}_CI_SE"])
            line_plotter_ci(df=df_mean, x_col=x_col, x_col_name=x_col_name, y_col=y_col, y_col_name=y_col_name,
                            hue_col=hue_col, hue_col_name=hue_col_name, ci_col=f"{y_col}_CI_SE",
                            title=f"{title_name} {lab}", save_path=save_path, save_name=f"{save_name}_{mod}_{lab}",
                            hue_names_map=hue_names_map, palette=palette, y_max=y_max, y_min=y_min)
        """
    return


def calculate_sample_time_in_trial(relevant_trials, trial_samps, additional_cols):
    for index, trial in relevant_trials.iterrows():  # for each sample, calculate its time relative to the epoch (trial) start
        # The EPOCH starts *before* stimulus onset https://osf.io/gm3vd
        trial_samps.loc[trial_samps[DataParser.TRIAL] == trial[DataParser.TRIAL_NUMBER], TIME_IN_EPOCH] = trial_samps[
                                                                                                              DataParser.T_SAMPLE] - \
                                                                                                          trial[
                                                                                                              "EpochWindowStart"] - ET_param_manager.EPOCH_START
        for col in additional_cols:
            trial_samps.loc[trial_samps[DataParser.TRIAL] == trial[DataParser.TRIAL_NUMBER], col] = trial[col]

    """
    Example: SA125
    That subject (who was found to be a bit erroneous already at the prepro stage, but fixed - see comments there) 
    for some reason has excessive samples (i.e., more samples than defined in an epoch - exceeds the epoch end time 
    by 4 samples). 
    In order to ensure that time-based plots and stats begin at epoch start and end at epoch end, we trim:
    """
    trial_samps = trial_samps[trial_samps[TIME_IN_EPOCH] <= ET_param_manager.EPOCH_END]
    return trial_samps


def calculate_fixation_density(num_of_bins_x, num_of_bins_y, screen_dims, gaze_x, gaze_y):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin.
    NOTE: we assume that regardless of the actual screen size, the visual stimulus was THE SAME SIZE IN VISUAL ANGLES
    for ALL subjects (i.e., larger screens --> subjects sat farther).
    For fixation density, we are interested in a
    histogram showing the fixation distances (in VA) from the screen center and from points of interest in the screen
    (which all should be the SAME DISTANCE IN VA across subjects). Thus, when aggregating across labs with different
    screen sizes, we assume that all screens spanned the same VA SIZE. And so, for density, we want to BIN ALL SCREENS
    to the SAME NUMBER OF BINS across screens. Because each bin spans THE SAME VISUAL ANGLE.
    This function calculates fixation density based on a SET NUMBER OF BINS (num_of_bins_x * num_of_bins_y), which
    is expected to be the same across all subjects and screens.

    :param num_of_bins_x: Number of bins in the X axis (WIDTH) of the screen
    :param num_of_bins_y: Number of bins in the Y axis (HEIGHT) of the screen
    :param screen_dims: screen dims are described as [W, H]
    :param gaze_x: X coordinates of gaze
    :param gaze_y: Y coordinates of gaze
    :return: fixation density array
    """
    screen_rows = screen_dims[1]  # the screen HEIGHT is like "rows" in dataframe
    screen_cols = screen_dims[0]  # the screen WIDTH is like "columns"

    """
    Initialize the fixation density matrix: the bins' order is starting from the TOP LEFT and fills up accordingly!
    NOTE that TL -> BR is also the order of things in Eyelink data we parsed in ET_data_extraction, so this is ok.
    Gaze coordinates (X, Y) in Eyelink are such that (0, 0) is the TOP LEFT corner of the screen!!!
    This means that when gaze goes DOWN --> Y coordinate goes UP! 
    Source: EL1000 User manual 1.5 chapter 4.4.2.3 GAZE
    http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    """
    # fix_density = np.zeros((int(np.ceil(screen_rows / scale)), int(np.ceil(screen_cols / scale))))
    fix_density = np.zeros((num_of_bins_y, num_of_bins_x))
    scale_x = screen_cols / num_of_bins_x
    scale_y = screen_rows / num_of_bins_y

    # loop through the bins
    L = len(gaze_x)
    for i in range(0, fix_density.shape[1]):  # go over COLUMNS (screen WIDTH)
        for j in range(0, fix_density.shape[0]):  # go over ROWS (screen HEIGHT)
            if L == 0:  # no samples at all
                fix_density[j, i] = 0
            else:  # for each bin, we ask how many points fall in this bin
                bin_sum = np.sum(((gaze_x >= scale_x * i) & (gaze_x <= scale_x * (i + 1))) & (
                            (gaze_y >= scale_y * j) & (
                                gaze_y <= scale_y * (j + 1))))  # how many times did the gaze points hit this bin
                fix_density[j, i] = bin_sum / L  # normalize the sum
    return fix_density

def calculate_fixation_density_va(num_of_bins_x, num_of_bins_y, gaze_x, gaze_y, minimal_dims_va, params):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin.
    NOTE: we assume that regardless of the actual screen size, the visual stimulus was THE SAME SIZE IN VISUAL ANGLES
    for ALL subjects (i.e., larger screens --> subjects sat farther).
    For fixation density, we are interested in a
    histogram showing the fixation distances (in VA) from the screen center and from points of interest in the screen
    (which all should be the SAME DISTANCE IN VA across subjects). Thus, when aggregating across labs with different
    screen sizes, we assume that all screens spanned the same VA SIZE. And so, for density, we want to BIN ALL SCREENS
    to the SAME NUMBER OF BINS across screens. Because each bin spans THE SAME VISUAL ANGLE.
    This function calculates fixation density based on a SET NUMBER OF BINS (num_of_bins_x * num_of_bins_y), which
    is expected to be the same across all subjects and screens.

    :param num_of_bins_x: Number of bins in the X axis (WIDTH) of the screen
    :param num_of_bins_y: Number of bins in the Y axis (HEIGHT) of the screen
    :param screen_dims: screen dims are described as [W, H]
    :param gaze_x: X coordinates of gaze
    :param gaze_y: Y coordinates of gaze
    :return: fixation density array
    """
    screen_rows = minimal_dims_va[1]  # the screen HEIGHT is like "rows" in dataframe
    screen_cols = minimal_dims_va[0]  # the screen WIDTH is like "columns"

    screen_rows_real = params["ScreenResolution"][1] * params["DegreesPerPix"]
    screen_cols_real = params["ScreenResolution"][0] * params["DegreesPerPix"]
    print(f"screen rows(va): {screen_rows}, screen cols(va): {screen_cols}")
    print(f"screen rows real(va): {screen_rows_real}, screen real cols(va): {screen_cols_real}")

    """
    Initialize the fixation density matrix: the bins' order is starting from the TOP LEFT and fills up accordingly!
    NOTE that TL -> BR is also the order of things in Eyelink data we parsed in ET_data_extraction, so this is ok.
    Gaze coordinates (X, Y) in Eyelink are such that (0, 0) is the TOP LEFT corner of the screen!!!
    This means that when gaze goes DOWN --> Y coordinate goes UP! 
    Source: EL1000 User manual 1.5 chapter 4.4.2.3 GAZE
    http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
    """
    # fix_density = np.zeros((int(np.ceil(screen_rows / scale)), int(np.ceil(screen_cols / scale))))
    fix_density = np.zeros((num_of_bins_y, num_of_bins_x))
    scale_x = screen_cols / num_of_bins_x
    scale_y = screen_rows / num_of_bins_y
    x_start = (screen_cols_real - screen_cols) / 2
    y_start = (screen_rows_real - screen_rows) / 2
    # loop through the bins
    L = len(gaze_x)
    for i in range(0, fix_density.shape[1]):  # go over COLUMNS (screen WIDTH)
        for j in range(0, fix_density.shape[0]):  # go over ROWS (screen HEIGHT)
            if L == 0:  # no samples at all
                fix_density[j, i] = 0
            else:  # for each bin, we ask how many points fall in this bin
                bin_sum = np.sum(((gaze_x >= scale_x * i + x_start) & (gaze_x <= scale_x * (i + 1) + x_start)) &
                                 ((gaze_y >= scale_y * j + y_start) & (gaze_y <= scale_y * (j + 1) + y_start)))  # how many times did the gaze points hit this bin
                fix_density[j, i] = bin_sum / L  # normalize the sum
    return fix_density


def fixation_denisity_worker(df_samples, time_window, eye, screen_dims):
    """
    Calculates the fixation density across all trials in trial list, by using the samples in df_samples that belong
    to this trial are within the time window, and the dims in screen dims per eye
    :return: 1 df per eye of fixation density
    """
    # take only REAL fixations in the RELEVANT time window
    real_fixations = df_samples.loc[(df_samples[time_window] != -1) & (df_samples[DataParser.REAL_FIX] == True), :]
    sample_count = real_fixations.shape[0]
    if sample_count != 0:
        # LX, LY here are in PIXELS!!!
        x = np.array(real_fixations[f"{eye}X"])
        y = np.array(real_fixations[f"{eye}Y"])
        fix_density = calculate_fixation_density(FIXATION_DENSITY_XBINS, FIXATION_DENSITY_YBINS, screen_dims, x, y)
    else:
        fix_density = None
    return fix_density, sample_count


def fixation_denisity_worker_va(df_samples, time_window, eye, params, minimal_dims_va, in_va=False, filter_fix=True):
    """
    Calculates the fixation density across all trials in trial list, by using the samples in df_samples that belong
    to this trial are within the time window, and the dims in screen dims per eye
    :return: 1 df per eye of fixation density
    """
    # take only REAL fixations in the RELEVANT time window
    if filter_fix:
        real_fixations = df_samples.loc[(df_samples[time_window] != -1) & (df_samples[DataParser.REAL_FIX] == True), :]
    else:
        real_fixations = df_samples.loc[(df_samples[time_window] != -1) & (df_samples["is_missing"].isna()), :]
    sample_count = real_fixations.shape[0]
    if sample_count != 0:
        if not in_va:
            # LX, LY here are in PIXELS!!!
            """
            This is the version before the BL correction
            x = np.array(real_fixations[f"{eye}X"]) * params['DegreesPerPix']
            y = np.array(real_fixations[f"{eye}Y"]) * params['DegreesPerPix']
            """
            x = np.array(real_fixations[f"{eye}X{ET_data_extraction.BL_CORRECTED}"]) * params['DegreesPerPix']
            y = np.array(real_fixations[f"{eye}Y{ET_data_extraction.BL_CORRECTED}"]) * params['DegreesPerPix']
        else:
            x = np.array(real_fixations[f"{eye}X{ET_data_extraction.BL_CORRECTED}"])
            y = np.array(real_fixations[f"{eye}Y{ET_data_extraction.BL_CORRECTED}"])
        fix_density = calculate_fixation_density_va(FIXATION_DENSITY_XBINS, FIXATION_DENSITY_YBINS, x, y, minimal_dims_va, params)
    else:
        fix_density = None
    return fix_density, sample_count


def fix_hist_mod(subs_list, modality, time, save_path, minimal_dims_va, max_val=0, phase_name="", plot=False, square=True, im_name="exp1pic.jpg", in_va=False, filter_fix=True):
    """
    This is a WEIGHTED average across all subjects. Meaning, from each subject, we take the number of samples where
    a fixation occurred within a specific window. Then, all of these samples' locations are binned and presented
    across ALL subjects.

    As long as the screen resolution is dividable by the bins we chose, this should run.
    :param subs_list:
    :param modality:
    :param save_path:
    :return:
    """

    samples_counter = 0  # count how many samples are in that window
    density_array = None
    sub_per_lab = dict()
    for sub_data_path in subs_list:
        fl = open(sub_data_path, 'rb')
        sub_data = pickle.load(fl)
        fl.close()
        samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
        screen_dims = sub_data[PARAMS]['ScreenResolution']
        if sub_data[PARAMS][ET_param_manager.SUBJECT_LAB] not in sub_per_lab:
            sub_per_lab[sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]] = sub_data[PARAMS]
        # then, calculate the fixation density bins PER SUBJECT with fixation_denisity_worker
        sub_fix_density, sub_fix_count = fixation_denisity_worker_va(samples, time, sub_data[PARAMS][DataParser.EYE], sub_data[PARAMS], minimal_dims_va, in_va=in_va, filter_fix=filter_fix)
        if sub_fix_density is not None:
            if density_array is None:
                density_array = sub_fix_density * sub_fix_count
            else:
                density_array += sub_fix_density * sub_fix_count
        samples_counter += sub_fix_count

    density_array /= samples_counter

    if density_array.max() > max_val:
        max_val = density_array.max()

    # PLOT: density_array is the thing to plot - 2D histogram
    if plot:
        # plot
        plt.gcf()
        plt.figure()
        sns.reset_orig()

        ax = sns.heatmap(density_array, cmap="magma", alpha=0.75, cbar=True,
                         cbar_kws={"shrink": .82, "label": "Density"},
                         xticklabels=[], yticklabels=[],
                         annot=False, square=True, zorder=2, vmin=0, vmax=max_val, mask=density_array==0)
        # add (0, 0) lines
        ax.axvline(density_array.shape[1] / 2, color="white", lw=1)  # vertical line
        ax.axhline(density_array.shape[0] / 2, color="white", lw=1)  # horizontal line
        # number of ticks: unify for the journal plots s.t. they all have the same number of ticks
        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=8)
        ax.collections[0].colorbar.locator = tick_locator
        ax.collections[0].colorbar.update_ticks()

        if square:
            screen_rows = minimal_dims_va[1]  # the screen HEIGHT is like "rows" in dataframe
            screen_cols = minimal_dims_va[0]  # the screen WIDTH is like "columns"

            va_per_bin_x = (screen_cols / FIXATION_DENSITY_XBINS)
            va_per_bin_y = (screen_rows / FIXATION_DENSITY_YBINS)
            print(f"va_per_bin_x: {va_per_bin_x}, va_per_bin_y: {va_per_bin_y}")

            """
            CREATE A 6 X 6 VA RECTANGLE around fixation by using patches.Rectangle 
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
            
            The size is EXP1_STIM_VA x EXP1_STIM_VA around the screen center -- meaning, it spans (EXP1_STIM_VA/2) from
            the center in each direction
            """
            point_y = (density_array.shape[0] / 2) - ((EXP1_STIM_VA/2) / va_per_bin_y)
            point_x = (density_array.shape[1] / 2) - ((EXP1_STIM_VA/2) / va_per_bin_x)
            rect = patches.Rectangle((point_x, point_y), (EXP1_STIM_VA / va_per_bin_x), (EXP1_STIM_VA / va_per_bin_y), linewidth=2, edgecolor='r', fill=False, alpha=0.8, zorder=3)
            print(f"point_x: {point_x}, point_y:{point_y}, xlen: {(EXP1_STIM_VA / va_per_bin_x)}, ylen: {(EXP1_STIM_VA / va_per_bin_y)}")
            ax.add_patch(rect)

        if not(im_name is None):  # background image
            im_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), im_name)
            arr_img = plt.imread(im_path, format='jpg')
            ax.imshow(arr_img, aspect=ax.get_aspect(), extent=ax.get_xlim() + ax.get_ylim(), zorder=1, alpha=0.95)

        plt.title(f"Fixation Density Plot: {time} - {modality}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)

        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        if square:
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{phase_name}_square.png"), dpi=1000)
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{phase_name}_square.svg"), format="svg", dpi=1000)
        else:
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{phase_name}.png"), dpi=1000)
            plt.savefig(os.path.join(save_path, f"fix_dens_{time}_{modality}_{phase_name}.svg"), format="svg", dpi=1000)

        # save memory
        del figure
        plt.close()
        gc.collect()

    return 0


def fix_dist_from_center(subs_dict, save_path, do_each_category=False, group_with=DataParser.STIM_DUR_PLND_SEC):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    if group_with == DataParser.STIM_DUR_PLND_SEC:
        hue_col = group_with
        hue_col_name = "Stimulus Duration"
        save_name = f"fix_dist_line"
        hue_names_map = STIMULUS_DURATION_NAME_MAP
        palette = STIMULUS_DURATION_MAP
    else:
        hue_col = group_with
        hue_col_name = "Stimulus Relevance"
        save_name = f"fix_dist_line_tr"
        hue_names_map = STIMULUS_RELEVANCE_NAME_MAP
        palette = STIMULUS_RELEVANCE_MAP

    all_dfs_list = []
    for modality in subs_dict.keys():
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            trial_fix_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_fix_samples = trial_fix_samples.loc[trial_fix_samples[DataParser.REAL_FIX] == True, :]  # only REAL fixations
            trial_fix_samples = trial_fix_samples.loc[trial_fix_samples[DataParser.TRIAL] != -1,
                                :]  # ones that are within a TRIAL epoch (not between trials)
            analyzed_eye = sub_data[PARAMS][DataParser.EYE]
            trials = sub_data[TRIAL_INFO]
            trial_fix_samples = calculate_sample_time_in_trial(trials, trial_fix_samples, [DataParser.STIM_DUR_PLND_SEC,
                                                                                           DataParser.IS_TASK_RELEVANT_COL,
                                                                                           DataParser.STIM_TYPE_COL])

            # collapse across stimulus duration groups, at each timepoint (from epoch start to end) average
            if not do_each_category:
                trial_means = trial_fix_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, f"{analyzed_eye}{CENTER_DIST_DEGS}"]]
            else:
                trial_means = trial_fix_samples.groupby([TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL,
                                                  f"{analyzed_eye}{CENTER_DIST_DEGS}"]]

            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            trial_means.rename(columns={f"{analyzed_eye}{CENTER_DIST_DEGS}": f"{CENTER_DIST_DEGS}"}, inplace=True)
            only_mod_dfs_list.append(trial_means)  # add this subject data to the total
        # summarize this modality - average across all subjects' averages (and calculate std)
        mod_mean_df = pd.concat(only_mod_dfs_list)
        all_dfs_list.append(mod_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(all_dfs_list)

    if not do_each_category:
        line_plot_per_mod(data=all_mean_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=CENTER_DIST_DEGS, y_col_name="Average Distance from Center (Degrees VA)",
                          hue_col=group_with, hue_col_name=hue_col_name,
                          title_name="Average Fixation Distance from Center in Trial",
                          save_path=save_path, save_name=save_name, global_y_max=False,
                          hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP,
                          global_y_min=0, epsilon=0.1)
    else:
        for category in STIM_TYPES:
            df_cat = all_mean_df[all_mean_df[DataParser.STIM_TYPE_COL] == category]
            line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                              y_col=CENTER_DIST_DEGS, global_y_max=False,
                              y_col_name="Average Distance from Center (Degrees VA)",
                              hue_col=hue_col, hue_col_name=hue_col_name,
                              title_name=f"Average Fixation Distance from Center in **{category}** Trials",
                              save_path=save_path, save_name=f"{save_name}_{category}",
                              hue_names_map=hue_names_map, palette=palette,
                              global_y_min=0, epsilon=0.1)

    return


def fixation_dist_hist_mod(subs_list, modality, save_path, num_of_bins=100, bin_min=-10.0, bin_max=10.0):
    """
    This methods prepares the data for plotting a histogram where the Y = trial counts and the X = distance of fixation
    from the center of the screen
    :param subs_list:
    :param modality:
    :param save_path:
    :param num_of_bins:
    :param bin_min:
    :param bin_max:
    :return:
    """
    for axis in ["X", "Y"]:
        axis_dfs_list = []
        for sub_data_path in subs_list:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            eye = sub_data[PARAMS][DataParser.EYE]
            trials_means = sub_data[TRIAL_INFO][f"{axis}{DataParser.TRIAL}{CENTER_DIST_DEGS}{SIGNED}"]

            cdat = trials_means.to_numpy()
            cdat = cdat[~np.isnan(cdat)]
            chist = np.histogram(cdat, bins=num_of_bins, range=(bin_min, bin_max))  # chist[0] = frequency, chist[1] = bin value

            freq_data = pd.DataFrame()
            freq_data['Bin Value'] = chist[1]  # this has 101 values (as they define bin limits)
            freq_data[f"{axis}{CENTER_DIST_DEGS}{SIGNED}"] = np.concatenate((np.array([np.nan]), chist[0]), axis=0)
            freq_data[SUBJECT] = sub_data[PARAMS]['SubjectName']
            freq_data[LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            # TODO: Check why FMRI subject degs are SKYROCKET HIGH
            axis_dfs_list.append(freq_data)

        axis_df = pd.concat(axis_dfs_list)
        axis_mean = axis_df.groupby(['Bin Value', LAB]).mean(numeric_only=True).reset_index()
        # TODO: PLOT AXIS_MEAN HISTOGRAM
        x = 5
    return


def poc_helper(samps, saccs, eye):
    saccs = saccs[saccs[DataParser.EYE] == eye]  # get rid of the non-analyzed eye
    ek_sacc_no_blink = saccs.loc[saccs[f"{DataParser.HERSHMAN_PAD}"] == False, :]

    for sacc in ek_sacc_no_blink.itertuples():
        samps.loc[samps[DataParser.T_SAMPLE].between(sacc.tStart,
                                                     sacc.tEnd), SACC_DIST_TO_CENTER] = sacc.saccade_dist_to_center
    return samps


def roller_plotter(df, x_col, x_col_name, avg_col, avg_col_name, lower_col, upper_col,
                   hue_col, hue_col_name, palette, hue_names_map, title, y_min, y_max, save_path, save_name):
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for hue_name in hue_names_map:
        hue_color = palette[hue_name]
        df_hue = df[df[hue_col] == hue_name]
        plt.plot(df_hue[x_col], df_hue[avg_col], ls="-", color=hue_color, label=hue_name)
        plt.fill_between(df_hue[x_col], df_hue[lower_col], df_hue[upper_col], alpha=0.2, color=hue_color)

    plt.title(f"{title}", fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.xticks(fontsize=TICK_SIZE)
    if y_max is not None and y_min is not None:
        plt.ylim(y_min, y_max)
    plt.yticks(fontsize=TICK_SIZE)
    plt.xlabel(x_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.ylabel(avg_col_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    new_labels = [hue_names_map[label] for label in palette]
    legend = plt.legend(markers, new_labels, title=hue_col_name, markerscale=1, fontsize=TICK_SIZE - 2)
    plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)

    # save
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)

    # now free some memory
    del figure
    plt.close()
    gc.collect()
    return


def rolling_plot_per_mod(data, x_col, x_col_name, avg_col, avg_col_name, se_col, hue_col, hue_col_name, hue_names_map,
                         palette, title_name, save_path, save_name):
    modality_list = [ET_param_manager.ECOG, ET_param_manager.FMRI, ET_param_manager.MEG]

    # get the lower and upper SE values for avg_col
    data[se_col].fillna(0, inplace=True)  # replace nans with 0s
    data.loc[:, "lower"] = [y - se for y, se in zip(data[avg_col], data[se_col])]
    data.loc[:, "upper"] = [y + se for y, se in zip(data[avg_col], data[se_col])]

    y_max = np.nanmax(data["upper"].values)
    y_min = np.nanmin(data["lower"].values)

    for mod in modality_list:
        df = data[data[MODALITY] == mod]
        df.to_csv(os.path.join(save_path, f"{save_name}_{mod}.csv"), index=False)
        # this is a correction for nulls:
        for hue in data[hue_col].unique():
            ind = df[hue_col] == hue
            df.loc[ind, [avg_col, "lower", "upper"]] = df.loc[ind, [avg_col, "lower", "upper"]].bfill()

        roller_plotter(df=df, x_col=x_col, x_col_name=x_col_name, avg_col=avg_col, avg_col_name=avg_col_name,
                       lower_col="lower", upper_col="upper", hue_col=hue_col, hue_col_name=hue_col_name,
                       palette=palette,
                       hue_names_map=hue_names_map, title=f"{title_name} {mod}", y_min=y_min, y_max=y_max,
                       save_path=save_path, save_name=f"{save_name}_{mod}")

    return


def rolling_plot_per_lab(data, x_col, x_col_name, avg_col, avg_col_name, se_col, hue_col, hue_col_name, hue_names_map,
                         palette, title_name, save_path, save_name):
    # get the lower and upper SE values for avg_col
    data[se_col].fillna(0, inplace=True)  # replace nans with 0s
    data.loc[:, "lower"] = [y - se for y, se in zip(data[avg_col], data[se_col])]
    data.loc[:, "upper"] = [y + se for y, se in zip(data[avg_col], data[se_col])]

    y_max = np.nanmax(data["upper"].values)
    y_min = np.nanmin(data["lower"].values)

    for lab in list(data[LAB].unique()):
        df = data[data[LAB] == lab]
        df.to_csv(os.path.join(save_path, f"{save_name}_{lab}.csv"), index=False)
        # this is a correction for nulls:
        for hue in data[hue_col].unique():
            ind = df[hue_col] == hue
            df.loc[ind, [avg_col, "lower", "upper"]] = df.loc[ind, [avg_col, "lower", "upper"]].bfill()

        roller_plotter(df=df, x_col=x_col, x_col_name=x_col_name, avg_col=avg_col, avg_col_name=avg_col_name,
                       lower_col="lower", upper_col="upper", hue_col=hue_col, hue_col_name=hue_col_name,
                       palette=palette,
                       hue_names_map=hue_names_map, title=f"{title_name} {lab}", y_min=y_min, y_max=y_max,
                       save_path=save_path, save_name=f"{save_name}_{lab}")

    return


def saccade_amp_no_category(subs_dict, save_path, phase_name):
    """
    This method calculates the average saccade amplitude PER *** SUBJECT X CATEGORY X DURATION ***

    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    modality_sampling_rate = {ET_param_manager.ECOG: None, ET_param_manager.FMRI: None, ET_param_manager.MEG: None}

    total_df_list = []
    for modality in list(subs_dict.keys()):
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            sub_sampling_rate = sub_data[PARAMS][
                DataParser.SAMPLING_FREQ]  # subject's sampling rate - for rolling avg
            if modality_sampling_rate[
                modality] is None:  # THIS ASSUMES WE HAVE THE SAME SAMPLING RATE WITHIN A MODALITY!!!
                modality_sampling_rate[modality] = sub_sampling_rate
            trial_sacc_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.REAL_SACC] == True,
                                 :]  # only REAL saccades
            trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL] != -1,
                                 :]  # that are within trials
            relevant_trials = sub_data[TRIAL_INFO]
            trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL].isin(
                list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
            trial_sacc_samples = calculate_sample_time_in_trial(relevant_trials, trial_sacc_samples,
                                                                [DataParser.STIM_DUR_PLND_SEC])

            # calculate for each time point the MEAN SACCADE AMPLITUDE
            trial_means = trial_sacc_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                numeric_only=True).reset_index()
            trial_means = trial_means.loc[:, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, DataParser.AMP_DEG]]
            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            """
            Raw data is noisy - for plotting purposes only, let's smooth the data over a window of ROLLING_SIZE_MS.
            For that, we will use pandas' rolling average method, to create a new "rolling" column containing the
            smoothed data. This method is also called SMA (standard moving average).
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

            for dur in STIMULUS_DURATIONS:
                dur_means = trial_means[trial_means[DataParser.STIM_DUR_PLND_SEC] == dur]
                rolling_data = dur_means.rolling(int((ROLLING_SIZE_MS/1000)/(1/sub_sampling_rate))).mean(numeric_only=True)
                dur_means.loc[:, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]
                only_mod_dfs_list.append(dur_means)
            """
            only_mod_dfs_list.append(trial_means)

        # AVERAGE all subjects in this modality
        mod_mean_df = pd.concat(only_mod_dfs_list)
        total_df_list.append(mod_mean_df)

    all_sub_means_df = pd.concat(total_df_list)

    """
    Now, all categories at once
    """
    df_mod_ci_list = []
    for modality in all_sub_means_df[MODALITY].unique():
        df_mod = all_sub_means_df[all_sub_means_df[MODALITY] == modality]
        df_mod_ci = calculate_within_ci(df_mod, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC], DataParser.STIM_DUR_PLND_SEC, DataParser.AMP_DEG)
        df_mod_ci_list.append(df_mod_ci)

    df_ci = pd.concat(df_mod_ci_list)
    #df_ci = calculate_within_ci(all_sub_means_df, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY], DataParser.AMP_DEG)
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).mean(
        numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).sem(
        numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    mean_df[f"{DataParser.AMP_DEG}_CI_SE"] = list(df_ci)
    """
    Calculate rolling average
    """
    for modality in list(subs_dict.keys()):
        for dur in STIMULUS_DURATIONS:
            ind = (mean_df[DataParser.STIM_DUR_PLND_SEC] == dur) & (mean_df[MODALITY] == modality)
            dur_means = mean_df[ind]
            rolling_data = dur_means.rolling(
                int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
            mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    df_cat = mean_df.reset_index(drop=True, inplace=False)
    rolling_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                         avg_col=f"{DataParser.AMP_DEG}Rolling", avg_col_name="Average Saccade Amplitude (Degrees VA)",
                         se_col=f"{DataParser.AMP_DEG}_CI_SE", hue_col=DataParser.STIM_DUR_PLND_SEC,  #se_col=f"{DataParser.AMP_DEG}_SE"
                         hue_col_name="Stimulus Duration", hue_names_map=STIMULUS_DURATION_NAME_MAP,
                         palette=STIMULUS_DURATION_MAP,
                         title_name=f"Average Saccade Amplitude Overall",
                         save_path=save_path, save_name=f"sacc_amp_line_{phase_name}")
    return


def saccade_amp(subs_dict, save_path, phase_name):
    """
    This method calculates the average saccade amplitude PER *** SUBJECT X CATEGORY X DURATION ***

    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    modality_sampling_rate = {ET_param_manager.ECOG: None, ET_param_manager.FMRI: None, ET_param_manager.MEG: None}

    total_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in list(subs_dict.keys()):
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                sub_sampling_rate = sub_data[PARAMS][
                    DataParser.SAMPLING_FREQ]  # subject's sampling rate - for rolling avg
                if modality_sampling_rate[
                    modality] is None:  # THIS ASSUMES WE HAVE THE SAME SAMPLING RATE WITHIN A MODALITY!!!
                    modality_sampling_rate[modality] = sub_sampling_rate
                trial_sacc_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.REAL_SACC] == True,
                                     :]  # only REAL saccades
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL] != -1,
                                     :]  # that are within trials
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL].isin(
                    list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
                trial_sacc_samples = calculate_sample_time_in_trial(relevant_trials, trial_sacc_samples,
                                                                    [DataParser.STIM_DUR_PLND_SEC])

                # calculate for each time point the MEAN SACCADE AMPLITUDE
                trial_means = trial_sacc_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, DataParser.AMP_DEG]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
                trial_means.loc[:, DataParser.STIM_TYPE_COL] = category
                trial_means.loc[:, MODALITY] = modality
                """
                Raw data is noisy - for plotting purposes only, let's smooth the data over a window of ROLLING_SIZE_MS.
                For that, we will use pandas' rolling average method, to create a new "rolling" column containing the
                smoothed data. This method is also called SMA (standard moving average).
                https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

                for dur in STIMULUS_DURATIONS:
                    dur_means = trial_means[trial_means[DataParser.STIM_DUR_PLND_SEC] == dur]
                    rolling_data = dur_means.rolling(int((ROLLING_SIZE_MS/1000)/(1/sub_sampling_rate))).mean(numeric_only=True)
                    dur_means.loc[:, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]
                    only_mod_dfs_list.append(dur_means)
                """
                only_mod_dfs_list.append(trial_means)

            # AVERAGE all subjects in this modality
            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(mod_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(all_dfs_list)
        total_df_list.append(all_mean_df)

    all_sub_means_df = pd.concat(total_df_list)

    """
    Now, all categories at once
    """
    #df_ci = calculate_within_ci(all_sub_means_df, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY], DataParser.AMP_DEG)
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).mean(
        numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).sem(
        numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    #mean_df[f"{DataParser.AMP_DEG}_CI_SE"] = list(df_ci)
    """
    Calculate rolling average
    """
    for modality in list(subs_dict.keys()):
        for dur in STIMULUS_DURATIONS:
            ind = (mean_df[DataParser.STIM_DUR_PLND_SEC] == dur) & (mean_df[MODALITY] == modality)
            dur_means = mean_df[ind]
            rolling_data = dur_means.rolling(
                int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
            mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    df_cat = mean_df.reset_index(drop=True, inplace=False)
    rolling_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                         avg_col=f"{DataParser.AMP_DEG}Rolling", avg_col_name="Average Saccade Amplitude (Degrees VA)",
                         se_col=f"{DataParser.AMP_DEG}_CI_SE", hue_col=DataParser.STIM_DUR_PLND_SEC,  #se_col=f"{DataParser.AMP_DEG}_SE"
                         hue_col_name="Stimulus Duration", hue_names_map=STIMULUS_DURATION_NAME_MAP,
                         palette=STIMULUS_DURATION_MAP,
                         title_name=f"Average Saccade Amplitude Overall",
                         save_path=save_path, save_name=f"sacc_amp_line_{phase_name}")
    """
    DEPRECATED
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY, DataParser.STIM_TYPE_COL]).sem(numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    Calculate rolling average
    for category in STIM_TYPES:
        for modality in modality_list:
            for dur in STIMULUS_DURATIONS:
                ind = (mean_df[DataParser.STIM_DUR_PLND_SEC] == dur) & (mean_df[MODALITY] == modality) & (mean_df[DataParser.STIM_TYPE_COL] == category)
                dur_means = mean_df[ind]
                rolling_data = dur_means.rolling(int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
                mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    # PLOT LINE OF SACCADE AMPLITUDE AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        df_cat = mean_df[mean_df[DataParser.STIM_TYPE_COL] == category].reset_index(drop=True, inplace=False)
        rolling_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                             avg_col=f"{DataParser.AMP_DEG}Rolling", avg_col_name="Average Saccade Amplitude (Degrees VA)",
                             se_col=f"{DataParser.AMP_DEG}_SE", hue_col=DataParser.STIM_DUR_PLND_SEC,
                             hue_col_name="Stimulus Duration", hue_names_map=STIMULUS_DURATION_NAME_MAP,
                             palette=STIMULUS_DURATION_MAP,
                             title_name=f"Average Saccade Amplitude in **{category}** Trials",
                             save_path=save_path, save_name=f"sacc_amp_line_{category}")

    Now, per LAB 
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY, DataParser.STIM_TYPE_COL, LAB]).mean(numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY, DataParser.STIM_TYPE_COL, LAB]).sem(numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]

    Calculate rolling average

    labs = list(mean_df[LAB].unique())
    for lab in labs:
        for category in STIM_TYPES:
            for dur in STIMULUS_DURATIONS:
                ind = (mean_df[DataParser.STIM_DUR_PLND_SEC] == dur) & (mean_df[LAB] == lab) & (mean_df[DataParser.STIM_TYPE_COL] == category)
                dur_means = mean_df[ind]
                rolling_data = dur_means.rolling(int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
                mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    # PLOT LINE OF SACCADE AMPLITUDE AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        for modality in modality_list:
            df_cat = mean_df[(mean_df[DataParser.STIM_TYPE_COL] == category) & (mean_df[MODALITY] == modality)].reset_index(drop=True, inplace=False)
            rolling_plot_per_lab(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                                 avg_col=f"{DataParser.AMP_DEG}Rolling", avg_col_name="Average Saccade Amplitude (Degrees VA)",
                                 se_col=f"{DataParser.AMP_DEG}_SE", hue_col=DataParser.STIM_DUR_PLND_SEC,
                                 hue_col_name="Stimulus Duration", hue_names_map=STIMULUS_DURATION_NAME_MAP,
                                 palette=STIMULUS_DURATION_MAP,
                                 title_name=f"Average Saccade Amplitude in **{category}** Trials",
                                 save_path=save_path, save_name=f"sacc_amp_line_{category}_{modality}")

    """
    return


def saccade_amp_relevance(subs_dict, save_path):
    """
    This method calculates the average saccade amplitude PER *** SUBJECT X CATEGORY X DURATION ***

    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    modality_list = [ET_param_manager.ECOG, ET_param_manager.FMRI, ET_param_manager.MEG]
    modality_sampling_rate = {ET_param_manager.ECOG: None, ET_param_manager.FMRI: None, ET_param_manager.MEG: None}

    total_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in modality_list:
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                sub_sampling_rate = sub_data[PARAMS][
                    DataParser.SAMPLING_FREQ]  # subject's sampling rate - for rolling avg
                if modality_sampling_rate[modality] is None:  # THIS ASSUMES WE HAVE THE SAME SAMPLING RATE WITHIN A MODALITY!!!
                    modality_sampling_rate[modality] = sub_sampling_rate
                trial_sacc_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.REAL_SACC] == True,
                                     :]  # only REAL saccades
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL] != -1,
                                     :]  # that are within trials
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                trial_sacc_samples = trial_sacc_samples.loc[trial_sacc_samples[DataParser.TRIAL].isin(
                    list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
                trial_sacc_samples = calculate_sample_time_in_trial(relevant_trials, trial_sacc_samples,
                                                                    [DataParser.IS_TASK_RELEVANT_COL])

                # calculate for each time point the MEAN SACCADE AMPLITUDE
                trial_means = trial_sacc_samples.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL]).mean(numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, DataParser.AMP_DEG]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
                trial_means.loc[:, DataParser.STIM_TYPE_COL] = category
                trial_means.loc[:, MODALITY] = modality
                """
                Raw data is noisy - for plotting purposes only, let's smooth the data over a window of ROLLING_SIZE_MS.
                For that, we will use pandas' rolling average method, to create a new "rolling" column containing the
                smoothed data. This method is also called SMA (standard moving average).
                https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

                for dur in STIMULUS_DURATIONS:
                    dur_means = trial_means[trial_means[DataParser.STIM_DUR_PLND_SEC] == dur]
                    rolling_data = dur_means.rolling(int((ROLLING_SIZE_MS/1000)/(1/sub_sampling_rate))).mean(numeric_only=True)
                    dur_means.loc[:, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]
                    only_mod_dfs_list.append(dur_means)
                """
                only_mod_dfs_list.append(trial_means)

            # AVERAGE all subjects in this modality
            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(mod_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(all_dfs_list)
        total_df_list.append(all_mean_df)

    all_sub_means_df = pd.concat(total_df_list)

    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.DataParser.IS_TASK_RELEVANT_COL, MODALITY, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY, DataParser.STIM_TYPE_COL]).sem( numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    """
    Calculate rolling average
    """
    for category in STIM_TYPES:
        for modality in modality_list:
            for relevance in [True, False]:
                ind = (mean_df[DataParser.IS_TASK_RELEVANT_COL] == relevance) & (mean_df[MODALITY] == modality) & (
                            mean_df[DataParser.STIM_TYPE_COL] == category)
                dur_means = mean_df[ind]
                rolling_data = dur_means.rolling(
                    int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
                mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    # PLOT LINE OF SACCADE AMPLITUDE AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        df_cat = mean_df[mean_df[DataParser.STIM_TYPE_COL] == category].reset_index(drop=True, inplace=False)
        rolling_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                             avg_col=f"{DataParser.AMP_DEG}Rolling",
                             avg_col_name="Average Saccade Amplitude (Degrees VA)",
                             se_col=f"{DataParser.AMP_DEG}_SE", hue_col=DataParser.IS_TASK_RELEVANT_COL,
                             hue_col_name="Stimulus Relevance", hue_names_map=STIMULUS_RELEVANCE_NAME_MAP,
                             palette=STIMULUS_RELEVANCE_MAP,
                             title_name=f"Average Saccade Amplitude in **{category}** Trials",
                             save_path=save_path, save_name=f"sacc_amp_line_{category}_tr")

    """
    Now, per LAB 
    """
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY, DataParser.STIM_TYPE_COL, LAB]).mean(numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY, DataParser.STIM_TYPE_COL, LAB]).sem(numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    """
    Calculate rolling average
    """
    labs = list(mean_df[LAB].unique())
    for lab in labs:
        for category in STIM_TYPES:
            for relevance in [True, False]:
                ind = (mean_df[DataParser.IS_TASK_RELEVANT_COL] == relevance) & (mean_df[LAB] == lab) & (
                            mean_df[DataParser.STIM_TYPE_COL] == category)
                dur_means = mean_df[ind]
                rolling_data = dur_means.rolling(
                    int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
                mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    # PLOT LINE OF SACCADE AMPLITUDE AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        for modality in modality_list:
            df_cat = mean_df[(mean_df[DataParser.STIM_TYPE_COL] == category) & (mean_df[MODALITY] == modality)].reset_index(drop=True, inplace=False)
            if not df_cat.empty:
                rolling_plot_per_lab(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                                     avg_col=f"{DataParser.AMP_DEG}Rolling",
                                     avg_col_name="Average Saccade Amplitude (Degrees VA)",
                                     se_col=f"{DataParser.AMP_DEG}_SE", hue_col=DataParser.IS_TASK_RELEVANT_COL,
                                     hue_col_name="Stimulus Relevance", hue_names_map=STIMULUS_RELEVANCE_NAME_MAP,
                                     palette=STIMULUS_RELEVANCE_MAP,
                                     title_name=f"Average Saccade Amplitude in **{category}** Trials",
                                     save_path=save_path, save_name=f"sacc_amp_line_{category}_{modality}_tr")

    """
    Now, all caterogies at once
    """
    mean_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY]).mean(
        numeric_only=True).reset_index()
    se_df = all_sub_means_df.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY]).sem(
        numeric_only=True).reset_index()
    mean_df.loc[:, f"{DataParser.AMP_DEG}_SE"] = se_df.loc[:, DataParser.AMP_DEG]
    """
    Calculate rolling average
    """
    for modality in modality_list:
        for relevance in [True, False]:
            ind = (mean_df[DataParser.IS_TASK_RELEVANT_COL] == relevance) & (mean_df[MODALITY] == modality)
            dur_means = mean_df[ind]
            rolling_data = dur_means.rolling(
                int((ROLLING_SIZE_MS / 1000) / (1 / modality_sampling_rate[modality]))).mean(numeric_only=True)
            mean_df.loc[ind, f"{DataParser.AMP_DEG}Rolling"] = rolling_data[DataParser.AMP_DEG]

    df_cat = mean_df.reset_index(drop=True, inplace=False)
    rolling_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                         avg_col=f"{DataParser.AMP_DEG}Rolling", avg_col_name="Average Saccade Amplitude (Degrees VA)",
                         se_col=f"{DataParser.AMP_DEG}_SE", hue_col=DataParser.IS_TASK_RELEVANT_COL,
                         hue_col_name="Stimulus Relevance", hue_names_map=STIMULUS_RELEVANCE_NAME_MAP,
                         palette=STIMULUS_RELEVANCE_MAP,
                         title_name=f"Average Saccade Amplitude Overall",
                         save_path=save_path, save_name=f"sacc_amp_line_tr")
    return


def saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, category, additional_col_name, additional_col_value,
                      modality):
    total_sub_len = trial_saccs_dur.shape[0]
    bin_dfs = pd.DataFrame()  # this contains the amt of saccades in this bin per bin
    bin_dfs[RANGE_START_RAD] = [deg * (math.pi / 180) for deg in range(-180, 180, bin_size)]
    bin_dfs["range_end_rads"] = [(deg + bin_size) * (math.pi / 180) for deg in range(-180, 180, bin_size)]
    bin_count_list = []
    # iterate over degree BINS (notice the step in the for) and find all saccades which are within
    # that range
    for deg in range(-180, 180, bin_size):
        range_start_rads = deg * (math.pi / 180)
        range_end_rads = (deg + bin_size) * (math.pi / 180)
        relevant = trial_saccs_dur.loc[(range_start_rads <= trial_saccs_dur[DataParser.SACC_DIRECTION_RAD]) &
                                       ((range_end_rads > trial_saccs_dur[DataParser.SACC_DIRECTION_RAD])), :]
        # add the number of saccades falling within that bin divided by the total number of saccades in that (sub)condition
        if total_sub_len == 0:
            bin_count_list.append(0)
        else:
            bin_count_list.append(relevant.shape[0])  # proportion
    bin_dfs["bin_count"] = bin_count_list
    if total_sub_len != 0:
        bin_dfs[BIN_PROPORTION] = bin_dfs["bin_count"] / total_sub_len
    else:
        bin_dfs[BIN_PROPORTION] = 0
    bin_dfs[SUBJECT] = sub_data[PARAMS]['SubjectName']
    bin_dfs[MODALITY] = modality
    bin_dfs[LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
    bin_dfs[additional_col_name] = additional_col_value
    bin_dfs[DataParser.STIM_TYPE_COL] = category
    return bin_dfs


def saccade_ciruclar_plot(subs_dict, save_path, phase_name):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    everything_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                all_stim_dur_dfs = []
                for stim_dur in STIMULUS_DURATIONS:
                    relevant_trials_dur = relevant_trials.loc[relevant_trials[DataParser.STIM_DUR_PLND_SEC] == stim_dur,
                                          :]
                    trial_saccs_dur = trial_saccs.loc[trial_saccs[DataParser.TRIAL].isin(
                        list(relevant_trials_dur[DataParser.TRIAL_NUMBER])), :]
                    all_stim_dur_dfs.append(saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, category,
                                                              DataParser.STIM_DUR_PLND_SEC, stim_dur, modality))
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        everything_df_list.append(all_bins_df)

    all_bins_df = pd.concat(everything_df_list)
    all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_all_{phase_name}.csv"), index=False)

    mean_df = all_bins_df.groupby(
        [RANGE_START_RAD, DataParser.STIM_DUR_PLND_SEC, DataParser.STIM_TYPE_COL, MODALITY]).mean(
        numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        df_cat = mean_df[mean_df[DataParser.STIM_TYPE_COL] == category]
        polar_plot_per_mod(data=df_cat, title=f"Saccadde Direction Density in **{category}** Trials",
                           hue_col=DataParser.STIM_DUR_PLND_SEC,
                           hue_col_name="Stimulus Durations",
                           hue_names_map=STIMULUS_DURATION_NAME_MAP, hue_color_map=STIMULUS_DURATION_MAP,
                           save_path=save_path, save_name=f"sacc_dir_polar_{category}_{phase_name}")

    # all
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.STIM_DUR_PLND_SEC, MODALITY]).mean(
        numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    polar_plot_per_mod(data=mean_df, title=f"Saccadde Direction Density in ALL Trials",
                       hue_col=DataParser.STIM_DUR_PLND_SEC,
                       hue_col_name="Stimulus Durations",
                       hue_names_map=STIMULUS_DURATION_NAME_MAP, hue_color_map=STIMULUS_DURATION_MAP,
                       save_path=save_path, save_name=f"sacc_dir_polar_{phase_name}")
    return


def saccade_ciruclar_relevance_plot(subs_dict, save_path):
    """
    This plots a circular (polar) plot of the average saccade direction.
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    We take all of the saccades of a subject that are in a trial of a certain category, then bin them, and calculate
    MEAN between BINS
    """
    BINS = 30
    bin_size = int(
        2 * 180 / BINS)  # MUST be an integer; if you want to change the number of bins note this number as well

    everything_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                trial_saccs = sub_data[ET_DATA_DICT][DataParser.DF_SACC]
                trial_saccs = trial_saccs.loc[trial_saccs[DataParser.HERSHMAN_PAD] == False, :]  # only REAL saccades
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                all_stim_dur_dfs = []
                for relevance in [True, False]:
                    relevant_trials_relevance = relevant_trials.loc[
                                                relevant_trials[DataParser.IS_TASK_RELEVANT_COL] == relevance, :]
                    trial_saccs_dur = trial_saccs.loc[trial_saccs[DataParser.TRIAL].isin(
                        list(relevant_trials_relevance[DataParser.TRIAL_NUMBER])), :]
                    all_stim_dur_dfs.append(saccade_calc_bins(trial_saccs_dur, bin_size, sub_data, category,
                                                              DataParser.IS_TASK_RELEVANT_COL, relevance, modality))
                only_mod_dfs_list.append(pd.concat(all_stim_dur_dfs))

            # Calculated everything for all subs in modality (for the category), now calculate the MEAN
            all_mod_bins_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(all_mod_bins_df)
        # Now, we do the mean for all mods combined
        all_bins_df = pd.concat(all_dfs_list)
        everything_df_list.append(all_bins_df)

    all_bins_df = pd.concat(everything_df_list)
    all_bins_df.to_csv(os.path.join(save_path, f"sacc_dir_polar_relevance_all.csv"), index=False)

    mean_df = all_bins_df.groupby(
        [RANGE_START_RAD, DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL, MODALITY]).mean(
        numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    for category in STIM_TYPES:
        df_cat = mean_df[mean_df[DataParser.STIM_TYPE_COL] == category]
        polar_plot_per_mod(data=df_cat, title=f"Saccadde Direction Density in **{category}** Trials",
                           hue_col=DataParser.IS_TASK_RELEVANT_COL,
                           hue_col_name="Task Relevance",
                           hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, hue_color_map=STIMULUS_RELEVANCE_MAP,
                           save_path=save_path, save_name=f"sacc_dir_polar_{category}_tr")

    # all
    mean_df = all_bins_df.groupby([RANGE_START_RAD, DataParser.IS_TASK_RELEVANT_COL, MODALITY]).mean(
        numeric_only=True).reset_index()
    # PLOT DIRECTION OF SACCADE DISTRIBUTION AVERAGED *** WITHIN A STIMULUS CATEGORY ***
    polar_plot_per_mod(data=mean_df, title=f"Saccadde Direction Density in ALL Trials",
                       hue_col=DataParser.IS_TASK_RELEVANT_COL,
                       hue_col_name="Task Relevance",
                       hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, hue_color_map=STIMULUS_RELEVANCE_MAP,
                       save_path=save_path, save_name=f"sacc_dir_polar_tr")
    return


def sacc_num_plot(subs_dict, save_path, phase_name):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    all_dfs_list = []
    for modality in subs_dict.keys():
        print(modality)
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            print(sub_data_path)
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_samples[f'{DataParser.REAL_SACC}'].fillna(0,
                                                            inplace=True)  # Replace non-saccs with 0, so the mean will actually be PROPORTION
            trials = sub_data[TRIAL_INFO]
            trial_samples = calculate_sample_time_in_trial(trials, trial_samples, [DataParser.STIM_DUR_PLND_SEC])

            trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                            :]  # ones that are within a TRIAL epoch (not between trials)
            # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCS
            trial_means = trial_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                numeric_only=True).reset_index()
            trial_means = trial_means.loc[:, [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, f"{DataParser.REAL_SACC}"]]
            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            only_mod_dfs_list.append(trial_means)
        # summarize this modality - average across all subjects' averages (and calculate std)
        mod_mean_df = pd.concat(only_mod_dfs_list)
        all_dfs_list.append(mod_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(all_dfs_list)
    all_mean_df = all_mean_df[all_mean_df[TIME_IN_EPOCH] <= 2000]

    # as we have samples where only one subject had a sacc, this skewes the plot. Therefore, create a version w/o it:
    count_samps = all_mean_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    all_mean_df = all_mean_df.loc[~ ((all_mean_df.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                                     (all_mean_df.plndStimulusDur.isin(count_samps["plndStimulusDur"])) &
                                     (all_mean_df["mod"].isin(count_samps["mod"])))]

    Y_DICT = {ET_param_manager.ECOG: (0.0, 0.2), ET_param_manager.FMRI: (0.0, 0.04), ET_param_manager.MEG: (0.0, 0.04)}
    Y_TICKS = {ET_param_manager.ECOG: [0.0, 0.05, 0.10, 0.15, 0.20], ET_param_manager.FMRI: [0.0, 0.01, 0.02, 0.03, 0.04], ET_param_manager.MEG: [0.0, 0.01, 0.02, 0.03, 0.04]}

    line_plot_per_mod(data=all_mean_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.REAL_SACC, y_col_name="Number of Saccades",
                      hue_col=DataParser.STIM_DUR_PLND_SEC, hue_col_name="Stimulus Duration",
                      title_name="Average Number of Saccades in Trial", global_y_max=False,
                      save_path=save_path, save_name=f"sacc_num_line_{phase_name}",
                      hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP,
                      black_vertical_x=[0], gray_vertical_x=[500, 1000, 1500], y_lim_dict=Y_DICT, y_ticks=Y_TICKS)

    return


def blink_plot(subs_dict, save_path, phase_name):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    all_dfs_list = []
    for modality in subs_dict.keys():
        print(modality)
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            print(sub_data_path)
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            analyzed_eye = sub_data[PARAMS][DataParser.EYE]
            trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                         inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
            trials = sub_data[TRIAL_INFO]
            trial_samples = calculate_sample_time_in_trial(trials, trial_samples, [DataParser.STIM_DUR_PLND_SEC])

            trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                            :]  # ones that are within a TRIAL epoch (not between trials)
            # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
            trial_means = trial_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                numeric_only=True).reset_index()
            trial_means = trial_means.loc[:,
                          [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"}, inplace=True)
            only_mod_dfs_list.append(trial_means)
        # summarize this modality - average across all subjects' averages (and calculate std)
        mod_mean_df = pd.concat(only_mod_dfs_list)
        all_dfs_list.append(mod_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(all_dfs_list)
    all_mean_df = all_mean_df[all_mean_df[TIME_IN_EPOCH] <= 2000]

    """ DOING THIS AT THIS STAGE WILL ALSO PLOT SAMPLES WHERE ONLY 1 SUBJECT HAD BLINKED - THAT SKEWES EVERYTHING, SO LET'S REMOVE IT PRIOR TO PLOTTING
        line_plot_per_mod(data=all_mean_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Number of Blinks",
                          hue_col=DataParser.STIM_DUR_PLND_SEC, hue_col_name="Stimulus Duration",
                          title_name="Average Number of Blinks in Trial",
                          save_path=save_path, save_name="blink_num_line",
                          hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP)
    """

    # as we have samples where only one subject had a blink, this skewes the plot. Therefore, create a version w/o it:
    count_samps = all_mean_df.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, MODALITY]).count().reset_index()
    count_samps = count_samps[count_samps["sub"] == 1]
    all_mean_df = all_mean_df.loc[~ ((all_mean_df.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                                     (all_mean_df.plndStimulusDur.isin(count_samps["plndStimulusDur"])) &
                                     (all_mean_df["mod"].isin(count_samps["mod"])))]

    Y_DICT = {ET_param_manager.ECOG: (0.0, 0.5), ET_param_manager.FMRI: (0.0, 0.5), ET_param_manager.MEG: (0.0, 0.5)}
    Y_TICKS = {ET_param_manager.ECOG: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ET_param_manager.FMRI: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], ET_param_manager.MEG: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    line_plot_per_mod(data=all_mean_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                      y_col=DataParser.HERSHMAN, y_col_name="Number of Blinks", global_y_max=False,
                      hue_col=DataParser.STIM_DUR_PLND_SEC, hue_col_name="Stimulus Duration",
                      title_name="Average Number of Blinks in Trial",
                      save_path=save_path, save_name=f"blink_num_line_{phase_name}",
                      hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP,
                      black_vertical_x=[0], gray_vertical_x=[500, 1000, 1500], y_lim_dict=Y_DICT, y_ticks=Y_TICKS)

    return


def sacc_num_plot_trial_relevent(subs_dict, save_path):
    everything_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                print(sub_data_path)
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_samples[f'{DataParser.REAL_SACC}'].fillna(0,
                                                                inplace=True)  # Replace non saccades with 0, so the mean will actually be PROPORTION
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                trial_samples = trial_samples.loc[
                                trial_samples[DataParser.TRIAL].isin(list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
                trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples,
                                                               [DataParser.IS_TASK_RELEVANT_COL])

                trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                                :]  # ones that are within a TRIAL epoch (not between trials)
                # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT SACCS
                trial_means = trial_samples.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:,
                              [TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, f"{DataParser.REAL_SACC}"]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
                trial_means.loc[:, DataParser.STIM_TYPE_COL] = category
                trial_means.loc[:, MODALITY] = modality
                only_mod_dfs_list.append(trial_means)
            # summarize this modality - average across all subjects' averages (and calculate std)
            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(mod_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(all_dfs_list)
        everything_df_list.append(all_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(everything_df_list)
    # as we have samples where only one subject had a saccade, this skewes the plot. Therefore, create a version w/o it:
    for category in STIM_TYPES:
        df_cat = all_mean_df[all_mean_df[DataParser.STIM_TYPE_COL] == category]

        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.isTaskRelevant.isin(count_samps[DataParser.IS_TASK_RELEVANT_COL])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.REAL_SACC, y_col_name="Number of Saccades", global_y_max=False,
                          hue_col=DataParser.IS_TASK_RELEVANT_COL, hue_col_name="Stimulus Relevance",
                          title_name="Average Number of Saccades in Trial",
                          save_path=save_path, save_name=f"sacc_num_line_tr_{category}",
                          hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP,
                          black_vertical_x=[0], gray_vertical_x=[500, 1000, 1500])
    return


def blink_plot_trial_relevent(subs_dict, save_path):
    """
    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    """
    everything_df_list = []
    for category in STIM_TYPES:
        all_dfs_list = []
        for modality in subs_dict.keys():
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                print(sub_data_path)
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()
                analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_samples[f'{analyzed_eye}{DataParser.HERSHMAN}'].fillna(0,
                                                                             inplace=True)  # Replace non blinks with 0, so the mean will actually be PROPORTION
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.STIM_TYPE_COL] == category, :]
                trial_samples = trial_samples.loc[
                                trial_samples[DataParser.TRIAL].isin(list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
                trial_samples = calculate_sample_time_in_trial(relevant_trials, trial_samples,
                                                               [DataParser.IS_TASK_RELEVANT_COL])

                trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1,
                                :]  # ones that are within a TRIAL epoch (not between trials)
                # collapse across stimulus duration groups, at each timepoint (from epoch start to end) and COUNT BLINKS
                trial_means = trial_samples.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:,
                              [TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, f"{analyzed_eye}{DataParser.HERSHMAN}"]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
                trial_means.loc[:, DataParser.STIM_TYPE_COL] = category
                trial_means.loc[:, MODALITY] = modality
                trial_means.rename(columns={f"{analyzed_eye}{DataParser.HERSHMAN}": f"{DataParser.HERSHMAN}"},
                                   inplace=True)
                only_mod_dfs_list.append(trial_means)
            # summarize this modality - average across all subjects' averages (and calculate std)
            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_dfs_list.append(mod_mean_df)

        # Now, we do the all subs plot
        all_mean_df = pd.concat(all_dfs_list)
        everything_df_list.append(all_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(everything_df_list)
    # as we have samples where only one subject had a blink, this skewes the plot. Therefore, create a version w/o it:
    for category in STIM_TYPES:
        df_cat = all_mean_df[all_mean_df[DataParser.STIM_TYPE_COL] == category]

        count_samps = df_cat.groupby([TIME_IN_EPOCH, DataParser.IS_TASK_RELEVANT_COL, MODALITY]).count().reset_index()
        count_samps = count_samps[count_samps["sub"] == 1]
        df_cat = df_cat.loc[~ ((df_cat.timeInEpoch.isin(count_samps["timeInEpoch"])) &
                               (df_cat.isTaskRelevant.isin(count_samps[DataParser.IS_TASK_RELEVANT_COL])) &
                               (df_cat["mod"].isin(count_samps["mod"])))]
        line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=DataParser.HERSHMAN, y_col_name="Number of Blinks",
                          hue_col=DataParser.IS_TASK_RELEVANT_COL, hue_col_name="Stimulus Relevance",
                          title_name="Average Number of Blinks in Trial", global_y_max=False,
                          save_path=save_path, save_name=f"blink_num_line_tr_{category}",
                          hue_names_map=STIMULUS_RELEVANCE_NAME_MAP, palette=STIMULUS_RELEVANCE_MAP)

    return


def add_pupil_to_lmm(sub_data, pupil_sub_means):
    eye = sub_data[PARAMS]["Eye"]
    samps = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
    trial_info = sub_data[TRIAL_INFO]
    # FIXATION ANALYSIS PREPARATION
    samps = samps.loc[samps[f"{eye}{DataParser.HERSHMAN_PAD}"].isna(), :]  # only REAL PUPIL SIZE
    samps[f"{DataParser.REAL_PUPIL}Norm"] = samps[DataParser.REAL_PUPIL] / pupil_sub_means[
        sub_data[PARAMS]['SubjectName']]
    for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
        # TODO: determine if the mean should be weighted or not (within trial+window). The Current implementation IS WEIGHTED
        mean_pupil = samps.groupby([window]).mean(numeric_only=True).reset_index()
        for index, trial in mean_pupil.iterrows():
            if trial[window] != -1:  # if -1, this means it is not part of the current window
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}{DataParser.REAL_PUPIL}Norm"] = trial[f"{DataParser.REAL_PUPIL}Norm"]

        # TODO: determine if the mean should be weighted or not (within trial+window). The Current implementation IS WEIGHTED
        mean_pupil = samps.groupby([window]).median(numeric_only=True).reset_index()
        for index, trial in mean_pupil.iterrows():
            if trial[window] != -1:  # if -1, this means it is not part of the current window
                trial_info.loc[trial_info["trialNumber"] == trial[window], f"{window}{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm"] = trial[f"{DataParser.REAL_PUPIL}Norm"]

    sub_data[TRIAL_INFO] = trial_info
    return sub_data


def lmm_data(subs_dict, pupil_sub_means, save_path, phase_name):
    """
    Concatenate all subjects' trial data, to later perform LMM analysis (in R)
    :param subs_dict:
    :param save_path:
    :return:
    """
    all_subs_list = []
    trial_list = list()
    for modality in subs_dict.keys():  # for each modality
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            if sub_data[PARAMS][ET_param_manager.SUBJECT_LAB] != "SF" and sub_data[PARAMS]['SubjectName'] in pupil_sub_means:
                sub_data = add_pupil_to_lmm(sub_data, pupil_sub_means)

            """
            This is for reporting overall stats
            """
            trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]
            try:
                trial_samples[f"{sub_data[PARAMS][DataParser.EYE]}{DataParser.HERSHMAN}"].fillna(0, inplace=True)
            except Exception as e:
                print(f"Subject {sub_data_path} is BAD!")
                raise e
            sub_trials = trial_samples.groupby(["Trial"]).mean().reset_index()
            sub_trials.loc[:, "SubjectName"] = sub_data[PARAMS]['SubjectName']
            cols = list(sub_trials.columns)
            eye_relevant_cols = [x for x in cols if x.startswith(sub_data[PARAMS][DataParser.EYE])]
            eye_relevant_no_prefix = [x[1:] for x in cols if x.startswith(sub_data[PARAMS][DataParser.EYE])]
            sub_trials[eye_relevant_no_prefix] = sub_trials[eye_relevant_cols]
            sub_trials.drop(columns=eye_relevant_cols, inplace=True)
            sub_trials = sub_trials.groupby(["SubjectName"]).mean().reset_index()
            trial_list.append(sub_trials)
            all_subs_list.append(sub_data[TRIAL_INFO])

    all_subs_df = pd.concat(all_subs_list)
    trial_df = pd.concat(trial_list)
    trial_df.to_csv(os.path.join(save_path, f"data_by_trial_{phase_name}.csv"), index=False)

    cols = list(all_subs_df.columns)
    time_unrelated = [col for col in cols if
                      (DataParser.FIRST_WINDOW not in col) and (DataParser.SECOND_WINDOW not in col) and (
                              DataParser.THIRD_WINDOW not in col)]
    df_list = []
    for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
        time_df = pd.DataFrame()
        time_relevant_cols = [x for x in cols if window in x]
        time_relevant_cols_no_prefix = [x.replace(window, "") for x in cols if window in x]
        time_df[time_unrelated] = all_subs_df[time_unrelated]
        time_df[time_relevant_cols_no_prefix] = all_subs_df[time_relevant_cols]
        time_df["timeWindow"] = window
        df_list.append(time_df)
    total_df = pd.concat(df_list)

    # last-minute accuracy request
    accuracy_df = total_df.dropna(subset=["CenterDistDegs_BLcorrected"], inplace=False)
    accuracy_df["isFix<2"] = accuracy_df["CenterDistDegs_BLcorrected"].apply(lambda x: 0 if x >= 2 else 1)
    accuracy_fix = accuracy_df.groupby("subCode").mean().reset_index()
    accuracy_fix.to_csv(os.path.join(save_path, f"lmm_df_{phase_name}_accuracy.csv"), index=False)
    total_df.to_csv(os.path.join(save_path, f"lmm_df_{phase_name}.csv"), index=False)
    return


def pupil_plot_no_tr(subs_dict, save_path, phase_name, do_each_category=False, group_with=DataParser.STIM_DUR_PLND_SEC):
    """
    This function prepares the data for plotting the average pupil size at each timepoint within a trial EPOCH,
    such that the X axis will depict time (relative to stimulus onset = 0), and the Y axis will depict SIZE.
    Notably, the pupil sizes are in ARBITRARY EYELINK UNITS.

    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    :param subs_dict:
    :param save_path:
    :return:
    """
    if group_with == DataParser.STIM_DUR_PLND_SEC:
        hue_col = group_with
        hue_col_name = "Stimulus Duration"
        save_name = f"pupil_line_{phase_name}"
        hue_names_map = STIMULUS_DURATION_NAME_MAP
        palette = STIMULUS_DURATION_MAP

    else:
        hue_col = group_with
        hue_col_name = "Stimulus Relevance"
        save_name = f"pupil_line_tr_{phase_name}"
        hue_names_map = STIMULUS_RELEVANCE_NAME_MAP
        palette = STIMULUS_RELEVANCE_MAP

    sub_means_dict = {}
    total_df_list = []
    for modality in subs_dict.keys():  # for each modality
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()

            analyzed_eye = sub_data[PARAMS][DataParser.EYE]
            trial_samps = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_samps = trial_samps.loc[trial_samps[f"{analyzed_eye}{DataParser.HERSHMAN_PAD}"].isna(), :]  # only REAL PUPIL SIZE
            trial_samps = trial_samps.loc[trial_samps[DataParser.TRIAL] != -1, :]  # that are within trials
            if sub_data[PARAMS]['SubjectName'] not in sub_means_dict:
                sub_mean = trial_samps[DataParser.REAL_PUPIL].mean()
                sub_means_dict[sub_data[PARAMS]['SubjectName']] = sub_mean
            else:
                sub_mean = sub_means_dict[sub_data[PARAMS]['SubjectName']]
            relevant_trials = sub_data[TRIAL_INFO]
            trial_samps = trial_samps.loc[trial_samps[DataParser.TRIAL].isin(list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
            # SD156: trial_samps was empty as all it had in this condition was BLINKS (no pupil size). garbage in -> garbage out
            if trial_samps.empty:
                print(f"Subject {sub_data_path} has no samples remaining with task relevance")
                continue
            else:
                """
                *** PUPIL SIZE NORMALIZATION ***
                In the process of normalizing pupil size, we normalize the size PER SUBJECT, across all trials
                of that subjects (i.e., across all EPOCHs). We do that by calculating the AVERAGE of all pupil 
                size samples that are within an epoch. Then, each sample is DIVIDED (*not* substracted!) by the 
                subject's own mean. 
                """
                trial_samps[f"{DataParser.REAL_PUPIL}Norm"] = trial_samps[f"{DataParser.REAL_PUPIL}"] / sub_mean
                trial_samps = calculate_sample_time_in_trial(relevant_trials, trial_samps, [group_with, DataParser.STIM_TYPE_COL])

            # calculate for each time point the MEAN PUPIL SIZE
            if not do_each_category:
                trial_means = trial_samps.groupby([TIME_IN_EPOCH, group_with]).mean(numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, group_with, f"{DataParser.REAL_PUPIL}Norm"]]
            else:
                trial_means = trial_samps.groupby([TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
                trial_means = trial_means.loc[:, [TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL, f"{DataParser.REAL_PUPIL}Norm"]]
            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            only_mod_dfs_list.append(trial_means)

        mod_mean_df = pd.concat(only_mod_dfs_list)
        total_df_list.append(mod_mean_df)

    # AVERAGE all subjects in this modality
    all_sub_means_df = pd.concat(total_df_list)
    all_sub_means_df = all_sub_means_df[all_sub_means_df[TIME_IN_EPOCH] <= 2000]

    """
    For each modality, plot a lineplot where X = time, Y = average pupil size for a duration category (short/medium/long)
    across subjects (i.e., one line = one average across subjects, not weighted)
    """
    Y_DICT = {ET_param_manager.ECOG: (0.9, 1.05), ET_param_manager.FMRI: (0.9, 1.05), ET_param_manager.MEG: (0.9, 1.05)}
    Y_TICKS = {ET_param_manager.ECOG: [0.9, 0.95, 1.0, 1.05], ET_param_manager.FMRI: [0.9, 0.95, 1.0, 1.05], ET_param_manager.MEG: [0.9, 0.95, 1.0, 1.05]}

    if not do_each_category:
        line_plot_per_mod(data=all_sub_means_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_PUPIL}Norm", y_col_name="Average Pupil Size (Normalized)",
                          hue_col=DataParser.STIM_DUR_PLND_SEC, hue_col_name="Stimulus Duration",
                          title_name="Average Pupil Size in Trial", global_y_max=False,
                          save_path=save_path, save_name=save_name,
                          hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP,
                          black_vertical_x=[0], gray_vertical_x=[500, 1000, 1500], y_lim_dict=Y_DICT, y_ticks=Y_TICKS)
    else:
        for category in STIM_TYPES:
            df_cat = all_sub_means_df[all_sub_means_df[DataParser.STIM_TYPE_COL] == category]
            line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                              y_col=f"{DataParser.REAL_PUPIL}Norm", y_col_name="Average Pupil Size (Normalized)",
                              hue_col=hue_col, hue_col_name=hue_col_name, global_y_max=False,
                              title_name="Average Pupil Size in Trial",
                              save_path=save_path, save_name=f"{save_name}_{category}",
                              hue_names_map=hue_names_map, palette=palette,
                              black_vertical_x=[0], gray_vertical_x=[500, 1000, 1500], y_lim_dict=Y_DICT, y_ticks=Y_TICKS)

    return sub_means_dict



def pupil_plot(subs_dict, save_path, phase_name, do_each_category=False, group_with=DataParser.STIM_DUR_PLND_SEC):
    """
    This function prepares the data for plotting the average pupil size at each timepoint within a trial EPOCH,
    such that the X axis will depict time (relative to stimulus onset = 0), and the Y axis will depict SIZE.
    Notably, the pupil sizes are in ARBITRARY EYELINK UNITS.

    NOTE - this is NOT a supersubject method. We calculate a mean PER SUBJECT, and then calculate the mean and the std
    of the MEANS
    :param subs_dict:
    :param save_path:
    :return:
    """
    if group_with == DataParser.STIM_DUR_PLND_SEC:
        hue_col = group_with
        hue_col_name = "Stimulus Duration"
        save_name = f"pupil_line_{phase_name}"
        hue_names_map = STIMULUS_DURATION_NAME_MAP
        palette = STIMULUS_DURATION_MAP

    else:
        hue_col = group_with
        hue_col_name = "Stimulus Relevance"
        save_name = f"pupil_line_tr_{phase_name}"
        hue_names_map = STIMULUS_RELEVANCE_NAME_MAP
        palette = STIMULUS_RELEVANCE_MAP

    sub_means_dict = {}
    total_df_list = []
    for is_tr in [True, False]:  # for each task relevance type
        all_tr_list = []
        for modality in subs_dict.keys():  # for each modality
            only_mod_dfs_list = []
            for sub_data_path in subs_dict[modality]:
                fl = open(sub_data_path, 'rb')
                sub_data = pickle.load(fl)
                fl.close()

                analyzed_eye = sub_data[PARAMS][DataParser.EYE]
                trial_samps = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
                trial_samps = trial_samps.loc[trial_samps[f"{analyzed_eye}{DataParser.HERSHMAN_PAD}"].isna(), :]  # only REAL PUPIL SIZE
                trial_samps = trial_samps.loc[trial_samps[DataParser.TRIAL] != -1, :]  # that are within trials
                if sub_data[PARAMS]['SubjectName'] not in sub_means_dict:
                    sub_mean = trial_samps[DataParser.REAL_PUPIL].mean()
                    sub_means_dict[sub_data[PARAMS]['SubjectName']] = sub_mean
                else:
                    sub_mean = sub_means_dict[sub_data[PARAMS]['SubjectName']]
                relevant_trials = sub_data[TRIAL_INFO]
                relevant_trials = relevant_trials.loc[relevant_trials[DataParser.IS_TASK_RELEVANT_COL] == is_tr, :]
                trial_samps = trial_samps.loc[trial_samps[DataParser.TRIAL].isin(list(relevant_trials[DataParser.TRIAL_NUMBER])), :]
                # SD156: trial_samps was empty as all it had in this condition was BLINKS (no pupil size). garbage in -> garbage out
                if trial_samps.empty:
                    print(f"Subject {sub_data_path} has no samples remaining with task relevance {is_tr}")
                    continue
                else:
                    """
                    *** PUPIL SIZE NORMALIZATION ***
                    In the process of normalizing pupil size, we normalize the size PER SUBJECT, across all trials
                    of that subjects (i.e., across all EPOCHs). We do that by calculating the AVERAGE of all pupil 
                    size samples that are within an epoch. Then, each sample is DIVIDED (*not* substracted!) by the 
                    subject's own mean. 
                    """
                    trial_samps[f"{DataParser.REAL_PUPIL}Norm"] = trial_samps[f"{DataParser.REAL_PUPIL}"] / sub_mean
                    trial_samps = calculate_sample_time_in_trial(relevant_trials, trial_samps, [group_with, DataParser.STIM_TYPE_COL])

                # calculate for each time point the MEAN PUPIL SIZE
                if not do_each_category:
                    trial_means = trial_samps.groupby([TIME_IN_EPOCH, group_with]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, group_with, f"{DataParser.REAL_PUPIL}Norm"]]
                else:
                    trial_means = trial_samps.groupby([TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
                    trial_means = trial_means.loc[:, [TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL, f"{DataParser.REAL_PUPIL}Norm"]]
                trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
                trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
                trial_means.loc[:, MODALITY] = modality
                trial_means.loc[:, DataParser.IS_TASK_RELEVANT_COL] = is_tr
                only_mod_dfs_list.append(trial_means)

            mod_mean_df = pd.concat(only_mod_dfs_list)
            all_tr_list.append(mod_mean_df)

        all_mean_df = pd.concat(all_tr_list)
        total_df_list.append(all_mean_df)

    # AVERAGE all subjects in this modality
    all_sub_means_df = pd.concat(total_df_list)

    """
    For each modality, plot a lineplot where X = time, Y = average pupil size for a duration category (short/medium/long)
    across subjects (i.e., one line = one average across subjects, not weighted)
    """
    if not do_each_category:
        line_plot_per_mod(data=all_sub_means_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=f"{DataParser.REAL_PUPIL}Norm", y_col_name="Average Pupil Size (Normalized)",
                          hue_col=DataParser.STIM_DUR_PLND_SEC, hue_col_name="Stimulus Duration",
                          title_name="Average Pupil Size in Trial", global_y_max=False,
                          save_path=save_path, save_name=save_name,
                          hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP)
    else:
        for category in STIM_TYPES:
            df_cat = all_sub_means_df[all_sub_means_df[DataParser.STIM_TYPE_COL] == category]
            line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                              y_col=f"{DataParser.REAL_PUPIL}Norm", y_col_name="Average Pupil Size (Normalized)",
                              hue_col=hue_col, hue_col_name=hue_col_name, global_y_max=False,
                              title_name="Average Pupil Size in Trial",
                              save_path=save_path, save_name=f"{save_name}_{category}",
                              hue_names_map=hue_names_map, palette=palette)

    return sub_means_dict


def fix_dist_from_center_pixels(subs_dict, save_path, do_each_category=False, group_with=DataParser.STIM_DUR_PLND_SEC):
    """
    THE SAME AS fix_dist_from_center BUT IN RAW PIXELS!
    """
    if group_with == DataParser.STIM_DUR_PLND_SEC:
        hue_col = group_with
        hue_col_name = "Stimulus Duration"
        save_name = f"fix_dist_line"
        hue_names_map = STIMULUS_DURATION_NAME_MAP
        palette = STIMULUS_DURATION_MAP
    else:
        hue_col = group_with
        hue_col_name = "Stimulus Relevance"
        save_name = f"fix_dist_line_tr"
        hue_names_map = STIMULUS_RELEVANCE_NAME_MAP
        palette = STIMULUS_RELEVANCE_MAP

    all_dfs_list = []
    for modality in subs_dict.keys():
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            trial_fix_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]
            trial_fix_samples = trial_fix_samples.loc[trial_fix_samples[DataParser.REAL_FIX] == True,
                                :]  # only REAL fixations
            trial_fix_samples = trial_fix_samples.loc[trial_fix_samples[DataParser.TRIAL] != -1,
                                :]  # ones that are within a TRIAL epoch (not between trials)
            analyzed_eye = sub_data[PARAMS][DataParser.EYE]
            trials = sub_data[TRIAL_INFO]
            trial_fix_samples = calculate_sample_time_in_trial(trials, trial_fix_samples, [DataParser.STIM_DUR_PLND_SEC,
                                                                                           DataParser.IS_TASK_RELEVANT_COL,
                                                                                           DataParser.STIM_TYPE_COL])

            # collapse across stimulus duration groups, at each timepoint (from epoch start to end) average
            if not do_each_category:
                trial_means = trial_fix_samples.groupby([TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:,
                              [TIME_IN_EPOCH, DataParser.STIM_DUR_PLND_SEC, f"{analyzed_eye}{CENTER_DIST_PIX}"]]
            else:
                trial_means = trial_fix_samples.groupby([TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL]).mean(
                    numeric_only=True).reset_index()
                trial_means = trial_means.loc[:,
                              [TIME_IN_EPOCH, group_with, DataParser.STIM_TYPE_COL, f"{analyzed_eye}{CENTER_DIST_PIX}"]]

            trial_means.loc[:, SUBJECT] = sub_data[PARAMS]['SubjectName']
            trial_means.loc[:, LAB] = sub_data[PARAMS][ET_param_manager.SUBJECT_LAB]
            trial_means.loc[:, MODALITY] = modality
            trial_means.rename(columns={f"{analyzed_eye}{CENTER_DIST_PIX}": f"{CENTER_DIST_PIX}"}, inplace=True)
            only_mod_dfs_list.append(trial_means)  # add this subject data to the total
        # summarize this modality - average across all subjects' averages (and calculate std)
        mod_mean_df = pd.concat(only_mod_dfs_list)
        all_dfs_list.append(mod_mean_df)

    # Now, we do the all subs plot
    all_mean_df = pd.concat(all_dfs_list)

    if not do_each_category:
        line_plot_per_mod(data=all_mean_df, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                          y_col=CENTER_DIST_PIX, y_col_name="Average Distance from Center (Degrees VA)",
                          hue_col=group_with, hue_col_name=hue_col_name,
                          title_name="Average Fixation Distance from Center in Trial (PIXELS)",
                          save_path=save_path, save_name=f"{save_name}_PIX",
                          hue_names_map=STIMULUS_DURATION_NAME_MAP, palette=STIMULUS_DURATION_MAP,
                          global_y_min=0, epsilon=0.1)
    else:
        for category in STIM_TYPES:
            df_cat = all_mean_df[all_mean_df[DataParser.STIM_TYPE_COL] == category]
            line_plot_per_mod(data=df_cat, x_col=TIME_IN_EPOCH, x_col_name="Time (ms)",
                              y_col=CENTER_DIST_PIX,
                              y_col_name="Average Distance from Center (Degrees VA)",
                              hue_col=hue_col, hue_col_name=hue_col_name,
                              title_name=f"Average Fixation Distance from Center in **{category}** Trials (PIXELS)",
                              save_path=save_path, save_name=f"{save_name}_{category}_PIX",
                              hue_names_map=hue_names_map, palette=palette,
                              global_y_min=0, epsilon=0.1)

    return


def get_minimal_dims_va(subs_list):
    min_va_col = 100000
    min_va_row = 100000
    for sub_data_path in subs_list:
        fl = open(sub_data_path, 'rb')
        sub_data = pickle.load(fl)
        fl.close()

        screen_dims = sub_data[PARAMS]['ScreenResolution']
        screen_rows = screen_dims[1] * sub_data[PARAMS]['DegreesPerPix']  # the screen HEIGHT is like "rows" in dataframe
        screen_cols = screen_dims[0] * sub_data[PARAMS]['DegreesPerPix']  # the screen WIDTH is like "columns"

        if min_va_col > screen_cols:
            min_va_col = screen_cols

        if min_va_row > screen_rows:
            min_va_row = screen_rows

    print(f"mins are col:{min_va_col}, row: {min_va_row}")
    return (min_va_col, min_va_row)


def fixation_plots(subs_dict, save_path, phase_name):
    print("--- Fixation spread (2D histogram) per window ---")
    #if phase_name == "phase3":
    #    minimal_dims_va = get_minimal_dims_va(subs_dict[ET_param_manager.ECOG])
    #else:
    #    minimal_dims_va = (24.670391061452513, 13.877094972067038) # This is because phase2 doesn't have ecog
    minimal_dims_va = (24.670391061452513, 13.877094972067038) # This is because phase2 doesn't have ecog

    for time in [DataParser.FIRST_WINDOW]:  # DataParser.TRIAL,
        print(time)
        max_val = 0
        for modality in subs_dict.keys():
            print(modality)
            max_val = fix_hist_mod(subs_dict[modality], modality, time, save_path, minimal_dims_va, max_val, phase_name, plot=True, square=True)

        """  DEPRECATED
        sub_list_total = []
        for modality in subs_dict.keys():
            for sub in subs_dict[modality]:
                sub_list_total.append(sub)
        max_val = fix_hist_mod(sub_list_total, "ALL", time, save_path, minimal_dims_va, max_val, phase_name, plot=True, square=True)
        """

    """
    DEPRECATED
    print("--- Histogram of trial counts in each fixation distance from the center (degrees VA) on the X and Y axes ---")
    for modality in subs_dict.keys():
        print(modality)
        fixation_dist_hist_mod(subs_dict[modality], modality, save_path)

    print("--- Fixation distance from center over time (=epoch) IN PIXELS ---")
    fix_dist_from_center_pixels(subs_dict, save_path)
    fix_dist_from_center_pixels(subs_dict, save_path, do_each_category=True, group_with=DataParser.IS_TASK_RELEVANT_COL)

    print("--- Fixation distance from center over time (=epoch) ---")
    fix_dist_from_center(subs_dict, save_path)
    fix_dist_from_center(subs_dict, save_path, do_each_category=True, group_with=DataParser.IS_TASK_RELEVANT_COL)
    """

    return


def filter_out_tobii_subs(subs_dict):
    """
    The subjects who do not have pupil data are filtered out, as they don't have that data.
    These are only SF subjects. Therefore, we will ignore them in the pupil and blink data analyses.
    """
    relevant_subs = {}
    counter = 0
    filtered_out_counter = 0
    for modality in subs_dict.keys():  # for each modality
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            counter += 1
            if sub_data[PARAMS][ET_param_manager.SUBJECT_LAB] == 'SF':
                filtered_out_counter += 1
                continue
            if modality not in relevant_subs:
                relevant_subs[modality] = []
            relevant_subs[modality].append(sub_data_path)

    print(
        f"{filtered_out_counter} tobii subjects were filtered out of {counter}; in pupil and blink analyses, {counter - filtered_out_counter} subjects are analyzed")
    return relevant_subs


def filter_out_bad_phase(subs_dict, beh_data_path):
    beh_df = pd.read_csv(os.path.join(beh_data_path, "quality_checks_phase3.csv"))
    relevant_subs = {}
    counter = 0
    filtered_out_counter = 0
    for modality in subs_dict.keys():  # for each modality
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            counter += 1
            if beh_df.loc[(beh_df['subCode'] == sub_data[PARAMS]['SubjectName']) & (beh_df['Is_Valid?'] == True), :].shape[0] == 0:
                filtered_out_counter += 1
                continue
            if modality not in relevant_subs:
                relevant_subs[modality] = []
            relevant_subs[modality].append(sub_data_path)

    print(
        f"{filtered_out_counter} phase 3 subjects were filtered out of {counter}; {counter - filtered_out_counter} subjects are analyzed")
    return relevant_subs


def single_plot_per_tr_category(data, y_col, ymin, ymax, plot_title, plot_y_label, save_path, save_name, skip=1):
    """
    This method plots a single plot where X = stimulus duration, HUE = stimulus category, and Y is a parameter.
    """
    # X axis params
    stim_xs = {FACE: -0.15, OBJ: -0.05, LETTER: 0.05, FALF: 0.15}
    trs = {False: "Task Irrelevant", True: "Task Relevant"}
    tr_locs = {False: 0.5, True: 1.0}
    # plot
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for tr in trs:
        df_tr = data[data[DataParser.IS_TASK_RELEVANT_COL] == tr]
        for stim_type in STIM_HUE_DICT:
            df_stim = df_tr[df_tr[DataParser.STIM_TYPE_COL] == stim_type]
            x_loc = tr_locs[tr] + stim_xs[stim_type]
            y_vals = df_stim[y_col]
            # plot violin
            violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.17, showmeans=True, showextrema=False,
                                    showmedians=False)
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(STIM_HUE_DICT[stim_type])
            # change the color of the mean lines (showmeans=True)
            violin['cmeans'].set_color("black")
            violin['cmeans'].set_linewidth(2)
            # control the length like before
            m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
            # then scatter
            scat_x = (np.ones(len(y_vals)) * (x_loc - 0.05)) + (np.random.rand(len(y_vals)) * 0.048)
            plt.scatter(x=scat_x, y=y_vals, marker="o", color=STIM_HUE_DICT[stim_type], s=60, alpha=0.5,
                        edgecolor=STIM_HUE_DICT[stim_type])
    # cosmetics
    plt.xticks(ticks=list(tr_locs.values()), labels=list(trs.values()), fontsize=TICK_SIZE)
    plt.yticks([y for y in np.arange(ymin, ymax + (1 * skip), skip)], fontsize=TICK_SIZE)
    plt.title(plot_title, fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
    plt.ylabel(plot_y_label, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel("Task Relevance", fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    # The following two lines generate custom fake lines that will be used as legend entries:
    # markers = [plt.Line2D([0, 0], [0, 0], color=STIM_HUE_DICT[label], marker='o', linestyle='') for label in STIM_TITLE_DICT]
    # new_labels = [STIM_TITLE_DICT[label] for label in STIM_TITLE_DICT]
    # legend = plt.legend(markers, new_labels, title="Stimulus Category", markerscale=1, fontsize=TICK_SIZE - 2)
    # plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    plt.legend('', frameon=False)
    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000)
    del figure
    plt.close()
    gc.collect()
    return


def calc_df_stats_permutations(df, dependent_col, permutation_col_list):
    """
    This function calculates the mean and std of a single column (dependent_col) in a dataframe (df),
    across all combinations of column values in permutation_col_list.
    (e.g., for face x 1.5 dur x task relevant - average and sd of dependent_col.

    df: a dataframe containing all the columns in permutation_col_list PLUS a dependent variable column (dependent_col)
    dependent_col: the column with the data we want to average (sd) across some conditions (permutation_col_list)
    permutation_col_list: all the columns that by which the combinations should be calculated
    """
    lst = list()
    x = pd.DataFrame(columns=permutation_col_list)  # exists only in order to order the columns
    lst.append(x)
    total_mean = df[dependent_col].mean()
    total_std = df[dependent_col].std()
    means = pd.DataFrame({f"{dependent_col}_mean": [total_mean], f"{dependent_col}_std": [total_std]})
    lst.append(means)
    for i in range(1, len(permutation_col_list) + 1):
        for comb in list(itertools.combinations(permutation_col_list, i)):
            comb_list = list(comb)
            means = df.groupby(comb_list).mean(numeric_only=True).reset_index()
            stds = df.groupby(comb_list).std(numeric_only=True).reset_index()
            means[f"{dependent_col}_std"] = stds[dependent_col]
            means.rename({dependent_col: f"{dependent_col}_mean"}, axis=1, inplace=True)
            means = means.loc[:, comb_list+[f"{dependent_col}_mean", f"{dependent_col}_std"]]
            lst.append(means)

    result = pd.concat(lst)
    return result


def plot_first_model(subs_dict, save_path, phase_name):
    trials_df = pd.read_csv(os.path.join(save_path, f"lmm_df_{phase_name}.csv"))
    trials_df = trials_df[trials_df["timeWindow"] == DataParser.FIRST_WINDOW]

    dependent_vars = {f"{CENTER_DIST_DEGS}": "Fixation Distance from Center",
                      f"MaxAmp": "Saccade Amplitude", "NumBlinks": "Number of Blinks",
                      f"{DataParser.REAL_PUPIL}Norm": "Normalized Pupil Size",
                      f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}": "Median Fixation Distance from Center",
                      f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm": "Median Normalized Pupil Size"}

    dependent_vars_minmaxskip = {f"{CENTER_DIST_DEGS}": [0, 8, 1],
                                 f"MaxAmp": [0, 15, 5],
                                 "NumBlinks": [0, 1, 0.1],
                                 f"{DataParser.REAL_PUPIL}Norm": [0.5, 1.5, 0.1],
                                 f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}": [0, 8, 1],
                                 f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm": [0.5, 1.5, 0.1]}

    trials_df = trials_df.loc[:, ["subCode", "modality", "lab", DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL, f"{CENTER_DIST_DEGS}", f"{CENTER_DIST_DEGS}{ET_data_extraction.BL_CORRECTED}", f"MaxAmp", "NumBlinks", f"{DataParser.REAL_PUPIL}Norm",
                                 f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}", f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm"]]
    total_df = trials_df.groupby(["subCode", "lab", "modality", DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
    total_df.to_csv(os.path.join(save_path, f"first_catxtr_{phase_name}.csv"), index=False)

    for dep in dependent_vars:
        dep_min = dependent_vars_minmaxskip[dep][0]
        dep_max = dependent_vars_minmaxskip[dep][1]
        skip = dependent_vars_minmaxskip[dep][2]
        no_nan_df = total_df[~total_df[dep].isna()]
        permutations_res = calc_df_stats_permutations(total_df, dep, permutation_col_list=[DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL, "modality"])
        permutations_res.to_csv(os.path.join(save_path, f"first_{dep}_stats_{phase_name}.csv"), index=False)

        for modality in subs_dict.keys():
            mod_df = no_nan_df[no_nan_df["modality"] == modality]
            if not mod_df.empty:
                single_plot_per_tr_category(data=mod_df, y_col=dep, ymin=dep_min, ymax=dep_max,
                                            plot_title=f"{dependent_vars[dep]} by TR and Category - {modality}",
                                            plot_y_label=dep, save_path=save_path,
                                            save_name=f"first_{dep}_catxtr_{modality}_{phase_name}", skip=skip)

    for dep in dependent_vars:
        dep_min = dependent_vars_minmaxskip[dep][0]
        dep_max = dependent_vars_minmaxskip[dep][1]
        skip = dependent_vars_minmaxskip[dep][2]
        no_nan_df = total_df[~total_df[dep].isna()]
        single_plot_per_tr_category(data=no_nan_df, y_col=dep, ymin=dep_min, ymax=dep_max,
                                    plot_title=f"{dependent_vars[dep]} by TR and Category - ALL",
                                    plot_y_label=dep, save_path=save_path,
                                    save_name=f"first_{dep}_catxtr_{phase_name}", skip=skip)
    return


def plot_second_model(subs_dict, save_path, phase_name):
    trials_df = pd.read_csv(os.path.join(save_path, f"lmm_df_{phase_name}.csv"))
    trials_df = trials_df[trials_df[DataParser.STIM_DUR_PLND_SEC] == 1.5]

    dependent_vars = {f"{CENTER_DIST_DEGS}": "Fixation Distance from Center",
                      f"MaxAmp": "Saccade Amplitude", "NumBlinks": "Number of Blinks",
                      f"{DataParser.REAL_PUPIL}Norm": "Normalized Pupil Size",
                      f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}": "Median Fixation Distance from Center",
                      f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm": "Median Normalized Pupil Size"}

    dependent_vars_minmaxskip = {f"{CENTER_DIST_DEGS}": {DataParser.FIRST_WINDOW: [0, 8, 1],
                                                         DataParser.SECOND_WINDOW: [0, 17, 1],  # TODO: SA125, w/o it 8
                                                         DataParser.THIRD_WINDOW: [0, 11, 1]},
                                 f"MaxAmp": {DataParser.FIRST_WINDOW: [0, 25, 5],
                                             DataParser.SECOND_WINDOW: [0, 25, 5],
                                             DataParser.THIRD_WINDOW: [0, 25, 5]},
                                 "NumBlinks": {DataParser.FIRST_WINDOW: [0, 1, 0.1],
                                               DataParser.SECOND_WINDOW: [0, 1, 0.1],
                                               DataParser.THIRD_WINDOW: [0, 1, 0.1]},
                                 f"{DataParser.REAL_PUPIL}Norm": {DataParser.FIRST_WINDOW: [0, 1.5, 0.1],
                                                                  DataParser.SECOND_WINDOW: [0, 1.5, 0.1],
                                                                  DataParser.THIRD_WINDOW: [0, 1.5, 0.1]},
                                 f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}": {DataParser.FIRST_WINDOW: [0, 8, 1],
                                                                                    DataParser.SECOND_WINDOW: [0, 17, 1],  # TODO: SA125, w/o it 8
                                                                                    DataParser.THIRD_WINDOW: [0, 11, 1]},
                                 f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm": {DataParser.FIRST_WINDOW: [0, 1.5, 0.1],
                                                                                             DataParser.SECOND_WINDOW: [0, 1.5, 0.1],
                                                                                             DataParser.THIRD_WINDOW: [0, 1.5, 0.1]}}

    trials_df = trials_df.loc[:,["subCode", "modality", "lab", "timeWindow", DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL, f"{CENTER_DIST_DEGS}",  f"{CENTER_DIST_DEGS}{ET_data_extraction.BL_CORRECTED}", f"MaxAmp", "NumBlinks", f"{DataParser.REAL_PUPIL}Norm",
                                 f"{ET_data_extraction.MEDIAN}{CENTER_DIST_DEGS}", f"{ET_data_extraction.MEDIAN}{DataParser.REAL_PUPIL}Norm"]]
    total_df = trials_df.groupby(["subCode", "lab", "modality", "timeWindow", DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL]).mean(numeric_only=True).reset_index()
    total_df.to_csv(os.path.join(save_path, f"long_catxtr_{phase_name}.csv"), index=False)

    for dep in dependent_vars:
        no_nan_df = total_df[~total_df[dep].isna()]
        permutations_res = calc_df_stats_permutations(total_df, dep, permutation_col_list=[DataParser.IS_TASK_RELEVANT_COL, DataParser.STIM_TYPE_COL, "modality", "timeWindow"])
        permutations_res.to_csv(os.path.join(save_path, f"long_{dep}_stats_{phase_name}.csv"), index=False)

        for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
            window_df = no_nan_df[no_nan_df["timeWindow"] == window]
            dep_min = dependent_vars_minmaxskip[dep][window][0]
            dep_max = dependent_vars_minmaxskip[dep][window][1]
            skip = dependent_vars_minmaxskip[dep][window][2]
            for modality in subs_dict.keys():
                mod_df = window_df[window_df["modality"] == modality]
                if not mod_df.empty:
                    single_plot_per_tr_category(data=mod_df, y_col=dep, ymin=dep_min, ymax=dep_max,
                                                plot_title=f"{dependent_vars[dep]} by TR and Category in {window}- {modality}",
                                                plot_y_label=dep, save_path=save_path,
                                                save_name=f"long_{window}_{dep}_catxtr_{modality}_{phase_name}", skip=skip)

    for dep in dependent_vars:
        no_nan_df = total_df[~total_df[dep].isna()]
        for window in [DataParser.FIRST_WINDOW, DataParser.SECOND_WINDOW, DataParser.THIRD_WINDOW]:
            window_df = no_nan_df[no_nan_df["timeWindow"] == window]
            dep_min = dependent_vars_minmaxskip[dep][window][0]
            dep_max = dependent_vars_minmaxskip[dep][window][1]
            skip = dependent_vars_minmaxskip[dep][window][2]
            single_plot_per_tr_category(data=window_df, y_col=dep, ymin=dep_min, ymax=dep_max,
                                        plot_title=f"{dependent_vars[dep]} by TR and Category in {window}- ALL",
                                        plot_y_label=dep, save_path=save_path,
                                        save_name=f"long_{window}_{dep}_catxtr_{phase_name}", skip=skip)
    return


def per_sub_means(save_path, phase_name):
    trials_df = pd.read_csv(os.path.join(save_path, f"lmm_df_{phase_name}.csv"))
    per_sub_means = trials_df.groupby(["subCode", "modality", "lab"]).mean(numeric_only=True).reset_index()
    per_sub_means.to_csv(os.path.join(save_path, f"lmm_df_meanPerSub_{phase_name}.csv"), index=False)
    return


def extract_trial_stats(subs_dict, pupil_sub_means, save_path, phase_name):
    all_dfs_list = []
    for modality in subs_dict.keys():
        print(modality)
        only_mod_dfs_list = []
        for sub_data_path in subs_dict[modality]:
            print(sub_data_path)
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            eye = sub_data[PARAMS]["Eye"]
            trial_samples = sub_data[ET_DATA_DICT][DataParser.DF_SAMPLES]

            relevant_trials = sub_data[TRIAL_INFO]
            trial_samples = trial_samples.loc[trial_samples[DataParser.TRIAL] != -1, :]  # ones that are within a TRIAL epoch (not between trials)

            unique_sacc_per_trial = trial_samples.groupby([DataParser.TRIAL]).sacc_number.nunique()
            unique_blinks_per_trial = trial_samples.groupby([DataParser.TRIAL]).blink_number.nunique()

            trial_samples = trial_samples.loc[trial_samples[f"{eye}{DataParser.HERSHMAN_PAD}"].isna(), :]  # only REAL PUPIL SIZE
            trial_samples[f"{DataParser.REAL_PUPIL}Norm"] = trial_samples[DataParser.REAL_PUPIL] / pupil_sub_means[sub_data[PARAMS]['SubjectName']]
            pupil_mean = trial_samples.groupby([DataParser.TRIAL]).mean(numeric_only=True).reset_index()

            for index, trial in pupil_mean.iterrows():
                relevant_trials.loc[relevant_trials[DataParser.TRIAL_NUMBER] == trial[DataParser.TRIAL], f"{DataParser.REAL_PUPIL}Norm"] = trial[f"{DataParser.REAL_PUPIL}Norm"]

            relevant_trials.loc[:, f"NumSaccs"] = unique_sacc_per_trial
            relevant_trials.loc[:, f"NumBlinks"] = unique_blinks_per_trial

            only_mod_dfs_list.append(relevant_trials)
        # summarize this modality - average across all subjects' averages (and calculate std)
        mod_mean_df = pd.concat(only_mod_dfs_list)
        all_dfs_list.append(mod_mean_df)

    total_df = pd.concat(all_dfs_list)
    mod_df = total_df.groupby(["modality"]).mean(numeric_only=True).reset_index()
    mod_std = total_df.groupby(["modality"]).std(numeric_only=True).reset_index()
    mod_df[f"NumSaccs_std"] = mod_std["NumSaccs"]
    mod_df[f"NumBlinks_std"] = mod_std["NumBlinks"]
    mod_df[f"{DataParser.REAL_PUPIL}Norm_std"] = mod_std[f"{DataParser.REAL_PUPIL}Norm"]

    total_means = total_df[["NumSaccs", "NumBlinks", f"{DataParser.REAL_PUPIL}Norm"]].mean()
    total_stds = total_df[["NumSaccs", "NumBlinks", f"{DataParser.REAL_PUPIL}Norm"]].std()
    res_df = pd.DataFrame(columns=["NumSaccs", "NumBlinks", f"{DataParser.REAL_PUPIL}Norm", "NumSaccs_std", "NumBlinks_std", f"{DataParser.REAL_PUPIL}Norm_std"])
    res_df.loc[0] = list(total_means) + list(total_stds)
    res_df = pd.concat([res_df, mod_df])
    res_df.to_csv(os.path.join(save_path, f"stats_trial_{phase_name}.csv"), index=False)

    return


def filter_out_bad_data(subs_dict):
    relevant_subs = {}
    for modality in subs_dict.keys():  # for each modality
        for sub_data_path in subs_dict[modality]:
            fl = open(sub_data_path, 'rb')
            sub_data = pickle.load(fl)
            fl.close()
            if sub_data[PARAMS]["SubjectName"] in INVALID_LIST:
                # DO NOT INCLUDE THIS SUBJECT IN FURTHER ET analyses!
                continue
            if modality not in relevant_subs:
                relevant_subs[modality] = []
            relevant_subs[modality].append(sub_data_path)

    print(f"Filtered out participants with bad ET data.")
    return relevant_subs


def process_data(subs_dict, beh_data_path, save_path, phase_name, valid_beh=True):

    subs_dict = {modality: subs_dict[modality] for modality in list(subs_dict.keys()) if len(subs_dict[modality]) > 0}

    if valid_beh:
        # then, filter out all the subjects that in the behavioral analysis were found to be invalid
        print("-------- ONLY VALID in PHASE 3 --------")
        subs_dict = filter_out_bad_phase(subs_dict, beh_data_path)

    print("--------Taking out subjects with BAD ET DATA ---------")
    subs_dict = filter_out_bad_data(subs_dict)

    print("--------Pupil & blink are NOT done for Tobii subs: filter out ---------")
    filtered_subs = filter_out_tobii_subs(subs_dict)
    fl = open(os.path.join(save_path, f"temp_filtered_subs_{phase_name}.pickle"), 'wb')
    pickle.dump(filtered_subs, fl)
    fl.close()

    processed_subs_df = pd.DataFrame({"subPath": sorted({x for v in filtered_subs.values() for x in v})})
    processed_subs_df.to_csv(os.path.join(save_path, f"filtered_subs_{phase_name}.csv"), index=False)

    print("--------PUPIL PLOTS---------")
    pupil_sub_means = pupil_plot_no_tr(filtered_subs, save_path, phase_name)

    print("--------LMM DATA---------")
    lmm_data(filtered_subs, pupil_sub_means, save_path, phase_name)
    # calculate saccade proportion overall
    extract_trial_stats(filtered_subs, pupil_sub_means, save_path, phase_name)

    print("-------- RAINCLOUDPLOTS --------")
    plot_first_model(filtered_subs, save_path=save_path, phase_name=phase_name)
    plot_second_model(filtered_subs, save_path=save_path, phase_name=phase_name)
    per_sub_means(save_path=save_path, phase_name=phase_name)

    print("--------BLINK PLOTS---------")
    blink_plot(filtered_subs, save_path, phase_name)
    """
    DEPRECATED
    blink_plot_trial_relevent(filtered_subs, save_path)
    """


    # now for the unfiltered analyses
    print(f"--------SACCADE PLOTS---------")
    # saccade num plot
    sacc_num_plot(filtered_subs, save_path, phase_name)
    # polar plot
    saccade_ciruclar_plot(filtered_subs, save_path, phase_name)
    # amplitude in time plot (SMOOTHED)
    saccade_amp_no_category(filtered_subs, save_path, phase_name)
    # Deprecated
    # saccade_amp(subs_dict, save_path, phase_name)

    """
    DEPRECATED
    sacc_num_plot_trial_relevent(subs_dict, save_path)
    saccade_ciruclar_relevance_plot(subs_dict, save_path)
    saccade_amp_relevance(subs_dict, save_path)
    """

    print(f"--------FIXATION PLOTS---------")
    fixation_plots(filtered_subs, save_path, phase_name)
    return
