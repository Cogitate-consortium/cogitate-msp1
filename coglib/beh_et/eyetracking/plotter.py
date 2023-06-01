import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import os
import pandas as pd
import numpy as np
import warnings
import random
import math
import gc

""" Plotting Module

This module includes all the methods used for plotting data. 

@authors: RonyHirsch
"""

warnings.filterwarnings('ignore')
PNG = ".png"

W = 10
H = 7.5
DPI = 1000
# the amount of (horizontal) jitter to create in the plot so that points with the same X,Y won't overlap
JITTER_WIDTH = 0.08
VIOLIN_OFFSET = 0.15

F_HEADER = 16
F_AXES_TITLE = 14
F_HORIZ_LINES = 11
XLABELPAD = 20
YLABELPAD = 20


def hist_plot(data, x_col, title, x_label, y_label, save_path, save_name):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    sns.histplot(data=data, x=x_col, bins=100)
    plt.title(title, fontsize=15)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 8)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", bbox_inches="tight")
    counts, bin_edges = np.histogram(data[x_col], bins=100)
    return


def get_min_max(dict_of_dfs):
    """
    Given a dictionary of dataframes of numbers, finds the minimum and maximum values across all dfs.
    :param dict_of_dfs: ictionary of dataframes of numbers
    :return: min, max
    """
    total_min = 10000000
    total_max = -10000000
    for key in dict_of_dfs.keys():
        try:
            tmp_min = min(dict_of_dfs[key].min())
            tmp_max = max(dict_of_dfs[key].max())
        except TypeError:
            tmp_min = dict_of_dfs[key].min()
            tmp_max = dict_of_dfs[key].max()
        if total_min > tmp_min > 0:  # the > 0 part is because 0 will be removed in the heatmap itself
            total_min = tmp_min
        if tmp_max > total_max:
            total_max = tmp_max
    return total_min, total_max


def err_line_plot(cond_data, x_label, y_label, title, save_path, vertical_lines=None, x_conversion=None, x_zero=None,
                  num_of_x_ticks=10, y_min=None, y_max=None, y_intervals=None,
                  color_list=None, figsize=(8, 6), dpi=500, title_size=15, label_size=10):
    f = plt.figure(figsize=figsize)
    sns.set_style('dark')

    if isinstance(cond_data, pd.DataFrame):  # if we are plotting 1 line of mean + sem
        cond_df = cond_data
        if color_list is None:
            # get a color palette - and randomly choose one color
            randrange = 10
            linepal = sns.color_palette("colorblind", n_colors=randrange)
            ind = random.randint(0, randrange-1)
            color = linepal[ind]
        else:
            color = color_list[0]
        g_ax = sns.lineplot(x=list(cond_df.index), y=cond_df['mean'], linewidth=0.6, color=color,
                            label="mean")
        g_ax.fill_between(x=list(cond_df.index), y1=cond_df['mean'] - cond_df['sem'],
                          y2=cond_df['mean'] + cond_df['sem'], facecolor=color, alpha=0.35)
        actual_ymin = min(cond_df['mean'] + cond_df['sem'])
        actual_ymax = max(cond_df['mean'] + cond_df['sem'])

    elif isinstance(cond_data, dict):  # if we have several lines to plot on the same error line plot
        if color_list is None:
            # get a color palette
            linepal = sns.color_palette("colorblind", n_colors=len(list(cond_data.keys())))
        else:
            linepal = color_list
        actual_ymin = 10000000000000
        actual_ymax = -1000000000000
        for sub_cond_ind in range(len(list(cond_data.keys()))):
            cond_df = cond_data[list(cond_data.keys())[sub_cond_ind]]
            g_ax = sns.lineplot(x=list(cond_df.index), y=cond_df['mean'], linewidth=0.6, color=linepal[sub_cond_ind],
                                label=list(cond_data.keys())[sub_cond_ind])
            g_ax.fill_between(x=list(cond_df.index), y1=cond_df['mean'] - cond_df['sem'],
                              y2=cond_df['mean'] + cond_df['sem'], facecolor=linepal[sub_cond_ind], alpha=0.35)
            tmp_ymin = min(cond_df['mean'] + cond_df['sem'])
            tmp_ymax = max(cond_df['mean'] + cond_df['sem'])
            if tmp_ymin < actual_ymin:
                actual_ymin = tmp_ymin
            if tmp_ymax > actual_ymax:
                actual_ymax = tmp_ymax

    if vertical_lines is not None:
        for txt in vertical_lines.keys():
            g_ax.axvline(x=vertical_lines[txt], color='r', linewidth=0.4, ls='--', alpha=0.7)
            ymax = g_ax.get_ylim()[1]
            #g_ax.text(vertical_lines[txt] + 10, (y_max + ymax) / 2, txt, rotation=90, color='k', fontsize='small')
            g_ax.text(vertical_lines[txt] + 10, (y_max + ymax) / 2, txt, color='k', fontsize='small')

    # the next part is in order to convert ORIGINAL X axis units (say, samples) to NEW X axis units (say, time)
    # what we do is convert the LABELS ONLY, using the original x as x-ticks
    if x_conversion is not None:  # if we need to convert
        x_in_new_units = [(x * 1000) / x_conversion for x in list(cond_df.index)]
    else:
        x_in_new_units = list(cond_df.index)

    if x_zero is not None:  # if we want a specific point on X to be considered "0" (time-lock)
        if x_conversion is not None:
            x_zero = (x_zero * 1000) / x_conversion
        else:
            x_zero = x_zero

        x_in_new_offset = [x - x_zero for x in x_in_new_units]
    else:
        x_in_new_offset = x_in_new_units

    # modify xticks accordingly
    tick_size = label_size-2
    x_ticks = list(cond_df.index)
    plt.xticks(ticks=x_ticks, labels=x_in_new_offset, fontsize=tick_size)
    # this shows only the amount of x axis points set in num_of_x_ticks:
    f.gca().xaxis.set_major_locator(plt.MaxNLocator(num_of_x_ticks))

    # Y ticks and axis scale
    if y_min is not None and y_max is not None and y_intervals is not None:
        plt.ylim(min(y_min, actual_ymin), max(y_max, actual_ymax), y_intervals)
    elif y_min is not None and y_max is not None:
        plt.ylim(min(y_min, actual_ymin), max(y_max, actual_ymax))
    elif y_min is not None:
        plt.ylim(bottom=min(y_min, actual_ymin))
    elif y_max is not None:
        plt.ylim(top=max(y_max, actual_ymax))
    plt.yticks(fontsize=label_size-2)

    # title, axes, stuff like that
    plt.title(title, fontsize=title_size, fontweight='bold')
    plt.xlabel(x_label, fontsize=label_size, fontweight='bold')
    plt.ylabel(y_label, fontsize=label_size, fontweight='bold')
    plt.legend(fontsize=tick_size, loc='upper right')
    f.tight_layout()

    save_name = title.replace(" ", "") + PNG
    f.savefig(os.path.join(save_path, save_name), dpi=dpi, format='png')
    f.savefig(os.path.join(save_path, save_name), dpi=dpi, format='svg')
    return plt


def err_line_plot_multidf(cond_df, x_col, y_col, x_label, y_label, title, save_path, save_name, vertical_lines=None,
                          num_of_x_ticks=10, y_min=None, y_max=None, y_intervals=None,
                          subcond_col=None, subcond_color_dict=None, sd_col=None,
                          figsize=(8, 6), dpi=500, title_size=15, label_size=10):
    """
    :param cond_df: dataframe in which each line is a sample
    :param x_col: the X-axis data
    :param y_col: the Y-axis data
    :param subcond_col: the column by which to split the data to multiple lines; each unique value in this column will
    be a separate line
    :param subcond_color_dict: the colors for wach unique value in subcond_col
    :param x_label:
    :param y_label:
    :param title:
    :param save_path:
    :param vertical_lines:
    :param x_conversion:
    :param x_zero:
    :param num_of_x_ticks:
    :param y_min:
    :param y_max:
    :param y_intervals:
    :param figsize:
    :param dpi:
    :param title_size:
    :param label_size:
    :return:
    """
    f = plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    # Set the color scheme
    if subcond_color_dict is None:
        # get a color palette
        keys = list(cond_df[subcond_col].unique())
        values = sns.color_palette("colorblind", n_colors=len(list(cond_df[subcond_col].unique())))
        linepal = dict(zip(keys, values))
    else:
        linepal = subcond_color_dict

    # Plot the data
    actual_ymin = 10000000000000
    actual_ymax = -1000000000000
    for subcond in cond_df[subcond_col].unique():
        subcond_df = cond_df[cond_df[subcond_col] == subcond]
        # Plot this sub-condition's line
        g_ax = sns.lineplot(x=subcond_df[x_col], y=subcond_df[y_col], linewidth=0.6, color=linepal[subcond], label=subcond)
        # Plot std
        if sd_col is not None:
            g_ax.fill_between(x=subcond_df[x_col], y1=subcond_df[y_col] - subcond_df[sd_col],
                              y2=subcond_df[y_col] + subcond_df[sd_col], facecolor=linepal[subcond], alpha=0.35)
            # Set Y-axis bounds
            tmp_ymin = min(subcond_df[y_col] - subcond_df[sd_col])
            tmp_ymax = max(subcond_df[y_col] + subcond_df[sd_col])
        else:
            tmp_ymin = min(subcond_df[y_col])
            tmp_ymax = max(subcond_df[y_col])
        if tmp_ymin < actual_ymin:
            actual_ymin = tmp_ymin
        if tmp_ymax > actual_ymax:
            actual_ymax = tmp_ymax

    # Y ticks and axis scale
    if y_min is not None and y_max is not None and y_intervals is not None:
        plt.ylim(min(y_min, actual_ymin), max(y_max, actual_ymax), y_intervals)
    elif y_min is not None and y_max is not None:
        plt.ylim(min(y_min, actual_ymin), max(y_max, actual_ymax))
    elif y_min is not None:
        plt.ylim(bottom=min(y_min, actual_ymin))
    elif y_max is not None:
        plt.ylim(top=max(y_max, actual_ymax))
    plt.yticks(fontsize=label_size - 2)
    y_max = actual_ymax

    # Handle x and y ticks
    tick_size = label_size - 2
    plt.xticks(ticks=subcond_df[x_col], fontsize=tick_size)
    # this shows only the amount of x axis points set in num_of_x_ticks:
    f.gca().xaxis.set_major_locator(plt.MaxNLocator(num_of_x_ticks))

    # Plot vertical lines
    if vertical_lines is not None:
        for txt in vertical_lines.keys():
            g_ax.axvline(x=vertical_lines[txt], color='r', linewidth=0.4, ls='--', alpha=0.7)
            ymax = g_ax.get_ylim()[1]
            #g_ax.text(vertical_lines[txt] + 10, (y_max + ymax) / 2, txt, rotation=90, color='k', fontsize='small')
            g_ax.text(vertical_lines[txt] + 10, (y_max + ymax) / 2, txt, color='k', fontsize='small')

    # title, axes, stuff like that
    plt.title(title, fontsize=title_size, fontweight='bold')
    plt.xlabel(x_label, fontsize=label_size, fontweight='bold')
    plt.ylabel(y_label, fontsize=label_size, fontweight='bold')
    plt.legend(fontsize=tick_size, loc='upper right')
    f.tight_layout()

    plot_name = f"{save_name}{PNG}"
    f.savefig(os.path.join(save_path, plot_name), dpi=dpi, format='png')
    return plt


def simple_lineplot(data, line_key, shadow_key, x_col, y_col, x_label, y_label, title, save_path, save_name, colors=None,
                    y_min=0, y_max=1, y_ticks=0.1, x_min=0, x_max=1, x_ticks=0.1, label_size=10, sub_cond_name_dict=None,
                    vertical_line=None):
    """
    This plotter expects either:
    - a dictionary where keys are [line_key, shadow_key] and values are dataframes (identical columns across the 2)
    in which we have the x_col, y_col to plot. (e.g., "line_key" is "mean" and "shadow_key" is std, and we have at least
    2 columns in each dataframe, s.t the mean-df will be plotted as a line and the std-df will be plotted as the shadow
    around it)
    OR
    - a dictionary OF such dictionaries, i.e, several {line_key:df, shadow_key:df} dictionaries to be plotted in the
    same figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    if line_key not in data.keys():  # several lineplots in the same fig
        if not colors:
            colors = sns.color_palette("colorblind", len(list(data.keys())))
        i = 0
        for sub_cond in data:
            if data[sub_cond] is not None:
                if sub_cond_name_dict is None:
                    sub_cond_name = sub_cond.capitalize().replace("_", " ")
                else:
                    sub_cond_name = sub_cond_name_dict[sub_cond]
                data_line = data[sub_cond][line_key]
                ax.plot(data_line[x_col], data_line[y_col], c=colors[i], label=sub_cond_name)
                if shadow_key is not None:
                    shadow_line = data[sub_cond][shadow_key]
                    ax.fill_between(x=data_line[x_col], y1=data_line[y_col] - shadow_line[y_col],
                                    y2=data_line[y_col] + shadow_line[y_col],
                                    facecolor=colors[i], alpha=0.35)
                i += 1

    else:  # no sub conditions to plot
        if not colors:
            colors = sns.color_palette("colorblind", 1)
        data_line = data[line_key]
        ax.plot(data_line[x_col], data_line[y_col], c=colors[0], label="")
        if shadow_key is not None:
            shadow_line = data[shadow_key]
            ax.fill_between(x=data_line[x_col], y1=data_line[y_col] - shadow_line[y_col],
                            y2=data_line[y_col] + shadow_line[y_col],
                            facecolor=colors[0], alpha=0.35)

    # free up memory
    del data
    del data_line
    del shadow_line
    gc.collect()

    if vertical_line is not None:
        for line in vertical_line:
            plt.axvline(x=vertical_line[line]["x"], ymin=y_min, ymax=y_max, color=vertical_line[line]["c"], ls=vertical_line[line]["ls"], lw=vertical_line[line]["lw"])
            plt.text(x=vertical_line[line]["x"] + 100, y=1.00 - (2 * 0.1), s=line, fontsize=F_HORIZ_LINES + 2)

    if isinstance(y_ticks, list):
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks(np.arange(y_min, y_max, y_ticks))

    ax.set_xticks(np.arange(x_min, x_max, x_ticks))

    plt.title(title)
    plt.xlabel(x_label, fontsize=label_size, fontweight='bold')
    plt.ylabel(y_label, fontsize=label_size, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.01))

    plot_save_name = save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W, H)
    gc.collect()
    plt.savefig(os.path.join(save_path, f"line_{plot_save_name}.png"), dpi=DPI, bbox_inches="tight")
    return


def heatmaps(heatmap_data, x_label, y_label, title, save_path, n_rows=1, n_cols=1, target_focus_size=None,
             plot_axes_lines=True, with_pic=None, figsize=(8, 6), dpi=500, title_size=18, label_size=13, save_name=None,
             min_val=0.00, max_val=0.1):
    f, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if isinstance(heatmap_data, dict):
        heatmap_dict = heatmap_data
        if min_val is None or max_val is None:
            min_val, max_val = get_min_max(heatmap_dict)
        redundant = list()
        for i in range(n_rows):
            for j in range(n_cols):
                ind = i * n_cols + j
                if ind < len(list(heatmap_dict.keys())):
                    if n_rows == 1 or n_cols == 1:
                        axs = axes[ind]
                    else:
                        axs = axes[i, j]
                    heatmap(heatmap_df=heatmap_dict[list(heatmap_dict.keys())[ind]]["density"],
                            plot_title=list(heatmap_dict.keys())[ind],
                            plot_x_label=x_label, plot_y_label=y_label, title_size=label_size, min_val=min_val,
                            max_val=max_val,
                            target_focus_size=target_focus_size, plot_axes_lines=plot_axes_lines, with_pic=with_pic,
                            ax=axs)
                else:
                    redundant.append((i, j))

        # remove redundant subplots:
        for loc in redundant:
            f.delaxes(axes[loc[0]][loc[1]])

    elif isinstance(heatmap_data, np.ndarray) or isinstance(heatmap_data, pd.DataFrame):
        if min_val is None or max_val is None:
            if isinstance(heatmap_data, np.ndarray):
                plot_min = np.min(heatmap_data)
                plot_max = np.max(heatmap_data)
            else:
                plot_min = heatmap_data.to_numpy().min()
                plot_max = heatmap_data.to_numpy().max()
        else:
            plot_min = min_val
            plot_max = max_val
        heatmap(heatmap_df=heatmap_data, plot_title="", plot_x_label=x_label, plot_y_label=y_label,
                title_size=0, min_val=plot_min, max_val=plot_max, target_focus_size=target_focus_size,
                plot_axes_lines=plot_axes_lines, with_pic=with_pic)
        axes.set_ylabel(x_label, fontdict={'size': title_size-2, 'weight': 'normal'})
        axes.set_xlabel(y_label, fontdict={'size': title_size-2, 'weight': 'normal'})

    f.suptitle(title, fontsize=title_size)
    s_name = title.replace(" ", "") + PNG if save_name is None else save_name
    del heatmap_data  # free some memory
    f.savefig(os.path.join(save_path, s_name), dpi=dpi, format='png')
    return


def heatmap(heatmap_df, plot_title, plot_x_label, plot_y_label, min_val, max_val, title_size=10,
            target_focus_size=None, with_pic=None,
            plot_axes_lines=True, ax=None):
    """
    Plots a single heatmap sub plot, for vars see "heatmaps" method.
    """
    plot_title = plot_title.replace("_", " ")

    if with_pic is not None:
        g_ax = sns.heatmap(heatmap_df, cmap="YlGnBu_r", alpha=0.8, ax=ax, cbar=True, cbar_kws=dict(ticks=np.arange(min_val, max_val+0.01, step=0.01).tolist()),
                           vmin=min_val, vmax=max_val, zorder=2, mask=heatmap_df==0)
    else:
        g_ax = sns.heatmap(heatmap_df, cmap="YlGnBu_r", alpha=1, ax=ax, cbar=True, cbar_kws=dict(ticks=np.arange(min_val, max_val+0.01, step=0.01).tolist()),
                           vmin=min_val, vmax=max_val, mask=heatmap_df==0)

    g_ax.set_title(plot_title, fontdict={'size': title_size, 'weight': 'bold'})
    g_ax.set_ylabel(plot_y_label, fontdict={'size': title_size-2, 'weight': 'normal'})
    g_ax.set_xlabel(plot_x_label, fontdict={'size': title_size-2, 'weight': 'normal'})

    if with_pic is not None:
        arr_img = plt.imread(with_pic, format='png')
        g_ax.imshow(arr_img, aspect=g_ax.get_aspect(), extent=g_ax.get_xlim() + g_ax.get_ylim(), zorder=1, alpha=0.9)

    if plot_axes_lines:
        # draw center lines
        g_ax.axhline(y=heatmap_df.shape[0] / 2, color='w', linewidth=0.3, alpha=0.5)
        g_ax.axvline(x=heatmap_df.shape[1] / 2, color='w', linewidth=0.3, alpha=0.5)

    if target_focus_size is not None:
        # draw roi rectangle
        fix_marker = plt.Circle((heatmap_df.shape[1] / 2, heatmap_df.shape[0] / 2), radius=target_focus_size/2, color='g', fill=False)
        g_ax.add_patch(fix_marker)

    g_ax.set_yticks([])
    g_ax.set_xticks([])

    return plt


def polar(r_data, title, r_range=[0, 0.4, 0.1], colors=None, save_name="", save_path=""):
    """
    Plot a polar DISTRIBUTION plot (radians). Receive either data as a LIST (representing the AMOUNT of samples within
    each bin) - we deduce the bins (thetas) from the length of this list (=number of bins).
    :param r_data: If list, then a single plot of distribution and r_data is a list in the length equal to the number
    of bins the data was divided to, with the values representing amount of information samples falling within this bin.
    If this is a dictionary, then this is a dictionary of such lists, with each key in the dictionary representing
    different condition of the data. They will all be plotted together on the same plot.
    :return:
    """
    # start by making sure we have enough memory
    gc.collect()

    polar_name = "polar"
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')

    if isinstance(r_data, dict):  # these are sub_conds we want to plot within the same plot
        if not colors:
            colors = sns.color_palette("colorblind", len(list(r_data.keys())))
        i = 0
        for sub_cond in r_data:
            sub_cond_name = sub_cond.capitalize().replace("_" , " ")
            r = r_data[sub_cond]
            bins = 360 // len(r)
            theta = [(t + bins/2) * (math.pi / 180) for t in range(-180, 180, bins)]  # in radians, the MIDDLE of the bin
            ax.plot(theta, r, c=colors[i], label=sub_cond_name)
            i += 1

    else:  # data is a list of rads
        if not colors:
            colors = sns.color_palette("colorblind", 1)
        r = r_data
        bins = 360 // len(r)
        theta = [(t + bins / 2) * (math.pi / 180) for t in range(-180, 180, bins)]  # in radians, the MIDDLE of the bin
        ax.plot(theta, r, c=colors[0], label="")

    ax.set_rmax(r_range[1])
    ax.set_rticks(np.arange(r_range[0], r_range[1], r_range[2]))
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plot_save_name = save_name.replace(": ", "").replace(" ", "_").replace(r"\(.*\)", "")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W, H)
    # free memory
    del theta
    del bins
    del r_data
    gc.collect()
    plt.savefig(os.path.join(save_path, f"{polar_name}_{plot_save_name}.png"),  dpi=DPI, bbox_inches="tight")

    # free memory
    del figure
    gc.collect()
    return
