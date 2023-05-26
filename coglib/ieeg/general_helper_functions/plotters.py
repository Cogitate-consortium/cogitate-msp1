""" This script contains various functions performing some plotting
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import matplotlib as mpl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import mne
from mne_bids import BIDSPath
from general_helper_functions.pathHelperFunctions import find_files
import matplotlib


fig_size = [15, 20]
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
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def shifted_color_map(cmap, midpoint=1.0, vmax=10, name='shiftedcmap', start=0):
    '''
    KUDOS to: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
      start: "start point of the cmap"
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(start, vmax, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def sort_epochs(trials_metadata, sort_conditions, order=None):
    """
    This function sorts the trials metadata table according to the passed sort_conditions columns in order to plot
    the single trials raster plots ordered accordingly. It further computes additional info about position on the y axis
    of the different conditions. So for example, if you want to sort your trials according to category and duration
    (or any other parmameters you can think off), this function will return the order of your trials to match this
    sorting and it will further return the index of the rows containing the category 1, category 2... So that you can
    then plot horizontal lines on your graph to mark the transition from one cond to the other. It will also return the
    location on the y axis of where to set your ticks and labels to identify them in the plot.
    :param trials_metadata: (pandas data frame) contain one row per trial and one column per experimental conditions.
    :param sort_conditions: (list) must match the name of teh meta data columns. Corresponds to the column by
    which to order your trials. The order matters! If you pass ["category", "orientation"], it will order things like
    so:
    category | orientation
    Face     |  left
    Face     |  left
    Face     |  right
    Face     |  right
    ...
    :param order: (dict of list of strings) order by which to sort the condition. If you want to sort by category, but
    with a specific order of categories, you can pass ["Face", "Object", "letter", "false"] to have the epochs sorted
    like that. Otherwise, it will be sorted alphabetically for the one condition you give
    :return:
    order: (list) index of epochs sorted accordingly
    hline_pos: (dict) for each experimental condition, contains the y axis position of the transition from one level to
    another (so if you have condition category with levels: Faces, objects, this is the row index where you go from
    face to obj)
    cond_y_ticks_pos: (dict) middle position between the different hlines to set your labels ticks
    cond_y_ticks_labels: (dict) strings of the labels
    """
    ordered_epochs = trials_metadata.copy().reset_index(
        drop=True).sort_values(sort_conditions)[sort_conditions]
    if order is not None:
        ordered_epochs_new = pd.DataFrame()
        for cond in order.keys():
            for lvl in order[cond]:
                ordered_epochs_new = ordered_epochs_new.append(
                    ordered_epochs.loc[ordered_epochs[cond] == lvl]
                )
        ordered_epochs = ordered_epochs_new

    # Now, getting the indices of transitions between each conditions:
    numerical_descs = pd.DataFrame({
        condition: ordered_epochs[condition].astype('category').cat.codes for condition in ordered_epochs.columns
    }).reset_index(drop=True)
    # Computing the diff:
    diff_df = numerical_descs.diff()
    # Find indices in each column of what is not a zero:
    hline_pos = {
        condition: diff_df.index[(pd.notna(diff_df[condition])) & (
                diff_df[condition] != 0)].to_list()
        for condition in diff_df.columns
    }
    # Now find the mid-points of each transitions (i.e. where we will put the labels)
    cond_y_ticks_pos = {condition: [] for condition in hline_pos.keys()}
    for condition in hline_pos.keys():
        for n, idx in enumerate(hline_pos[condition]):
            if n == 0:
                cond_y_ticks_pos[condition].append(int(idx / 2))
            else:
                cond_y_ticks_pos[condition].append(int(((idx - hline_pos[condition][n - 1]) / 2)
                                                       + hline_pos[condition][n - 1]))
        # Adding the last transition midpoint:
        cond_y_ticks_pos[condition].append(((len(ordered_epochs) - hline_pos[condition][-1]) / 2)
                                           + hline_pos[condition][-1])

    # Now get the unique trials descriptions of the conditions we sorted by:
    cond_y_ticks_labels = {
        condition: list(ordered_epochs[condition].unique()) * (
            int(len(cond_y_ticks_pos[condition]) / len(ordered_epochs[condition].unique())))
        for condition in ordered_epochs.columns
    }

    return ordered_epochs.index, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels


def plot_vlines(v_lines, ax):
    """
    This function plots vertical lines according to the passed coordinates, spanning the whole plot vertically.
    :param v_lines: (list or int or float) x coordinates of the vertical lines to plot
    :param ax: (matplotlib axis object) on which to plot
    :return:
    """
    # Adding back the t0 vertical line:
    if isinstance(v_lines, list) or isinstance(v_lines, float) or isinstance(v_lines, int):
        ax.vlines(v_lines, ax.get_ylim()[0], ax.get_ylim()[
            1], linestyles='dashed', linewidth=0.5)
    elif v_lines is None:
        ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[
            1], linestyles='dashed', linewidth=1, colors="black")
    else:
        print("Warning: The vlines argument was of an unsuported format and therefore won't be used!")

    return ax


def plot_patches(x_patch_coords, ax, patch_colors="r"):
    """
    This function plots vertical patches according to the x patch coordinates passed as an argument
    :param x_patch_coords: (list of int or float pairs OR list of list of pairs) contains pairs of x coordinates for
    each patch to plot
    :param ax: (matplotlib ax object) axis on which to plot the patch
    :param patch_colors:  (string or rgb triplets) color of the patches. See matplotlib doc for more info
    :return: None
    """
    if isinstance(x_patch_coords, list):
        if all(isinstance(el, list) for el in x_patch_coords):
            for patch_coords in x_patch_coords:
                ax.axvspan(patch_coords[0], patch_coords[1],
                           color=patch_colors, alpha=0.1)
        elif all(isinstance(el, float) for el in x_patch_coords) or all(isinstance(el, int) for el in x_patch_coords):
            ax.axvspan(x_patch_coords[0], x_patch_coords[1],
                       color=patch_colors, alpha=0.1)
        else:
            raise Exception(
                "You have passed unsupported list format as x_patch_coords!")
    elif x_patch_coords is None:
        return
    else:
        raise Exception(
            "You have passed unsupported format as x_patch_coords argument! Must be either a list or None!")


def plot_epochs_grids(epochs, plot_type="raster", within_condition="All", sub_conditions=None, file_prefix="",
                      save=True, signal="raw-LFP", units=None, subject="",
                      plot_standard_error=True, grid_dim=(8, 8)):
    """
    This function plots several electrodes onto the same figure in a grid like fashion, enabling to see the data in one
    go. There are two options: either raster or waveform (aka evoked). When the option raster is selected, this function
    generates a grid of raster plots of the trials, one raster per channel. The option sub-conditions in that case
    sorts the trials by the conditions passed in the list (note that the strings in the sub_conditions list must match
    the meta data column!). If the option waveform (aka evoked) is passed, the data are averaged across trials and
    plotted with standard error around the mean. In that case, the sub_conditions strings will control within which
    condition levels to average. So say you pass ["category", "duration], the function will average the data first
    separately in the different categories, and then separately in the different duration, leading to two groups of
    grids, one per conditions. If None, then the data will be average across all passed trials.
    The other parameters are more cosmetic (title, saving...)
    :param epochs: (mne epochs object) epochs to plot
    :param plot_type: (string) which type of plots to generate. Two options are available: raster and waveform
    (or evoked)
    :param within_condition: (string) in case the passed epochs correspond to only a specific condition level, this will
    be added to the title and file name. Say you have two experimental contexts: task relevant and irrelevant and you
    have in this case passed only the task irrelevant trials, this information will be added to the title
    :param sub_conditions: (list of strings or None) list of experimental conditions to order the trials by (in case
    raster) or to average within (in case waveform). The string must match column names in the metadata table.
    Say you pass: ["category", "duration"], the trials in the raster will be ordered accordingly. In case of waveform,
    two grids will be plotted, one per condition. For each, the data will be averaged in each condition level
    :param file_prefix: (str) prefix to file name to save including path
    :param save: (boolean) whether or not you want to save
    :param signal: (string) name of the signal under investigation
    :param units: (dict of strings) units of the different channels types
    :param subject: (string) name of the subject under investigation
    :param plot_standard_error: (boolean) whether or not to plot the standard deviation
    :param grid_dim: (tuple) dimension of the plotted grid, i.e. rows and columns of axis of the figure
    :return: fig (list) list of figures plotted
    """
    # Set the default fig sizes:
    fig_sizes = (25.6, 12.55)
    plt.rcParams.update({'xtick.labelsize': 'xx-small'})
    plt.rcParams.update({'ytick.labelsize': 'xx-small'})
    plt.rcParams.update({'figure.titlesize': 14})
    # Set the units:
    if units is None:
        units = dict(seeg='µV', ecog='µV')
    if signal == "high_gamma" or signal == "alpha" or signal == "beta" or signal == "theta":
        units = dict(seeg='gain', ecog='gain')
        raster_scal = dict(seeg=1e0, ecog=1e0)
        evoked_scal = dict(seeg=1e3, ecog=1e3)
        ylim = [0, 3]
    else:
        raster_scal = dict(seeg=1e6, ecog=1e6)
        evoked_scal = dict(seeg=1e0, ecog=1e0)
        ylim = [-250, 250]
    # This function requires the epochs metadata:
    if epochs.metadata is None:
        raise Exception(
            "You have tried to plot ordered epochs, but you haven't generated the epochs metadata!")
    # Compute number of electrodes that can fit on one figure:
    n_elec = grid_dim[0] * grid_dim[1]
    # Compute how many plots we will need to plot all the electrodes:
    n_fig = math.ceil(len(epochs.ch_names) / n_elec)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot rasters:
    if plot_type.lower() == "raster":
        # Ordering the data:
        if isinstance(sub_conditions, list):
            order, _, _, _ = sort_epochs(epochs.metadata, sub_conditions)
        elif sub_conditions is None:
            order = None
        else:
            raise Exception("You have passed something that is not a list nor None as sub_conditions! This is not "
                            "supported!")
        title = "{0} sub-{1}, sig-{2}, N={3}".format(
            within_condition, subject, signal, len(epochs))
        fig = []
        # Setting up the counter to loop over each electrode:
        ctr = 0
        # Looping through the figures to generate:
        for fig_i in range(n_fig):
            # Open a figure in the said dimensions:
            fig_n, axs = plt.subplots(
                grid_dim[0], grid_dim[1], figsize=fig_sizes)
            fig.append(fig_n)
            plt.suptitle(title)
            # Now looping through each dimensions of the grid:
            for col_i in range(axs.shape[1]):
                for row_i in range(axs.shape[0]):
                    if ctr < len(epochs.ch_names):
                        # Plot the raster:
                        mne.viz.plot_epochs_image(epochs, picks=epochs.ch_names[ctr], order=order, show=False,
                                                  colorbar=False, units=units, scalings=raster_scal, vmin=ylim[0],
                                                  vmax=ylim[1], evoked=False, axes=axs[row_i, col_i])
                        # Removing the labels to avoid crowding:
                        if row_i != 0 or col_i != 0:
                            axs[row_i, col_i].xaxis.label.set_visible(False)
                            axs[row_i, col_i].xaxis.set_ticklabels([])
                            axs[row_i, col_i].set_xticks([])
                        axs[row_i, col_i].yaxis.label.set_visible(False)
                        axs[row_i, col_i].yaxis.set_ticklabels([])
                        axs[row_i, col_i].set_yticks([])
                        # Update the counter:
                        ctr = ctr + 1
                    else:
                        if ctr == len(epochs.ch_names):
                            # Get the axis x and y position in the figure:
                            # get the original position
                            pos1 = axs[row_i, col_i].get_position()
                            cax = plt.axes(
                                [pos1.x0, pos1.y0, 0.01, pos1.height])
                            plt.colorbar(axs[0, 0].images[0], cax=cax)
                            ctr = ctr + 1

                        # Remove the axis:
                        axs[row_i, col_i].axis("off")
                        continue
            # Set tight layout
            plt.tight_layout()
            # We can now save that plot:
            if save:
                file_name = "{0}_{1}grid_{2}.png".format(
                    file_prefix, plot_type, str(fig_i))
                plt.savefig(os.path.join(file_name), transparent=True)
                plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Plot waveforms:
    elif plot_type.lower() == "evoked" or plot_type.lower() == "waveform":
        # There is the option to either average across all trials or within a specific condition:
        if sub_conditions is None:
            # Generating the title:
            title = "{0} sub-{1}, sig-{2}, N={3}".format(
                within_condition, subject, signal, len(epochs))
            fig = []
            # If the plotting is done across all trials, then there is only one group of grids
            evoked = {"all": epochs.average()}
            # Scaling the data accordingly:
            evoked["all"].data = evoked["all"].data * (1 / evoked_scal["seeg"])
            # Computing the standard error:
            ste = epochs.standard_error()
            up_ste = {"all": mne.EvokedArray(evoked["all"].data + ste.data,
                                             evoked["all"].info,
                                             tmin=evoked["all"].times[0])}
            low_ste = {"all": mne.EvokedArray(evoked["all"].data - ste.data,
                                              evoked["all"].info,
                                              tmin=evoked["all"].times[0])}
            # Setting up the counter:
            ctr = 0
            # Plotting all the electrodes on the grids:
            for fig_i in range(n_fig):
                # Open a figure in the said dimensions:
                fig_n, axs = plt.subplots(
                    grid_dim[0], grid_dim[1], figsize=fig_sizes)
                fig.append(fig_n)
                plt.suptitle(title)
                # Now looping through each dimensions of the grid:
                for col_i in range(axs.shape[1]):
                    for row_i in range(axs.shape[0]):
                        if ctr < len(epochs.ch_names):
                            # Setting the figure title:
                            mne.viz.plot_compare_evokeds(evoked, picks=epochs.ch_names[ctr], legend=False,
                                                         axes=axs[row_i, col_i], show=False, show_sensors=False,
                                                         invert_y=False)
                            if plot_standard_error:
                                # Generating the line styles for the ste:
                                line_styles = {"all": 'dashdot'}
                                styles = {"all": {"linewidth": 0.5}}
                                mne.viz.plot_compare_evokeds(up_ste, picks=epochs.ch_names[ctr], legend=False,
                                                             axes=axs[row_i,
                                                                      col_i], show=False,
                                                             show_sensors=False, linestyles=line_styles,
                                                             styles=styles, invert_y=False)
                                mne.viz.plot_compare_evokeds(low_ste, picks=epochs.ch_names[ctr], legend=False,
                                                             axes=axs[row_i,
                                                                      col_i], show=False,
                                                             show_sensors=False, linestyles=line_styles,
                                                             styles=styles, invert_y=False)
                                # Removing the labels to avoid crowding:
                                if row_i != 0 or col_i != 0:
                                    axs[row_i, col_i].xaxis.label.set_visible(
                                        False)
                                    axs[row_i, col_i].yaxis.label.set_visible(
                                        False)
                                    axs[row_i, col_i].xaxis.set_ticklabels([])
                                    axs[row_i, col_i].yaxis.set_ticklabels([])
                                    axs[row_i, col_i].set_xticks([])
                                    axs[row_i, col_i].set_yticks([])
                                # Update the counter:
                                ctr = ctr + 1
                        else:
                            # Remove the axis:
                            axs[row_i, col_i].axis("off")
                            continue
                # Uniformize limits across subplots:
                ylims = [ax.get_ylim() for ax in axs.flatten()]
                ylims = [item for sublist in ylims for item in sublist]
                plt.setp(axs, ylim=[max(ylims), min(ylims)])
                # Set tight layout
                plt.tight_layout()
                # We can now save that plot:
                if save:
                    file_name = "{0}_{1}grid_{2}_{3}.png".format(
                        file_prefix, plot_type, within_condition, str(fig_i))
                    plt.savefig(os.path.join(file_name), transparent=True)
                    plt.close()
        # If the waveform should be averaged within specific categories levels:
        elif isinstance(sub_conditions, list):
            # There will then be as many grids groups as there are sub_conditions:
            fig = []
            for n, cond in enumerate(sub_conditions):
                # Generating the title:
                title = "{0} trials, cond-{1}, sub-{2}, sig-{3}, N={4}".format(within_condition, cond, subject, signal,
                                                                               len(epochs))
                # Plotting the evoked response for each levels of this condition:
                evokeds = dict()
                up_ste = dict()
                low_ste = dict()
                query = cond + ' == "{}"'
                # Setting up the counter:
                ctr = 0
                for cond_lvl in epochs.metadata[cond].unique():
                    evokeds[str(cond_lvl)] = epochs[query.format(
                        cond_lvl)].average()
                    # Scaling the data accordingly:
                    evokeds[str(cond_lvl)].data = evokeds[str(
                        cond_lvl)].data * (1 / evoked_scal["seeg"])
                    ste = epochs[query.format(cond_lvl)].standard_error()
                    up_ste[str(cond_lvl)] = mne.EvokedArray(evokeds[str(cond_lvl)].data + ste.data,
                                                            evokeds[str(
                                                                cond_lvl)].info,
                                                            tmin=evokeds[str(cond_lvl)].times[0])
                    low_ste[str(cond_lvl)] = mne.EvokedArray(evokeds[str(cond_lvl)].data - ste.data,
                                                             evokeds[str(
                                                                 cond_lvl)].info,
                                                             tmin=evokeds[str(cond_lvl)].times[0])
                # Plotting the data separately for each condition:
                for fig_i in range(n_fig):
                    # Open a figure in the said dimensions:
                    fig_n, axs = plt.subplots(
                        grid_dim[0], grid_dim[1], figsize=fig_sizes)
                    fig.append(fig_n)
                    plt.suptitle(title)
                    # Now looping through each dimensions of the grid:
                    for col_i in range(axs.shape[1]):
                        for row_i in range(axs.shape[0]):
                            if ctr < len(epochs.ch_names):
                                # Setting the figure title:
                                mne.viz.plot_compare_evokeds(evokeds, picks=epochs.ch_names[ctr],
                                                             axes=axs[row_i,
                                                                      col_i], legend=False,
                                                             show=False, show_sensors=False, invert_y=False)
                                if plot_standard_error:
                                    # Generating the line styles for the ste:
                                    line_styles = {
                                        key: 'dashdot' for key in up_ste.keys()}
                                    styles = {key: {"linewidth": 0.5}
                                              for key in up_ste.keys()}
                                    mne.viz.plot_compare_evokeds(up_ste, picks=epochs.ch_names[ctr], legend=False,
                                                                 axes=axs[row_i,
                                                                          col_i], show=False,
                                                                 show_sensors=False, linestyles=line_styles,
                                                                 styles=styles, invert_y=False)
                                    mne.viz.plot_compare_evokeds(low_ste, picks=epochs.ch_names[ctr], legend=False,
                                                                 axes=axs[row_i,
                                                                          col_i], show=False,
                                                                 show_sensors=False, linestyles=line_styles,
                                                                 styles=styles, invert_y=False)
                                    # Removing the labels to avoid crowding:
                                    if row_i != 0 or col_i != 0:
                                        axs[row_i, col_i].xaxis.label.set_visible(
                                            False)
                                        axs[row_i, col_i].yaxis.label.set_visible(
                                            False)
                                        axs[row_i, col_i].xaxis.set_ticklabels(
                                            [])
                                        axs[row_i, col_i].yaxis.set_ticklabels(
                                            [])
                                        axs[row_i, col_i].set_xticks([])
                                        axs[row_i, col_i].set_yticks([])
                                    # Update the counter:
                                    ctr = ctr + 1
                            else:
                                # If we are at the first axs after the last channel, adding the legend:
                                if ctr == len(epochs.ch_names):
                                    mne.viz.plot_compare_evokeds(evokeds, picks=epochs.ch_names[ctr - 1], legend=True,
                                                                 axes=axs[row_i, col_i], show=False, show_sensors=False,
                                                                 invert_y=False)
                                    # Get the legend handle:
                                    leg = axs[row_i,
                                              col_i].get_legend_handles_labels()
                                    # Clearing the axis:
                                    axs[row_i, col_i].clear()
                                    # Adding only the legend:
                                    axs[row_i, col_i].legend(
                                        *leg, loc='center')
                                    ctr = ctr + 1
                                # Remove the axis:
                                axs[row_i, col_i].axis("off")
                                continue
                    # Uniformize limits across subplots:
                    ylims = [ax.get_ylim() for ax in axs.flatten()]
                    ylims = [item for sublist in ylims for item in sublist]
                    plt.setp(axs, ylim=[max(ylims), min(ylims)])
                    # Set tight layout
                    plt.tight_layout()
                    # We can now save that plot:
                    if save:
                        file_name = "{0}_{1}grid_{2}_{3}_{4}.png".format(file_prefix, plot_type, within_condition,
                                                                         cond, str(fig_i))
                        plt.savefig(os.path.join(file_name), transparent=True)
                        plt.close()
        else:
            raise Exception("The passed subconditions argument was not of the correct format. It must be either a "
                            "list or None!")
    else:
        raise Exception("You have passed an unsupported plot_type argument to the plot_epochs_grids function! Must be "
                        "raster OR waveform OR evoked!")
    return fig


def plot_ordered_epochs(epochs, channel, within_condition="All", sort_conditions=None, file_prefix="", save=True,
                        signal="raw-LFP", units=None, subject="", v_lines=None, patch_x_coords=None,
                        plot_evoked=True, plot_standard_error=True, raster_scal=None, evoked_scal=None, ylim=None,
                        title_suffix=""):
    """
    This functions plots heatmaps (i.e. raster plots) of single trials as well as single trials time series below that.
    If a list of conditions is passed in sort conditions, it will sort the trials according to the meta data column
    passed as sort_conditions. Note that the strings in the sort conditions list must match the metadata table column
    names. The order in this list dictates the sorting order. Furthermore, if plot_evoked is set to true, a separated
    plots will be generated to plot separately the evoked response to each specific condition passed in sort condition,
    with one evoked response per level of that specific condition.
    :param epochs: (mne epochs object) contains the data to plot. WARNING: MUST HAVE THE METADATA TABLE IF YOU WANT
    TO USE THE SORTING OPTIONS!
    :param channel: (string or int) name or number of the channel to plot
    :param within_condition: (string) if you are plotting only a subset of trials from your experiment, the condition
    of the subset can be passed to adjust the title of the plots
    :param sort_conditions: (list of str) conditions (i.e. metadata column names) you want to sort the data by.
    :param file_prefix: (str) prefix to file name to save including path
    :param save: (boolean) whether or not you want to save
    :param signal: (string) name of the signal under investigation
    :param units: (dict of strings) units of the different channels types
    :param subject: (string) name of the subject under investigation
    :param v_lines: (list of floats or int) coordinates of vlines to plot to mark specific things of interest
    :param patch_x_coords: (list of lists or tuples) pairs of coordinates along the x axis to plot patches overlaid on
    pic
    :param plot_evoked: (bool) whether or not to plot the evoked responses to the different conditions in a separate
    plot
    :param plot_standard_error: (bool) whether or not to plot the standard error around the mean in evoked responses
    :return: None
    """
    # Set the default fig sizes:
    fig_sizes = (18.5, 12)
    plt.rcParams.update({'font.size': 24})
    # Set the units:
    if signal == "high_gamma" or signal == "alpha" or signal == "beta" or signal == "theta":
        if units is None:
            units = dict(seeg='gain', ecog='gain')
        if raster_scal is None:
            raster_scal = dict(seeg=1e0, ecog=1e0)
        if evoked_scal is None:
            evoked_scal = dict(seeg=1e0, ecog=1e0)
        if ylim is None:
            ylim = [None, None]
    else:
        if units is None:
            units = dict(seeg='µV', ecog='µV')
        if raster_scal is None:
            raster_scal = dict(seeg=1e6, ecog=1e6)
        if evoked_scal is None:
            evoked_scal = dict(seeg=1e0, ecog=1e0)
        if ylim is None:
            ylim = [None, None]
    # This function requires the epochs metadata:
    if epochs.metadata is None:
        raise Exception(
            "You have tried to plot ordered epochs, but you haven't generated the epochs metadata!")
    # ------------------------------------------------------------------------------------------------------------------
    # Heat map figure:
    # Get the order of the epochs to plots:
    order, hline_pos, cond_y_ticks_pos, cond_y_ticks_labels = sort_epochs(
        epochs.metadata, sort_conditions)
    # Setting the figure title:
    title = \
        "{0} trials raster \n sub-{1}, sig-{2}, ch-{3}, N={4},\n{5}".format(within_condition, subject, signal, channel,
                                                                            len(epochs), title_suffix)
    print("=" * 40)
    print("Plotting channel {0}".format(channel))
    # Get the channel type:
    ch_type = epochs.get_channel_types(picks=channel)[0]
    # We can now plot the data with the mne_plot_image function:
    fig = mne.viz.plot_epochs_image(epochs, picks=channel, order=order, show=False,
                                    units=units, scalings=raster_scal, vmin=ylim[0], vmax=ylim[1], title=title,
                                    ts_args={"show_sensors": False, "ci": False})
    axs = fig[0].axes
    # Now adding lines separating the different conditions:
    [axs[0].hlines(hline_pos[cond], epochs.times[0], epochs.times[-1],
                   color="k", linewidth=1, linestyle=":") for cond in sort_conditions[0:2]]
    # Adding labels to the heat map for the two first sort conditions:
    for n in range(len(sort_conditions)):
        if n > 1:
            continue
        y_labels = list(cond_y_ticks_labels[sort_conditions[n]])
        labels_loc = cond_y_ticks_pos[sort_conditions[n]]
        if n == 0:
            # Add ticks and labels:
            axs[0].set_yticks(labels_loc)
            axs[0].set_yticklabels(
                y_labels * int((len(labels_loc) / len(y_labels))))
        else:
            # Add ticks and labels:
            ax2 = axs[0].twinx()
            # Adding an extra label, otherwise things get weirdly mushed together:
            y_labels.append(" ")
            # Only keeping the first letter:
            y_labels = [label[0] for label in y_labels]
            labels_loc.append(axs[0].get_ylim()[-1])
            ax2.set_yticks(labels_loc)
            ax2.set_yticklabels(
                y_labels * int((len(labels_loc) / len(y_labels))))
            # Setting the limits of the axis, otherwise it does weird stuff:
            ax2.set_xlim(axs[0].get_ylim())
    # Adding v lines:
    plot_vlines(v_lines, axs[0])

    # -------------------------
    # Plot single trials instead of average:
    # Removing the plotted line
    axs[1].lines[0].remove()
    axs[1].collections[0].remove()
    # Plotting single trials
    axs[1].plot(epochs.times,
                np.squeeze(epochs.get_data(picks=channel)).T * raster_scal[ch_type], linewidth=0.3, alpha=0.5,
                color="grey")
    # Plot the average time series on top
    axs[1].plot(epochs.times, np.mean(np.squeeze(epochs.get_data(picks=channel)) * raster_scal[ch_type], axis=0),
                linewidth=1,
                color="black")
    # Reset the limit to fit the newly plotted data:
    axs[1].relim()  # make sure all the data fits
    axs[1].autoscale()  # auto-scale
    # Set the ylabel
    axs[1].set_ylabel(units[ch_type])
    # Adding the v_lines:
    plot_vlines(v_lines, axs[1])
    # Adding patches:
    plot_patches(patch_x_coords, axs[1], patch_colors="r")
    # Increasing the figure size:
    fig[0].set_size_inches(*fig_sizes)
    # Setting to tight layout:
    plt.tight_layout()
    # Saving the figure it necessary:
    if save:
        file_name = "{0}_epochs_raster.png".format(file_prefix)
        plt.savefig(os.path.join(file_name), transparent=True)
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------
    # Plot evoked arrays for each evoked conditions if required:
    if plot_evoked:
        if sort_conditions is None:
            raise Exception(
                "You have asked to plot the evoked responses without specifying the ")
        # Preparing an additional figure to plot the evoked data:
        # Depending on the number of separate conditions to plot adjusting the shape of the plot:
        if len(sort_conditions) < 3:
            evoked_fig, evoked_axs = plt.subplots(
                len(sort_conditions), 1, figsize=fig_sizes)
        else:
            evoked_fig, evoked_axs = plt.subplots(
                int(math.ceil(len(sort_conditions) / 2)), 2, figsize=fig_sizes)
        # Adding the super title:
        title = "{0} sub-{1}, sig-{2}, ch-{3}, N={4},\n{5}".format(
            within_condition, subject, signal, channel, len(epochs), title_suffix)
        plt.suptitle(title)
        for n, cond in enumerate(sort_conditions):
            if len(sort_conditions) > 1 and len(evoked_axs.shape) > 1:
                if n % 2 != 0:
                    ax = evoked_axs[int(math.floor(n / 2)), 1]
                else:
                    ax = evoked_axs[int(math.floor(n / 2)), 0]
            elif len(sort_conditions) > 1:
                ax = evoked_axs[n]
            else:
                ax = evoked_axs
            # Plotting the evoked response for each levels of this condition:
            evokeds = dict()
            up_ste = dict()
            low_ste = dict()
            query = cond + ' == "{}"'
            for cond_lvl in epochs.metadata[cond].unique():
                evokeds[str(cond_lvl)] = epochs[query.format(
                    cond_lvl)].average(picks=channel)
                # Scaling the data accordingly:
                evokeds[str(cond_lvl)].data = evokeds[str(
                    cond_lvl)].data * (1 / evoked_scal[ch_type])
                ste = epochs[query.format(cond_lvl)].standard_error(
                    picks=channel)
                ste.data = ste.data * (1 / evoked_scal[ch_type])
                up_ste[str(cond_lvl)] = mne.EvokedArray(evokeds[str(cond_lvl)].data + ste.data,
                                                        evokeds[str(
                                                            cond_lvl)].info,
                                                        tmin=evokeds[str(cond_lvl)].times[0])
                low_ste[str(cond_lvl)] = mne.EvokedArray(evokeds[str(cond_lvl)].data - ste.data,
                                                         evokeds[str(
                                                             cond_lvl)].info,
                                                         tmin=evokeds[str(cond_lvl)].times[0])
            # Setting the figure title:
            mne.viz.plot_compare_evokeds(evokeds, picks=channel, axes=ax, show=False,
                                         show_sensors=False, title=cond, invert_y=False)
            if plot_standard_error:
                # Generating the line styles for the ste:
                line_styles = {key: 'dashdot' for key in up_ste.keys()}
                styles = {key: {"linewidth": 0.5} for key in up_ste.keys()}
                mne.viz.plot_compare_evokeds(up_ste, picks=channel, axes=ax, show=False,
                                             show_sensors=False, title=cond, linestyles=line_styles,
                                             styles=styles, invert_y=False)
                mne.viz.plot_compare_evokeds(low_ste, picks=channel, axes=ax, show=False,
                                             show_sensors=False, title=cond, linestyles=line_styles,
                                             styles=styles, invert_y=False)
            # Adding the units:
            ax.set_ylabel(units[ch_type])
            # Adding the v_lines:
            plot_vlines(v_lines, ax)
            # Adding the patches:
            plot_patches(patch_x_coords, ax, patch_colors="r")
        # Uniformize limits across subplots:
        if len(sort_conditions) > 1:
            ylims = [ax.get_ylim() for ax in evoked_axs.flatten()]
            ylims = [item for sublist in ylims for item in sublist]
            plt.setp(evoked_axs, ylim=[max(ylims), min(ylims)])
        plt.tight_layout()

        # Saving if necessary:
        if save:
            file_name = "{0}_evoked.png".format(file_prefix)
            plt.savefig(os.path.join(file_name), transparent=True)
        plt.close()
    return None


def plot_single_elec_on_brain(elec, subject, save_path, coord_space="T1",
                              bids_root="/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids",
                              session="V1", task="Dur",
                              views=None):
    """
    This function enables plotting one single electrode on the brain. This is useful to show single electrodes in some
    analyses
    :param elec: (string) name of the electrode to plot
    :param subject: (string) name of the subject
    :param save_path: (string) path to save the data
    :param coord_space: (string) T1 or MNI
    :param bids_root: (path) path to the bids root
    :param session: (string) name of the session of interest
    :param task: (string) name of the task of interest
    :param views: (list of strings) list of the different views
    :return: None, save figure to file
    """

    # Generate the bids path
    if views is None:
        views = ["lateral", "medial", "rostral", "caudal", "frontal", "parietal"]
    bids_path = BIDSPath(root=bids_root, subject=subject,
                         session=session,
                         datatype="ieeg",
                         task=task)
    # Get the info about the channels:
    channel_info_file = find_files(bids_path.directory, naming_pattern="*channels", extension=".tsv")
    # Get the channels coordinates:
    if coord_space.lower() == "t1":
        coords_csv = find_files(bids_path.directory, naming_pattern="*space-Other_electrodes", extension=".tsv")
    elif coord_space.lower() == "mni":
        coords_csv = find_files(bids_path.directory, naming_pattern="*space-fsaverage_electrodes", extension=".tsv")
    # Load the files:
    channels_info = pd.read_csv(channel_info_file[0], sep='\t')
    channels_coords = pd.read_csv(coords_csv[0], sep='\t')

    # Get the info for the channel of interest:
    channel_info = channels_info.loc[channels_info["name"] == elec]
    channel_coords = channels_coords.loc[channels_coords["name"] == elec]

    # Creating the appropriate montage:
    ch_pos = {elec: np.squeeze(np.array([channel_coords["x"], channel_coords["y"], channel_coords["z"]]))}
    if coord_space.lower() == "t1":
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mri")
        subject_id = "sub-" + subject
        subjects_dir = Path(bids_root, "derivatives", "fs")
        montage.add_estimated_fiducials(subject_id, subjects_dir)
        trans = mne.channels.compute_native_head_t(montage)
    elif coord_space.lower() == "mni":
        subject_id = "fsaverage"
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mni_tal")
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        montage.add_mni_fiducials(subjects_dir)
        trans = 'fsaverage'
    else:
        raise Exception("You have passed a montage space that doesn't exists")
    # Create the info
    info = mne.create_info(list(ch_pos.keys()), 100, ch_types="seeg")
    info.set_montage(montage)
    # Create the brain
    Brain = mne.viz.get_brain_class()
    if channel_info["type"].item() == "seeg":
        brain = Brain(subject_id, 'both', 'white', subjects_dir=subjects_dir, cortex='low_contrast',
                      background='white', size=(800, 600), alpha=0.7)
    else:
        brain = Brain(subject_id, 'both', 'pial', subjects_dir=subjects_dir, cortex='low_contrast',
                      background='white', size=(800, 600), alpha=0.7)
    # Plotting the sensors
    brain.add_sensors(info, trans=trans)
    # Now plotting all the different views:
    views = ["lateral", "medial", "rostral", "caudal", "frontal", "parietal"]
    for view in views:
        brain.show_view(view)
        fig, ax = plt.subplots(figsize=fig_size)
        img_mat = brain.screenshot()
        ax.imshow(img_mat)
        # Remove tick marks and labels:
        ax.set_axis_off()
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        file_name = Path(save_path, "sub-{}_ch-{}_{}.png".format(subject, elec, view))
        plt.savefig(file_name, transparent=True)
    mne.viz.close_all_3d_figures()
