import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config
from matplotlib import font_manager
# for plot_brain
from mne import read_surface
from mne.utils import get_subjects_dir, run_subprocess
from mne.label import _read_annot
from mne.surface import read_curvature, _mesh_borders
from mne.transforms import apply_trans
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib import cm, colors
from matplotlib.collections import PolyCollection

# import nilearn.image
# import nilearn.surface
from nilearn.image import load_img
from nilearn.surface import vol_to_surf
from nilearn.regions import Parcellations
from nilearn.plotting.surf_plotting import _get_faces_on_edge

# get the parameters dictionary
param = config.param

# Set Helvetica as the default font:
font_path = os.path.join(os.path.dirname(__file__), "Helvetica.ttf")
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

fig_size = param["figure_size_mm"]
def_cmap = param["colors"]["cmap"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = param["font"]
plt.rc('font', size=param["font_size"])  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"])  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"])  # legend fontsize
plt.rc('figure', titlesize=param["font_size"])  # fontsize of the fi


def mm2inch(val):
    return val / 25.4


def plot_matrix(data, x0, x_end, y0, y_end, mask=None, cmap=None, ax=None, ylim=None, midpoint=None, transparency=1.0,
                interpolation='lanczos',
                xlabel="Time (s)", ylabel="Time (s)", xticks=None, yticks=None, cbar_label="Accuracy", filename=None,
                vline=0,
                title=None, square_fig=False):
    """
    This function plots 2D matrices such as temporal generalization decoding or time frequency decompositions with or
    without significance. If a significance mask is passed, the significance pixels will be surrounded with significance
    line. The significance parts will be fully opaque but the non-significant patches transparency can be controlled
    by the transparency parameter
    :param data: (2D numpy array) data to plot
    :param x0: (float) first sample value for the x axis, for ex the first time point in the data to be able to
    create meaningful axes
    :param x_end: (float) final sample value for the x axis, for ex the last time point in the data to be able to create
    meaningful axes.
    :param y0: (float) first sample value for the y axis, for ex the first time point in the data to be able to
    create meaningful axes or the first frequency of a frequency decomposition...
    :param y_end: (float) final sample value for the y axis, for ex  the last time point in the data to be able to
    create meaningful axes or the first frequency of a frequency decomposition...
    :param mask: (2D numpy array of booleans) significance mask. MUST BE THE SAME SIZE as data. True where the data
    are significance, false elsewhere
    :param cmap: (string) name of the color map
    :param ax: (matplotlib ax object) ax on which to plot the data. If not passed, a new figure will be created
    :param ylim: (list of 2 floats) limits of the data for the plotting. If not passed, taking the 5 and 95 percentiles
    of the data
    :param midpoint: (float) midpoint of the data. Centers the color bar on this value.
    :param transparency: (float) transparency of the non-significant patches of the matrix
    :param xlabel: (string) xlabel fo the data
    :param ylabel: (string) ylabel fo the data
    :param xticks: (list of strings) xtick label names
    :param yticks: (list of strings) ytick label names
    :param cbar_label: (string) label of the color bar
    :param filename: (string or pathlib path object) name of the file to save the figures to. If not passed, nothing
    will be saved. Must be the full name with png extension. The script will take care of saving the data to svg as well
    and as csv
    :param vline: (float) coordinates of vertical and horizontal lines to plot
    :param title: (string) title of the figure
    :param square_fig: (boolean) whether or not to have the figure squared proportions. Useful for temporal
    generalization plots that are usually square!
    :return:
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[1])])
    if ylim is None:
        ylim = [np.percentile(data, 5), np.percentile(data, 95)]

    if midpoint is None:
        midpoint = np.mean([ylim[0], ylim[1]])

    try:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=midpoint, vmax=ylim[1])
    except ValueError:
        print("WARNING: The midpoint is outside the range defined by ylim[0] and ylim[1]! We will continue without"
              "normalization")
        norm = None

    if cmap is None:
        cmap = def_cmap
    if square_fig:
        aspect = "equal"
    else:
        aspect = "auto"
    # Plot matrix with transparency:
    im = ax.imshow(data, cmap=cmap, norm=norm,
                   extent=[x0, x_end, y0, y_end],
                   origin="lower", alpha=transparency, aspect=aspect, interpolation=interpolation)
    # Plot the significance mask on top:
    if mask is not None:
        sig_data = data
        sig_data[~mask] = np.nan
        if not np.isnan(mask).all():
            # Plot only the significant bits:
            ax.imshow(sig_data, cmap=cmap, origin='lower', norm=norm,
                      extent=[x0, x_end, y0, y_end],
                      aspect=aspect, interpolation=interpolation)
            ax.contour(mask > 0, mask > 0, colors="k", origin="lower",
                       extent=[x0, x_end, y0, y_end])

    # Add the axis labels and so on:
    ax.set_xlim([x0, x_end])
    ax.set_ylim([y0, y_end])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title)
    ax.axvline(vline, color='k')
    ax.axhline(vline, color='k')
    plt.tight_layout()
    cb = plt.colorbar(im)
    cb.ax.set_ylabel(cbar_label)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        if mask is not None:
            np.savetxt(filename + "_mask" + ".csv", mask, delimiter=",")

    return ax


def plot_pcolormesh(data, xs, ys, mask=None, cmap=None, ax=None, vlim=None, transparency=1.0,
                    xlabel="Time (s)", ylabel="Time (s)", cbar_label="Accuracy", filename=None, vline=0,
                    title=None, square_fig=False):
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[1])])

    if vlim is None:
        vlim = [np.percentile(data, 5), np.percentile(data, 95)]

    if cmap is None:
        cmap = def_cmap

    im = ax.pcolormesh(xs, ys, data,
                       cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       alpha=transparency, rasterized=True)

    if mask is not None:
        sig_data = data
        sig_data[~mask] = np.nan
        if not np.isnan(mask).all():
            ax.pcolormesh(xs, ys, sig_data,
                          cmap=cmap, vmin=vlim[0], vmax=vlim[1], rasterized=True)
            ax.contour(xs, ys, mask > 0, colors="k")

    # Add the axis labels and so on:
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([ys[0], ys[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.axvline(vline, color='k')
    ax.axhline(vline, color='k')
    plt.tight_layout()
    cb = plt.colorbar(im)
    cb.ax.set_ylabel(cbar_label)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        plt.savefig(filename + ".pdf", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        if mask is not None:
            np.savetxt(filename + "_mask" + ".csv", mask, delimiter=",")

    return ax


def plot_time_series(data, t0, tend, ax=None, err=None, colors=None, vlines=None, xlim=None, ylim=None,
                     xlabel="Time (s)", ylabel="Activation", err_transparency=0.2,
                     filename=None, title=None, square_fig=False, conditions=None, do_legend=True,
                     patches=None, patch_color="r", patch_transparency=0.2):
    """
    This function plots times series such as average of iEEG activation across trials and/or electrodes... If the error
    parameter is passed, the error will be plotted as shaded around the main line. Additionally, patches
    can be plotted over the data to represent significance or time windows of interest... Additionally, vertical lines
    can be plotted to delimitate relevant time points.
    :param data: (2D numpy array) contains time series to plot. The first dimension should be different conditiosn
    and the last dimension is time! The first dimension here should be ordered according to the other parameters,
    such as the conditions, errors...
    :param t0: (float) time 0, i.e. the first time point in the data to be able to create meaningful axes
    :param tend: (float) final time point, i.e. the last time point in the data to be able to create meaningful axes
    :param ax: (matplotlib ax object) ax on which to plot the data. If not passed, a new figure will be created
    :param err: (2D numpy array) contains errors of the time series to plot. The first dimension should be different
    conditions and the last dimension is time! The first dimension here should be ordered according to the other
    parameters, such as the data, conditions...
    :param colors: (list of string or RGB float triplets) colors of each condition. There should be as many as there
    are rows in the data
    :param vlines: (list of floats) x coordinates at which to draw the vertical lines
    :param xlim: (list of 2 floats) limits of the x axis if any
    :param ylim: (list of 2 floats) limits of the y axis if any
        :param xlabel: (string) xlabel fo the data
    :param ylabel: (string) ylabel fo the data
    :param filename: (string or pathlib path object) name of the file to save the figures to. If not passed, nothing
    will be saved. Must be the full name with png extension. The script will take care of saving the data to svg as well
    and as csv
    :param title: (string) title of the figure
    :param square_fig: (boolean) whether or not to have the figure squared proportions. Useful for temporal
    generalization plots that are usually square!
    :param err_transparency: (float) transparency of the errors around the mean
    :param conditions: (list of strings) name of each condition to be plotted, for legend
    :param do_legend: (boolean) whether or not to plot the legend
    :param patches: (list of 2 floats of list of list) x coordinates of the start and end of a patch
    :param patch_color: (string or RGB triplet) color of the patch
    :param patch_transparency: (float) transparency of the patch
    :return:
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[1])])
    if conditions is None:
        conditions = ["" for i in range(data.shape[0])]
    if colors is None:
        colors = [None for i in range(data.shape[0])]
    # Create the time axis:
    times = np.linspace(t0, tend, num=data.shape[1])
    # Plot matrix with transparency:
    for ind in range(data.shape[0]):
        ax.plot(times, data[ind], color=colors[ind],
                label=conditions[ind])
        # Plot the errors:
        if err is not None:
            ax.fill_between(times, data[ind] - err[ind], data[ind] + err[ind],
                            color=colors[ind], alpha=err_transparency)
    # Set the x limits:
    ax.set_xlim(times[0], times[-1])
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[-1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[-1])
    # Adding vlines:
    if vlines is not None:
        ax.vlines(vlines, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dashed', linewidth=1.5, colors='k')
    # Adding patches:
    if patches is not None:
        if not isinstance(patches[0], list):
            ax.axvspan(patches[0], patches[1], fc=patch_color, alpha=patch_transparency)
        else:
            for patch in patches:
                ax.axvspan(patch[0], patch[1], fc=patch_color, alpha=patch_transparency)

    # Add the labels title and so on:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if do_legend:
        ax.legend()
    plt.tight_layout()
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")
        np.savetxt(filename + "_error" + ".csv", err, delimiter=",")

    return ax


def plot_rasters(data, t0, tend, cmap=None, ax=None, ylim=None, midpoint=None, transparency=1.0,
                 xlabel="Time (s)", ylabel="Time (s)", cbar_label="Accuracy", filename=None, vlines=0,
                 title=None, square_fig=False, conditions=None, cond_order=None):
    """
    This function plots 2D matrices such as temporal generalization decoding or time frequency decompositions with or
    without significance. If a significance mask is passed, the significance pixels will be surrounded with significance
    line. The significance parts will be fully opaque but the non-significant patches transparency can be controlled
    by the transparency parameter
    :param data: (2D numpy array) data to plot
    :param t0: (float) time 0, i.e. the first time point in the data to be able to create meaningful axes
    :param tend: (float) final time point, i.e. the last time point in the data to be able to create meaningful axes
    :param cmap: (string) name of the color map
    :param ax: (matplotlib ax object) ax on which to plot the data. If not passed, a new figure will be created
    :param ylim: (list of 2 floats) limits of the data for the plotting. If not passed, taking the 5 and 95 percentiles
    of the data
    :param midpoint: (float) midpoint of the data. Centers the color bar on this value.
    :param transparency: (float) transparency of the non-significant patches of the matrix
    :param xlabel: (string) xlabel fo the data
    :param ylabel: (string) ylabel fo the data
    :param cbar_label: (string) label of the color bar
    :param filename: (string or pathlib path object) name of the file to save the figures to. If not passed, nothing
    will be saved. Must be the full name with png extension. The script will take care of saving the data to svg as well
    and as csv
    :param vlines: (float) coordinates of vertical and horizontal lines to plot
    :param title: (string) title of the figure
    :param square_fig: (boolean) whether or not to have the figure squared proportions. Useful for temporal
    generalization plots that are usually square!
    :param conditions: (list or iterable of some sort) condition of each trial to order them properly.
    :param cond_order: (list) order in which to sort the conditions. So say you are trying to plot faces,
    objects, letters and so on, and you want to enforce that in the plot the faces appear first, then the objects,
    then the letters, pass the list ["face", "object", "letter"]
    :return:
    """
    if ax is None:
        if square_fig:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[0])])
        else:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[1])])
    if ylim is None:
        ylim = [np.percentile(data, 5), np.percentile(data, 95)]
    if midpoint is not None:
        try:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=midpoint, vmax=ylim[1])
        except ValueError:
            print("WARNING: The midpoint is outside the range defined by ylim[0] and ylim[1]! We will continue without"
                  "normalization")
            norm = None
    else:
        norm = None
    if cmap is None:
        cmap = def_cmap
    if square_fig:
        aspect = "equal"
    else:
        aspect = "auto"
    # Sorting the epochs if not plotting:
    if conditions is not None:
        conditions = np.array(conditions)
        if cond_order is not None:
            inds = []
            for cond in cond_order:
                inds.append(np.where(conditions == cond)[0])
            inds = np.concatenate(inds)
        else:
            inds = np.argsort(conditions)
        data = data[inds, :]
    # Plot matrix with transparency:
    im = ax.imshow(data, cmap=cmap, norm=norm,
                   extent=[t0, tend, 0, data.shape[0]],
                   origin="lower", alpha=transparency, aspect=aspect)
    # Sort the conditions accordingly:
    if conditions is not None:
        conditions = conditions[inds]
        if cond_order is not None:
            y_labels = cond_order
        else:
            y_labels = np.unique(conditions)
        # Convert the conditions to numbers:
        for ind, cond in enumerate(y_labels):
            conditions[np.where(conditions == cond)[0]] = ind
        hlines_loc = np.where(np.diff(conditions.astype(int)) == 1)[0] + 1
        # Plot horizontal lines to delimitate the conditions:
        [ax.axhline(loc, color='k', linestyle=":") for loc in hlines_loc]
        # Add the tick marks in between each hline:
        ticks = []
        for ind, loc in enumerate(hlines_loc):
            if ind == 0:
                ticks.append(loc / 2)
            else:
                ticks.append(loc - ((loc - hlines_loc[ind - 1]) / 2))
        # Add the last tick:
        ticks.append(hlines_loc[-1] + ((data.shape[0] - hlines_loc[-1]) / 2))
        ax.set_yticks(ticks)
        ax.set_yticklabels(y_labels)

    # Add the axis labels and so on:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if vlines is not None:
        ax.vlines(vlines, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dashed', linewidth=1, colors='k')
    plt.tight_layout()
    cb = plt.colorbar(im)
    cb.ax.set_ylabel(cbar_label)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
    if filename is not None:
        # Save to png
        plt.savefig(filename, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        filename, file_extension = os.path.splitext(filename)
        plt.savefig(filename + ".svg", transparent=True)
        # Save all inputs to csv:
        np.savetxt(filename + "_data" + ".csv", data, delimiter=",")

    return ax


# plot_brain - surface plots
# credit where credit is due:
# This script heavily adapted from https://github.com/greydongilmore/seeg2bids-pipeline/blob/9055da475411032a49685ab6552253843b63b042/workflow/scripts/brain4views.py

# transformations from 3D -> 2D using PolyCollection
#   see also: https://matplotlib.org/matplotblog/posts/custom-3d-engine/

EPSILON = 1e-12  # small value to add to avoid division with zero


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens + EPSILON
    arr[:, 1] /= lens + EPSILON
    arr[:, 2] /= lens + EPSILON
    return arr


def normal_vectors(vertices, faces):
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(n)
    return n


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5 * np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def translate(x, y, z):
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float
    )


def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float
    )


def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )


def shading_intensity(vertices, faces, light=np.array([0, 0, 1]), shading=0.7):
    """shade calculation based on light source
       default is vertical light.
       shading controls amount of shading.
       Also saturates so top 20 % of vertices all have max intensity."""
    face_normals = normal_vectors(vertices, faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)] = 1
    shading = 0.7
    # top 20% all become fully coloured
    intensity = (1 - shading) + shading * (intensity - np.min(intensity)) / (
    (np.percentile(intensity, 80) - np.min(intensity)))
    # saturate
    intensity[intensity > 1] = 1

    return intensity


def plot_brain(subject='fsaverage', subjects_dir=None, surface='inflated', hemi='lh', sulc_map='curv',
               parc='aparc.a2009s', roi_map=None, roi_map_edge_color=None, roi_map_transparency=1.,
               views=['lateral', 'medial'],
               cmap='Oranges', colorbar=True, colorbar_title='ACC', colorbar_title_position='top', cmap_start=0.1,
               cmap_end=1.,
               vmin=None, vmax=None, overlay_method='overlay',
               overlays=None, overlay_threshold=None, outline_overlay=False,
               electrode_activations=None, vertex_distance=10., vertex_scaling='linear', vertex_scale=(1.0, 0.),
               vertex_method='max',
               scanner2tkr=(0, 0, 0), brain_cmap='Greys', brain_color_scale=(0.42, 0.58), brain_alpha=1, figsize=(8, 6),
               save_file=None, dpi=300):
    # plot brain surfaces and roi map overlays
    #
    # options:
    #   roi_map: dict of rois and associated values (e.g. ACC), example: {'G_front_middle': 0.98, 'Pole_temporal': 0.54, 'G_cuneus': 0.75}
    #   subject: freesurfer subject [default: fsaverage]
    #   subjects_dir: freesurfer subjects directory [default: none, find default freesurfer directory]
    #   surface: surface type to use, e.g., inflated, pial [default: 'inflated' ]
    #   hemi: hemisphere to use [default='lh']
    #   sulc_map: sulcus mapping to use, default is to use the curvature file [default: lh.curv]
    #   parc: parcellation file to use for roi_map. It should include the names of the roi_map keys [default: 'aparc.a2009s']
    #   views: views to show as strings [ 'lateral', 'medial'] or list of tuples as (x,y [,z]), x=rotation around horizontal plane, y = rotation on vertical plane, z = rotation on anterior/posterior plane
    #   colorbar: Add colorbar [default: True]
    #   cmap: colormap to use for roi_map [default: 'Oranges']
    #   overlay: load overlay (e.g. activation map) for coloring faces
    #   overlay_method: 'overlay' or 'blend' colors onto the brain [default: overlay]
    #
    #   electrode_activations:  Nx4 numpy array of electrodes coordinates (columsn 1-3) and activations (4th column)
    #   vertex_distance:        distance (in mm) of activation extent
    #   vertex_scaling:         method of activation decrease with distance (linear, gaussian, or exponential decay) [default: linear]
    #   vertex_scale:           tuple or list of float shape (scale_at_center, scale_at_distance). Extent of scaling [default: (1., 0.)]
    #   vertex_method:          Method for combining vertices [default: max]
    #
    #   brain_cmap: colormap used for brain [default: 'Greys']
    #   brain_color_scale: range of cmap values to use [default: (0.42, 0.58) produces narrow grey range used for standrd inflated surfaces]
    #   brain_alpha: transparency of the surface
    #   figsize: sigure size
    #   save_file: filename to save figure to
    #   dpi: dpi to save with

    # TODO: need to update lighting for non-standard views, (e.g., ventral, dorsal)

    # some checks
    assert type(figsize) is tuple and len(figsize) == 2, 'figsize must be tuple of length 2'
    if roi_map is not None:
        assert (parc is not None), 'Parcelelation file is required to map rois'
    if subjects_dir is None:
        subjects_dir = get_subjects_dir(raise_error=True)

    if not os.path.exists(surface):
        # generate from freesurfer folder
        surface = os.path.join(subjects_dir, subject, 'surf', hemi + os.path.extsep + surface)
    vertices, faces = read_surface(surface)
    if scanner2tkr is None:
        # get from original mri
        mri_orig = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
        scanner2tkr, _ = run_subprocess(['mri_info', '--scanner2tkr', mri_orig])
        scanner2tkr = np.fromstring(scanner2tkr, sep=' ').astype(float)
        scanner2tkr = scanner2tkr.reshape(4, 4)
        scanner2tkr = scanner2tkr[0:3, -1]
    vertices -= scanner2tkr

    vertices_orig, faces_orig = vertices, faces  # needed by overlays in original form

    # read in annotation file
    if parc is not None:
        if not os.path.exists(parc):
            parc = os.path.join(subjects_dir, subject, 'label', hemi + os.path.extsep + parc + '.annot')
        assert os.path.exists(parc), ('could not locate parcellation file at %s' % parc)
        annot_vert, annot_tbl, annot_anatomy = _read_annot(parc)

    # parse views
    view_strings = {'lh': {'lateral': (-90, 0, 0), 'medial': (90, 0, 0), 'rostral': (0, 0, 0), 'caudal': (180, 0, 0),
                           'dorsal': (180, 0, 90), 'ventral': (0, 0, 90)},
                    'rh': {'lateral': (90, 0, 0), 'medial': (-90, 0, 0), 'rostral': (0, 0, 0), 'caudal': (180, 0, 0),
                           'dorsal': (180, 0, 90), 'ventral': (0, 0, 90)},
                    }
    v = []
    for vv in views:
        if type(vv) == str:
            v.append(view_strings[hemi][vv])
        else:
            v.append(vv + (0,))  # append tuple(0) to ensure at least 3 dims
    views = v

    ##################################
    # eletrode activations.
    # Needs to be done before any scaling of vertices since distances are computed
    vertex_activations = []
    if electrode_activations is not None:

        # Needs to be done on the pial surface since electrode distance to pial surface is required to map activation.
        # But after that it can be applied to the inflated since the vertex indices are the same
        if os.path.splitext(surface)[1] == '.pial':
            vertices_pial = vertices
        else:
            surface = os.path.join(subjects_dir, subject, 'surf', hemi + os.path.extsep + 'pial')
            vertices_pial, _ = read_surface(surface)
            vertices_pial -= scanner2tkr

        # find closest vertex
        for i in range(electrode_activations.shape[0]):
            coords = electrode_activations[i, 0:3]
            val = electrode_activations[i, 3]
            tmp = np.sqrt((vertices_pial[:, 0] - coords[0]) ** 2 + (vertices_pial[:, 1] - coords[1]) ** 2 + (
                        vertices_pial[:, 2] - coords[2]) ** 2)
            vertex_id = np.argmin(tmp)
            tmp = np.sqrt((vertices_pial[:, 0] - vertices_pial[vertex_id, 0]) ** 2 + (
                        vertices_pial[:, 1] - vertices_pial[vertex_id, 1]) ** 2 + (
                                      vertices_pial[:, 2] - vertices_pial[vertex_id, 2]) ** 2)
            vertex_nbrs = np.where(tmp < vertex_distance)[0]
            vertex_nbrs = vertex_nbrs[np.argsort(tmp[vertex_nbrs])]

            # update this to be based on distance from center vertex
            if vertex_scaling == 'guassian':
                mx = np.max(tmp[vertex_nbrs])
                sigma = 0.8 / 2
                vertex_vals = np.exp(-(((tmp[vertex_nbrs] / mx) ** 2) / (2 * sigma ** 2)));
            else:  # linear
                vertex_vals = val * (1 - tmp[vertex_nbrs] / vertex_distance)
            vertex_activations.append([vertex_nbrs, vertex_vals])

    vert_range = max(vertices.max(0) - vertices.min(0))
    vertices = (vertices - (vertices.max(0) + vertices.min(0)) / 2) / vert_range

    face_normals = normal_vectors(vertices, faces)
    light = np.array([0, 0, 1])
    intensity = np.dot(face_normals, light)
    shading = 0.7  # shading 0-1. 0=none. 1=full
    # top 20% all become fully colored
    denom = np.percentile(intensity, 80) - np.min(intensity)
    intensity = (1 - shading) + shading * (intensity - np.min(intensity)) / denom
    intensity[intensity > 1] = 1

    # instantiate face colors
    face_colors = np.ones((faces.shape[0], 4))

    # generate sulcal map definition
    sulc = np.ones(vertices.shape[0]) * 0.5
    if sulc_map is not None:
        # read in sulcal file (typically curv file)
        if os.path.exists(sulc_map):
            # full path file
            sulc_file = sulc_map
        else:
            sulc_file = os.path.join(subjects_dir, subject, 'surf', hemi + os.path.extsep + sulc_map)
        assert os.path.exists(sulc_file), ('could not locate sulc_map file at %s' % sulc_file)
        sulc = read_curvature(sulc_file, binary=False)
    sulc_faces = np.mean(sulc[faces], axis=1)

    # binarize sulcal map
    if sulc_faces.min() != sulc_faces.max():
        neg_sulc = np.where(sulc_faces <= 0)
        pos_sulc = np.where(sulc_faces > 0)
        sulc_faces[neg_sulc] = 0
        sulc_faces[pos_sulc] = 1

    # mask, unused for now
    # mask = np.zeros(vertices.shape[0]).astype(bool)

    # assign greyscale colormap to sulcal map faces
    greys = plt.get_cmap(brain_cmap, 512)
    greys_narrow = ListedColormap(greys(np.linspace(brain_color_scale[0], brain_color_scale[1], 256)))
    face_colors = greys_narrow(sulc_faces)

    ##################################
    # nifti overlays
    overlay = []
    avg_method = "mean"
    if overlays is not None:
        kept_indices = np.arange(sulc_faces.shape[0])
        img = load_img(overlays[0])

        # we ned the pial surface to do the proper mapping
        if os.path.splitext(surface)[1] == '.pial':
            surf = surface
        else:
            surf = os.path.join(subjects_dir, subject, 'surf', hemi + os.path.extsep + 'pial')

        overlay = vol_to_surf(img, surf)  # Open brain plot:
        # create face values from vertex values by selected avg methods
        if avg_method == "mean":
            overlay_faces = np.mean(overlay[faces], axis=1)
        elif avg_method == "median":
            overlay_faces = np.median(overlay[faces], axis=1)

        # if no vmin/vmax are passed figure them out from the data
        if vmin is None:
            vmin = np.nanmin(overlay_faces)
        if vmax is None:
            vmax = np.nanmax(overlay_faces)

        # colors for roi map based on cmap provided
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # reindex the colormap so it starts deeper into the colorscale (e.g. color values = 0.1-1 over the scale of vmin-vmax (ACC) of purple)
        if isinstance(cmap, list):
            cmap = ListedColormap(cmap)
        else:
            cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(cmap_start, cmap_end, 128)))

        # threshold if indicated
        if overlay_threshold is not None:
            valid_indices = np.where(np.abs(overlay_faces) > overlay_threshold)[0]
            kept_indices = [i for i in kept_indices if i in valid_indices]

        # assign colormap to overlay
        overlay_faces = overlay_faces - vmin
        overlay_faces = overlay_faces / (vmax - vmin)
        face_colors[kept_indices] = cmap(overlay_faces[kept_indices])

        if outline_overlay:
            ward = Parcellations(method='ward', n_parcels=50,
                                 standardize=False, smoothing_fwhm=None,
                                 memory='nilearn_cache', memory_level=1,
                                 verbose=0)
            ward.fit(img)
            ward_labels_img = ward.labels_img_
            texture = vol_to_surf(ward_labels_img, surf)
            texture[overlay == 0] = 0
            texture[overlay > overlay_threshold] = 1

            levels = np.unique(texture)
            labels = [None] * len(levels)
            for level, label in zip(levels, labels):
                roi_indices = np.where(texture == level)[0]
                faces_outside = _get_faces_on_edge(faces, roi_indices)
                face_colors[faces_outside] = [0., 0., 0., 1.]

    ##################################
    # roi labeling/coloring
    label_masks = []
    label_val = []
    label_outlines = []
    if roi_map is not None:

        vals = [val for val in roi_map.values()]
        # figure out vmin/vmax from label_val, if vmin/vmax not provided
        if vmin is None:
            vmin = np.min(vals)
        if vmax is None:
            vmax = np.max(vals)

        # colors for roi map based on cmap provided
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # reindex the colormap so it starts deeper into the colorscale (e.g. color values = 0.1-1 over the scale of vmin-vmax (ACC) of purple)
        if isinstance(cmap, list):
            cmap = ListedColormap(cmap)
        else:
            cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(cmap_start, cmap_end, 128)))

        # loop through the dictionary and color rois
        for i, roi in enumerate(roi_map):
            L_mask = np.zeros(vertices.shape[0]).astype(bool)
            idx_ = annot_anatomy.index(roi.encode())
            vertex_val = annot_tbl[idx_, -1]
            vertex_idx = np.where(annot_vert == vertex_val)[0]

            L_mask[vertex_idx] = 1  # label vertices = 1
            label_masks.append(L_mask)
            label_val.append(roi_map[roi])

            if roi_map_edge_color is not None:
                L_outline = np.zeros(vertices.shape[0]).astype(bool)
                # get edges of this label
                scalars = np.zeros(vertices.shape[0])
                scalars[vertex_idx] = 1
                keep_idx = _mesh_borders(faces, scalars)
                keep_idx = np.in1d(faces.ravel(), keep_idx)
                keep_idx.shape = faces.shape
                keep_idx = faces[np.any(keep_idx, axis=1)]
                keep_idx = np.unique(keep_idx)
                vertex_idx = keep_idx
                L_outline[vertex_idx] = 1

                label_outlines.append(L_outline)

        label_mask_faces = [np.median(L[faces], axis=1) for L in label_masks]

        # assign label faces to appropriate color
        for i, L in enumerate(label_mask_faces):
            L_idx = np.where(L >= 0.5)
            # blend (multiply) label color with underlying color
            # face_colors[L_idx] = face_colors[L_idx] * [1., 0., 0., 1.]
            if overlay_method == 'blend':
                face_colors[L_idx] = face_colors[L_idx] * cmap(norm(label_val[i])) * np.array(
                    (1, 1, 1, roi_map_transparency))
            else:
                face_colors[L_idx] = cmap(norm(label_val[i])) * np.array((1, 1, 1, roi_map_transparency))

        if roi_map_edge_color is not None:
            if isinstance(roi_map_edge_color, str) and (roi_map_edge_color == 'thin'):
                texture = np.zeros(vertices.shape[0]).astype(bool)
                texture[faces[:, 0]] = (face_colors[:, 0] != face_colors[:, 1])
                levels = np.unique(texture)
                labels = [None] * len(levels)
                for level, label in zip(levels, labels):
                    roi_indices = np.where(texture == level)[0]
                    faces_outside = _get_faces_on_edge(faces, roi_indices)
                    face_colors[faces_outside] = [0., 0., 0., 1.]
            else:
                label_outline_faces = [np.median(L[faces], axis=1) for L in label_outlines]
                # assign label faces to appropriate color
                for i, L in enumerate(label_outline_faces):
                    L_idx = np.where(L >= 0.5)
                    # blend (multiply) label color with underlying color
                    # face_colors[L_idx] = face_colors[L_idx] * [1., 0., 0., 1.]
                    face_colors[L_idx] = roi_map_edge_color + list([1])

    ##################################
    # vertex coloring for activations
    if len(vertex_activations):

        # figure out vmin/vmax from activation values
        if vmin is None:
            vmin = np.min(electrode_activations[:, 3].min())
        if vmax is None:
            vmax = np.max(electrode_activations[:, 3].max())

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = ListedColormap(cm.get_cmap(cmap)(np.linspace(0.1, 1, 128)))
        # compile all vertex indices into a dict and then get mean value per index
        indices = []
        for vertex_nbrs, vertex_vals in vertex_activations:
            indices.append(vertex_nbrs)
        indices = np.concatenate(indices)
        vert_indices = {k: [] for k in indices}

        # now add values to them
        for vertex_nbrs, vertex_vals in vertex_activations:
            for i, idx in enumerate(vertex_nbrs):
                vert_indices[idx].append(vertex_vals[i])

        # finally add the face color based on these indices
        for idx in vert_indices.keys():
            idx_ = np.where(faces == idx)[0]
            if overlay_method == 'blend':
                face_colors[idx_] = face_colors[idx_] * cmap(norm(np.max(vert_indices[idx])))
            else:
                face_colors[idx_] = cmap(norm(np.max(vert_indices[idx])))

    # apply shading after the facecolors are set
    # face_colors[:, 0] *= intensity
    # face_colors[:, 1] *= intensity
    # face_colors[:, 2] *= intensity

    ##################################
    # Draw the plot
    fig = plt.figure(figsize=figsize)
    axwidth = 1 / len(views)
    for i, view in enumerate(views):
        MVP = (
                perspective(25, 1, 1, 100)
                @ translate(0, 0, -3 - (view[2] > 0) * 1)  # adjust if z-axis roated TODO: make more dynamic
                # the coordinate system is not like in mni space (e.g. x = left-to-right, y=anterior-to-posterior, z = ventral-to-dorsal)
                # first 2 rotaions, put in back in this type of coordinate systems. The view=[rotation_around_horizontal_pane   rotation_around_vertical_plane]
                @ xrotate(-90)
                @ zrotate(180)
                @ zrotate(view[0])  # around horizontal
                @ yrotate(view[1])  # around vertical
                @ xrotate(view[2])  # around other vertical
        )
        # adapt lighting
        light = np.array([0, 0, 1, 1]) @ zrotate(view[0]) @ yrotate(view[1]) @ xrotate(view[2])
        intensity = shading_intensity(vertices, faces, light=light[:3], shading=0.7)
        fcolor = face_colors.copy()
        fcolor[:, 0] *= intensity
        fcolor[:, 1] *= intensity
        fcolor[:, 2] *= intensity

        # translate coordinates based on viewing position
        V = np.c_[vertices, np.ones(len(vertices))] @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        V = V[faces]

        # triangle coordinates
        T = V[:, :, :2]
        # get Z values for ordering triangle plotting
        Z = -V[:, :, 2].mean(axis=1)
        # sort the triangles based on their z coordinate
        Zorder = np.argsort(Z)
        T, C = T[Zorder, :], fcolor[Zorder, :]

        # add subplot and plot PolyCollection
        # ax = fig.add_subplot(
        #  	1,
        #  	2,
        #  	i + 1,
        #  	xlim=[-1, +1],
        #  	ylim=[-0.6, +0.6],
        #  	frameon=False,
        #  	aspect=1,
        #  	xticks=[],
        #  	yticks=[],
        # )
        ax = fig.add_axes(
            ((i) * axwidth, 0, axwidth, 1),
            xlim=[-1, +1],
            ylim=[-0.6, +0.6],
            frameon=False,
            aspect=1,
            xticks=[],
            yticks=[],
            label=('mesh'),
        )
        collection = PolyCollection(
            T, closed=True, antialiased=False, facecolor=C, edgecolor=C, linewidth=0
        )
        collection.set_alpha(brain_alpha)
        # collection.set_rasterized(True)
        ax.add_collection(collection)
    plt.subplots_adjust(wspace=0)

    threshold = None
    if colorbar:

        our_cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        bounds = np.linspace(vmin, vmax, our_cmap.N)

        if threshold is None:
            ticks = [vmin, vmax]
        elif threshold == vmin:
            ticks = [vmin, vmax]
        else:
            if vmin >= 0:
                ticks = [vmin, threshold, vmax]
            else:
                ticks = [vmin, -threshold, threshold, vmax]

            cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
            # set colors to grey for absolute values < threshold
            istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
            istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
            for i in range(istart, istop):
                cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
            our_cmap = LinearSegmentedColormap.from_list(
                "Custom cmap", cmaplist, our_cmap.N
            )

        # we need to create a proxy mappable
        proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
        # proxy_mappable.set_array(overlay_faces)
        proxy_mappable.set_array(face_colors)
        cax = plt.axes([0.48, 0.5, 0.04, 0.2])
        cb = plt.colorbar(
            proxy_mappable,
            cax=cax,
            boundaries=bounds,
            ticks=ticks,
            drawedges=False,
            orientation="vertical",
        )
        cb.ax.tick_params(size=0, labelsize=8)
        if colorbar_title_position == 'top':
            cb.ax.set_title(colorbar_title, fontsize=10)
        elif colorbar_title_position == 'right':
            cb.ax.set_ylabel(colorbar_title, fontsize=10, rotation=270)
        elif colorbar_title_position == 'left':
            cb.ax.set_ylabel(colorbar_title, fontsize=10)
            cb.ax.yaxis.set_label_position('left')

    if save_file is not None:
        plt.savefig(save_file, dpi=dpi)
        # Save to svg:
        filename, file_extension = os.path.splitext(save_file)

        # raterize all "mesh" axes and save to pdf
        for axx in [ax for ax in fig.axes if ax.get_label() == 'mesh']:
            axx.collections[0].set_rasterized(True)

        # export pdf
        plt.savefig(filename + ".svg", transparent=True, dpi=dpi)
    return fig