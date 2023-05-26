import pandas as pd
from pathlib import Path
import config
import theories_rois
import glob
import os
import json
import numpy as np
import procrustes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from general_utilities import load_fsaverage_coord
from plotters import plot_matrix, mm2inch, plot_time_series
from ecog_plotters import plot_electrodes
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
from sklearn import manifold
import matplotlib.colors as colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# get the parameters dictionary
param = config.param
fig_size = param["figure_size_mm"]
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('font', size=param["font_size"])  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"])  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"])  # legend fontsize
plt.rc('figure', titlesize=param["font_size"])  # fontsize of the fi
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
                 }
mpl.rcParams.update(new_rc_params)
# Get the cmap and truncate it:
cmap = truncate_colormap(plt.get_cmap(param["colors"]["cmap"]), minval=0.2, maxval=1.0, n=200)

rois = theories_rois.rois
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "rsa"
sub = "super"
ses = "V1"
data_type = "ieeg"
zscore = True
p_val_thresh = 0.05
results_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "results")
figures_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "figure")
ylim = None
img_root = "/hpc/users/alexander.lepauvre/sw/github/plotting_uniformization/iEEG_rsa/images"

# Add smoothing parameters:
gaussian_sig_ms = 100
sfreq = 512 / 20
# Convert to samples:
gaussian_sig_samp_dflt = (gaussian_sig_ms / 2) * (sfreq / 1000)

# Get a list of all available identities:
avail_ids = {
    "face": np.unique(np.load("/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/rsa/"
                              "sub-super/ses-V1/ieeg/results/iit_face_identity_titr_1000ms_1500ms_all_to_all/"
                              "desbadcharej_notfil_lapref/"
                              "sub-super_ses-V1_task-Dur_analysis-rsa_iit_first_pres_labels.npy")),
    "object": np.unique(np.load("/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/rsa/"
                                "sub-super/ses-V1/ieeg/results/iit_object_identity_titr_1000ms_1500ms_all_to_all/"
                                "desbadcharej_notfil_lapref/"
                                "sub-super_ses-V1_task-Dur_analysis-rsa_iit_first_pres_labels.npy")),
    "letter": np.unique(np.load("/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/rsa/"
                                "sub-super/ses-V1/ieeg/results/iit_letter_identity_titr_1000ms_1500ms_all_to_all/"
                                "desbadcharej_notfil_lapref/"
                                "sub-super_ses-V1_task-Dur_analysis-rsa_iit_first_pres_labels.npy")),
    "false": np.unique(np.load("/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/rsa/"
                               "sub-super/ses-V1/ieeg/results/iit_false_identity_titr_1000ms_1500ms_all_to_all/"
                               "desbadcharej_notfil_lapref/"
                               "sub-super_ses-V1_task-Dur_analysis-rsa_iit_first_pres_labels.npy")),
}
iit_predicted_times = [0.2, 1.7]
iit_predicted_times_id = [0.2, 1.2]
gnw_predicted_times = [[0.3, 0.5], [1.8, 2.0]]
gnw_predicted_times_id = [[0.3, 0.5], [1.8, 2.0]]


def procrustes_rdm_agg(rdm, mask):
    # Find the RDM as this time point showing the highest within vs between corrected distances:
    proc_ref_ind = np.argmax(np.array([np.mean(rdm[iter_, :, :][~mask]) -
                                       np.mean(rdm[iter_, :, :][mask])
                                       for iter_ in range(rdm.shape[0])]))

    # Compute multidimensional scaling of the references:
    mds = PCA(n_components=2, random_state=0)
    ref_procrustes = rdm[proc_ref_ind, :, :]
    np.fill_diagonal(ref_procrustes, 0)
    ref_pca = mds.fit_transform(ref_procrustes)
    # Compute the PCA on each iteration:
    rdm_pcas = []
    for iter_ in range(rdm.shape[0]):
        iter_pca = rdm[iter_, :, :]
        np.fill_diagonal(iter_pca, 0)
        rdm_pcas.append(mds.fit_transform(iter_pca))
    # Aligned pcas:
    aligned_arr = procrustes.generalized(rdm_pcas, ref_pca)
    # Average across the aligned arrays:
    pts = np.mean(np.array(aligned_arr[0]), axis=0)
    return pts


def create_predicted_mat(time, predicted_times, tested_times):
    """

    """
    # Make the time array more fine grain:
    time = np.linspace(time[0], time[-1], 5000)
    pred_1d = np.zeros(time.shape[0])
    for t in predicted_times:
        onset_ind = np.where(time > t[0])[0][0]
        offset_ind = np.where(time >= t[1])[0][0]
        pred_1d[onset_ind:offset_ind] = 1
    # Creating a meshgrid out of it:
    xx, yy = np.meshgrid(pred_1d, pred_1d)
    predicted_mat = xx * yy

    # Same for the tested times:
    test_1d = np.zeros(time.shape[0]) + 0.5
    for t in tested_times:
        onset_ind = np.where(time > t[0])[0][0]
        offset_ind = np.where(time >= t[1])[0][0]
        test_1d[onset_ind:offset_ind] = 1
    # Creating a meshgrid out of it:
    xx, yy = np.meshgrid(test_1d, test_1d)
    test_mat = xx * yy

    # Multiplying the predicted matrix by the tested matrix:
    predicted_mat = predicted_mat * test_mat
    predicted_mat[np.where(predicted_mat == 0.25)] = 0.5
    return predicted_mat


def get_image(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def equate_rdm(rdms, label_1, label_2):
    """

    """
    constant_across_iter = np.all(label_1[0, :] == label_1) and np.all(label_2[0, :] == label_2)
    if constant_across_iter:
        # Subsample by selecting only the first occurence of a given label:
        lbl1_ind = np.unique(np.array([np.where(label_1[0, :] == lbl)[0][0] for lbl in label_1[0]]))
        lbl2_ind = np.unique(np.array([np.where(label_2[0, :] == lbl)[0][0] for lbl in label_2[0]]))
        rdms = rdms[:, :, lbl1_ind, :][:, :, :, lbl2_ind]
        # Average along first dimensions:
        rdms = np.average(rdms, axis=0)
    return rdms, np.unique(label_1), np.unique(label_2)


def get_rect_coord(time_win_dict):
    """

    """
    coords = []
    width = []
    for coord_x in time_win_dict["x"]:
        for coord_y in time_win_dict["y"]:
            coords.append([coord_x[0], coord_y[0]])
            width.append([coord_x[1] - coord_x[0],
                          coord_y[1] - coord_y[0]])
    return coords, width


def zscore_matrix(x, h0, axis=0):
    """
    This function computes a zscore between a value x and a
    :param x: (float) a single number for which to compute the zscore with respect ot the y distribution to the
    :param h0: (1d array) distribution of data with which to compute the std and mean:
    :param axis: (int) which axis along which to compute the zscore for the null distribution
    :return: zscore
    """
    assert isinstance(x, np.ndarray) and isinstance(h0, np.ndarray), "x and y must be numpy arrays!"
    assert len(h0.shape) == len(x.shape) + 1, "y must have 1 dimension more than x to compute mean and std over!"
    try:
        zscore = (x - np.mean(h0, axis=axis)) / np.std(h0, axis=axis)
    except ZeroDivisionError:
        Exception("y standard deviation is equal to 0, can't compute zscore!")

    return zscore


def rsa_plot_handler(folders_list, preprocess_steps=None, save_root=None, rdm_method="procrustes"):
    # Loop through each RSA folders:
    for folder in folders_list:
        print(folder)
        if "identity" in folder or "orientation" in folder or "gnw" in folder:
            ylim = [-2, 5]
            midpoint = 0.8
        else:
            ylim = [-2, 8]
            midpoint = 1.0
        # Load the analyses parameters:
        config_files = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*.json'))):
            config_files.append(file)
        with open(config_files[0], 'r') as f:
            rsa_param = json.load(f)
        save_dir = Path(save_root, folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # =====================================================================
        # Plot the cross temporal RSA:
        # Load the rsa results:
        cross_temp_rsa_file = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*rsa.npy'))):
            cross_temp_rsa_file.append(file)
        cross_temp_rsa = np.load(cross_temp_rsa_file[0])
        # Load the rsa label shuffled results:
        cross_temp_rsa_shuffle_file = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*rsa_label_shuffle.npy'))):
            cross_temp_rsa_shuffle_file.append(file)
        cross_temp_rsa_shuffle = np.load(cross_temp_rsa_shuffle_file[0])
        # Load the stats:
        cross_temp_stats = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*p_values.npy'))):
            cross_temp_stats.append(file)
        cross_temp_stats = np.load(cross_temp_stats[0])

        # zscore the results if needed:
        if zscore:
            cross_temp_rsa = zscore_matrix(np.mean(cross_temp_rsa, axis=0), cross_temp_rsa_shuffle)
        else:
            cross_temp_rsa = np.mean(cross_temp_rsa, axis=0)
        # Get the time limits of the RSA, depending on whether or not the data were cropped:
        if rsa_param["analysis_parameters"][folder]["equate_offset"]:
            tmin, tmax = rsa_param["analysis_parameters"][folder]["epo_onset_offset"][0], \
                         rsa_param["analysis_parameters"][folder]["epo_onset_offset"][1]
        else:
            tmin, tmax = rsa_param["analysis_parameters"][folder]["crop_time"][0], \
                         rsa_param["analysis_parameters"][folder]["crop_time"][1]
        times_to_plot = np.arange(-0.2, tmax, 1)
        time = np.linspace(tmin, tmax, cross_temp_rsa.shape[-1])
        # For identity, the data were cropped at 1.5. To make the matrices the same size, extending the rsa matrix with
        # nans until 2.0:
        # if tmax == 1.5:
        #     tmax_new = 2.0
        #     time_new = np.arange(tmin, tmax_new, time[1] - time[0])
        #     # Figure out how many pads are required:
        #     n_pads = time_new.shape[0] - time.shape[0]
        #     # Pad the array:
        #     cross_temp_rsa = np.pad(cross_temp_rsa, [(0, n_pads), (0, n_pads)], mode='constant',
        #     constant_values=np.nan)
        #     # Pad the stats array as well:
        #     if len(cross_temp_stats) > 0:
        #         cross_temp_stats = np.pad(cross_temp_stats, [(0, n_pads), (0, n_pads)],
        #                                   mode='constant', constant_values=np.nan)
        #     tmax = tmax_new
        #     time = time_new
        # else:
        #     n_pads = None
        n_pads = None
        # ===============================================================================================
        # Plot the cross temporal RSA matrix:
        # ====================================================================

        # Finally, create the predicted matrix:
        for theory in rsa_param["analysis_parameters"][folder]["theories_predictions"][0]:
            pred_mat = \
                create_predicted_mat(time,
                                     rsa_param["analysis_parameters"][folder]["theories_predictions"][0][theory]["x"],
                                     rsa_param["analysis_parameters"][folder][
                                         "matrix_subsampling_coordinates"]["200ms"]["x"])
            file_name = Path(save_dir, "pred_mat_{}.png".format(theory))
            ax = plot_matrix(pred_mat, tmin, tmax, tmin, tmax, mask=None, cmap="Greys", ax=None, ylim=[0, 1],
                             midpoint=0.5,
                             transparency=1.0,
                             xlabel="Time (s)", ylabel="Time (s)", cbar_label=None, filename=None, vline=0,
                             title=None, square_fig=True, do_cbar=True, c_contour="grey", interpolation=None)
            ax.set_xticks([0, 0.5, 1.0, 1.5])
            coord, width = get_rect_coord(
                rsa_param["analysis_parameters"][folder]["matrix_subsampling_coordinates"]["200ms"])
            for ind, xy in enumerate(coord):
                rect = patches.Rectangle((xy[0], xy[1]), width[ind][0], width[ind][1], linewidth=1,
                                         edgecolor=[0, 0, 0], facecolor='none', linestyle="-")
                ax.add_patch(rect)
            plt.tight_layout()
            # Save file:
            plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
            # Save to svg:
            file_name, file_extension = os.path.splitext(file_name)
            plt.savefig(file_name + ".svg", transparent=True)
            plt.close()

        # Plot without the stats
        file_name = Path(save_dir, "rsa_mat.png")
        # First with the colorbar:
        ax = plot_matrix(cross_temp_rsa, tmin, tmax, tmin, tmax, mask=None, cmap=cmap, ax=None, ylim=ylim,
                         midpoint=midpoint,
                         transparency=1.0,
                         xlabel="Time (s)", ylabel="Time (s)", cbar_label="z-score", filename=None, vline=0,
                         title=None, square_fig=True, c_contour="grey", crop_cbar=False)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        coord, width = get_rect_coord(
            rsa_param["analysis_parameters"][folder]["matrix_subsampling_coordinates"]["200ms"])
        for ind, xy in enumerate(coord):
            rect = patches.Rectangle((xy[0], xy[1]), width[ind][0], width[ind][1], linewidth=1,
                                     edgecolor=[0, 0, 0], facecolor='none', linestyle="-")
            ax.add_patch(rect)
        plt.tight_layout()
        # Save file:
        plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        file_name, file_extension = os.path.splitext(file_name)
        plt.savefig(file_name + ".svg", transparent=True)
        plt.close()

        # And without the color bar:
        file_name = Path(save_dir, "rsa_mat_no_cb.png")
        ax = plot_matrix(cross_temp_rsa, tmin, tmax, tmin, tmax, mask=None, cmap=cmap, ax=None, ylim=ylim,
                         midpoint=midpoint,
                         transparency=1.0,
                         xlabel="Time (s)", ylabel="Time (s)", cbar_label="z-score", filename=None, vline=0,
                         title=None, square_fig=True, do_cbar=False, c_contour="grey", crop_cbar=False)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        coord, width = get_rect_coord(
            rsa_param["analysis_parameters"][folder]["matrix_subsampling_coordinates"]["200ms"])
        for ind, xy in enumerate(coord):
            rect = patches.Rectangle((xy[0], xy[1]), width[ind][0], width[ind][1], linewidth=1,
                                     edgecolor=[0, 0, 0], facecolor='none', linestyle="-")
            ax.add_patch(rect)
        plt.tight_layout()
        # Save file:
        plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        file_name, file_extension = os.path.splitext(file_name)
        plt.savefig(file_name + ".svg", transparent=True)
        plt.close()

        # ====================================================================
        # Plot with the stats:
        mask = cross_temp_stats < p_val_thresh
        # With color bar:
        file_name = Path(save_dir, "rsa_mat_stats.png")
        ax = plot_matrix(cross_temp_rsa.copy(), tmin, tmax, tmin, tmax, mask=mask, cmap=cmap, ax=None, ylim=ylim,
                         midpoint=midpoint,
                         transparency=1.0,
                         xlabel="Time (s)", ylabel="Time (s)", cbar_label="z-score", filename=None, vline=0,
                         title=None, square_fig=True, do_cbar=True, c_contour=[[0.3, 0.3, 0.3]], crop_cbar=False)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        coord, width = get_rect_coord(
            rsa_param["analysis_parameters"][folder]["matrix_subsampling_coordinates"]["200ms"])
        for ind, xy in enumerate(coord):
            rect = patches.Rectangle((xy[0], xy[1]), width[ind][0], width[ind][1], linewidth=1,
                                     edgecolor=[0, 0, 0], facecolor='none', linestyle="-")
            ax.add_patch(rect)
        plt.tight_layout()
        # Save file:
        plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        file_name, file_extension = os.path.splitext(file_name)
        plt.savefig(file_name + ".svg", transparent=True)
        plt.close()

        file_name = Path(save_dir, "rsa_mat_stats_no_cb.png")
        ax = plot_matrix(cross_temp_rsa.copy(), tmin, tmax, tmin, tmax, mask=mask, cmap=cmap, ax=None, ylim=ylim,
                         midpoint=midpoint,
                         transparency=1.0,
                         xlabel="Time (s)", ylabel="Time (s)", cbar_label="z-score", filename=None, vline=0,
                         title=None, square_fig=True, do_cbar=False, c_contour="grey", crop_cbar=False)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        coord, width = get_rect_coord(
            rsa_param["analysis_parameters"][folder]["matrix_subsampling_coordinates"]["200ms"])
        for ind, xy in enumerate(coord):
            rect = patches.Rectangle((xy[0], xy[1]), width[ind][0], width[ind][1], linewidth=1,
                                     edgecolor=[0, 0, 0], facecolor='none', linestyle="-")
            ax.add_patch(rect)
        plt.tight_layout()
        # Save file:
        plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
        # Save to svg:
        file_name, file_extension = os.path.splitext(file_name)
        plt.savefig(file_name + ".svg", transparent=True)
        plt.close()

        # ====================================================================
        # Load the rdm:
        diag_rdm_file = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*sample_rdm.npy'))):
            diag_rdm_file.append(file)
        diag_rdm = np.load(diag_rdm_file[0])
        # Load the first and second presentation labels:
        first_pres_labels = []
        second_pres_labels = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*first_pres_labels.npy'))):
            first_pres_labels.append(file)
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*second_pres_labels.npy'))):
            second_pres_labels.append(file)
        first_pres_labels = np.load(first_pres_labels[0])
        try:
            second_pres_labels = np.load(second_pres_labels[0])
        except IndexError:
            second_pres_labels = None

        # Get the features matrix:
        features_mat_file = []
        for file in glob.glob(str(Path(results_root, folder, preprocess_steps, '*features.npy'))):
            features_mat_file.append(file)
        try:
            features_mat = np.load(features_mat_file[0])
            # Load the channels mni coordinates:
            subjects_list = list(set([ch.split("-")[0] for ch in list(np.unique(features_mat))]))
            fsaverage_coords = load_fsaverage_coord(bids_root, subjects_list, ses='V1')
        except IndexError:
            print("No saved features: ")
            features_mat = None
            fsaverage_coords = None
        except ValueError:
            print("No saved features: ")
            features_mat = None
            fsaverage_coords = None

        # Extract the labels from the first iteration, should be the same across all iterations
        if second_pres_labels is not None:
            first_pres_labels = first_pres_labels[0, :]
            second_pres_labels = second_pres_labels[0, :]
            trial_labels = np.concatenate([first_pres_labels, second_pres_labels])
        else:
            trial_labels = first_pres_labels[0, :]

        # Create the save file:
        save_dir_rdm = Path(save_root, folder, "rdm")
        if not os.path.exists(save_dir_rdm):
            os.makedirs(save_dir_rdm)
        for i, t in enumerate(times_to_plot):
            # ====================================================================
            # Plot the multidimension scaling:
            # Setting the color map depending on the conditions:
            if "face_vs_obj" in folder:
                c_dict = dict(face=param["colors"]["face"], object=param["colors"]["object"])
            elif "letter_vs_sym" in folder:
                c_dict = {"letter": param["colors"]["letter"],
                          "false": param["colors"]["false"]}
            elif "face_identity" in folder:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(avail_ids["face"]))
                sm = mpl.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
                c_dict = {lbl: sm.to_rgba(ind) for ind, lbl in enumerate(list(set(trial_labels)))}
            elif "object_identity" in folder:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(avail_ids["object"]))
                sm = mpl.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
                c_dict = {lbl: sm.to_rgba(ind) for ind, lbl in enumerate(list(set(trial_labels)))}
            elif "letter_identity" in folder:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(avail_ids["letter"]))
                sm = mpl.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
                c_dict = {lbl: sm.to_rgba(ind) for ind, lbl in enumerate(list(set(trial_labels)))}
            elif "false_identity" in folder:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(avail_ids["false"]))
                sm = mpl.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
                c_dict = {lbl: sm.to_rgba(ind) for ind, lbl in enumerate(list(set(trial_labels)))}
            elif "orientation" in folder:
                c_dict = {ori: sns.color_palette("colorblind")[ind]
                          for ind, ori in enumerate(list(set(trial_labels)))}
            else:
                c_dict = None

            # Set the save dir:
            file_name = Path(save_dir_rdm, "0{}_{:.2f}s_mds.png".format(i, t))
            # Find the index of the time to plot:
            ind = np.where(time > t)[0][0]
            # Get the mask:
            msk = np.meshgrid(trial_labels, trial_labels)[1] == \
                  np.meshgrid(trial_labels, trial_labels)[0]

            if rdm_method == "procrustes":
                pts = procrustes_rdm_agg(np.squeeze(diag_rdm[:, ind, ...]), msk)
            elif rdm_method == "highest":
                highest_ind = np.argmax(np.array([np.mean(diag_rdm[iter_, ind, :, :][~msk]) -
                                                  np.mean(diag_rdm[iter_, ind, :, :][msk])
                                                  for iter_ in range(diag_rdm.shape[0])]))

                # Compute multidimensional scaling of the references:
                mds = PCA(n_components=2, random_state=0)
                high_rdm = diag_rdm[highest_ind, ind, :, :]
                np.fill_diagonal(high_rdm, 0)
                pts = mds.fit_transform(high_rdm)
            elif rdm_method == "avg":
                avg_rdm = np.mean(diag_rdm[:, ind, :, :], axis=0)
                mds = PCA(n_components=2, random_state=0)
                np.fill_diagonal(avg_rdm, 0)
                pts = mds.fit_transform(avg_rdm)

            # Plot the RDM:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                            mm2inch(fig_size[0])])
            if "identity" in folder:
                oris = np.array(["center", "left", "right"] * int(trial_labels.shape[0] / 3))
                ctr = 0
                for x0, y0, img_file in zip(pts[:, 0], pts[:, 1], trial_labels):
                    ori = oris[ctr]
                    ctr += 1
                    ab = AnnotationBbox(get_image(Path(img_root, ori, img_file + ".png"), zoom=0.1), (x0, y0),
                                        frameon=True, bboxprops={"boxstyle": "Square, pad=0", "alpha": 0.8,
                                                                 "edgecolor": c_dict[img_file],
                                                                 "linewidth": 2
                                                                 })
                    ax.add_artist(ab)
            elif "orientation" in folder:
                if "face" in folder:
                    img_file = avail_ids["face"]
                elif "object" in folder:
                    img_file = avail_ids["object"]
                elif "letter" in folder:
                    img_file = avail_ids["letter"]
                elif "false" in folder:
                    img_file = avail_ids["false"]
                for x0, y0, ori in zip(pts[:, 0], pts[:, 1], trial_labels):
                    img = np.random.choice(img_file, size=1)[0]
                    ab = AnnotationBbox(get_image(Path(img_root, ori.lower(), img + ".png"), zoom=0.1), (x0, y0),
                                        frameon=True, bboxprops={"boxstyle": "Square, pad=0", "alpha": 0.8,
                                                                 "edgecolor": c_dict[ori],
                                                                 "linewidth": 2
                                                                 })
                    ax.add_artist(ab)
            else:
                for x0, y0, cate in zip(pts[:, 0], pts[:, 1], trial_labels):
                    ori = np.random.choice(["center", "left", "right"], size=1)[0]
                    img_file = np.random.choice(avail_ids[cate], size=1)[0]
                    ab = AnnotationBbox(get_image(Path(img_root, ori, img_file + ".png"), zoom=0.1), (x0, y0),
                                        frameon=True, bboxprops={"boxstyle": "Square, pad=0", "alpha": 0.8,
                                                                 "edgecolor": c_dict[cate],
                                                                 "linewidth": 2
                                                                 })
                    ax.add_artist(ab)
            ax.spines[['right', 'top']].set_visible(False)
            # Set the tick marks to be the same between both axes:
            xrange = np.max(pts) - np.min(pts)
            ax.set_xlim([np.min(pts) - 0.2 * xrange, np.max(pts) + 0.1 * xrange])
            ax.set_ylim([np.min(pts) - 0.2 * xrange, np.max(pts) + 0.1 * xrange])
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            plt.tight_layout()
            plt.axis('square')
            # Save file:
            plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
            # Save to svg:
            file_name, file_extension = os.path.splitext(file_name)
            plt.savefig(file_name + ".svg", transparent=True)
            plt.close()

            # =====================================================================
            # Plot the selected features:
            if features_mat is not None:
                # Get the features at that sample:
                feat = features_mat[:, ind, :]
                # Count how often each channel was selected:
                ch_ctr = {ch: (feat == ch).sum() for ch in list(np.unique(feat))}
                # Get the max and min values:
                cts_val = [ch_ctr[ch] for ch in ch_ctr.keys()]
                # Create the color map:
                norm = mpl.colors.Normalize(vmin=min(cts_val), vmax=max(cts_val))
                scalar_map = cm.ScalarMappable(norm=norm, cmap=param["colors"]["cmap"])
                # Looping through each channel again to get the relevant info:
                coords = fsaverage_coords.loc[fsaverage_coords["name"].isin(list(ch_ctr.keys()))]
                coords = coords.rename(columns={"name": "channel"})
                ch_colors = pd.DataFrame()
                for ch in ch_ctr.keys():
                    if ch_ctr[ch] == 0:
                        continue
                    ch_colors = ch_colors.append(pd.DataFrame({
                        "channel": ch,
                        "r": scalar_map.to_rgba(ch_ctr[ch])[0],
                        "g": scalar_map.to_rgba(ch_ctr[ch])[1],
                        "b": scalar_map.to_rgba(ch_ctr[ch])[2]
                    }, index=[0]))
                ch_colors = ch_colors.reset_index(drop=True)
                # Save to csv:
                save_dir = Path(save_root, folder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name_coords = Path(save_dir, "{:.2f}s_coords.csv".format(t))
                file_name_colors = Path(save_dir, "{:.2f}s_coords_colors.csv".format(t))
                file_name_roi = Path(save_dir, "rois_dict.json")
                coords.to_csv(file_name_coords)
                # Convert the channel colors to a dataframe:
                ch_colors.to_csv(file_name_colors)

                # Generate a dict of the ROI:
                rois_dict = {roi.replace("ctx_rh_", "").replace("ctx_lh_", ""): [1, 1, 1]
                             for roi in
                             rsa_param["rois"][list(rsa_param["rois"].keys())[0]]}
                with open(file_name_roi, 'w') as fp:
                    json.dump(rois_dict, fp)

            # =========================================================================================
            # Compute the within and between correlation distances:
            mean_within = []
            mean_between = []
            # Get the mask:
            msk = np.meshgrid(trial_labels, trial_labels)[1] == \
                  np.meshgrid(trial_labels, trial_labels)[0]
            for ii in range(diag_rdm.shape[0]):
                rdm_ii = diag_rdm[ii, ind, :, :]
                np.fill_diagonal(rdm_ii, np.nan)
                mean_within.append(np.nanmean(rdm_ii[msk]))
                mean_between.append(np.nanmean(rdm_ii[~msk]))
            # Plot the within - between correlation distances:
            file_name = Path(save_dir, "within_between_bar_t{:.2f}.png".format(t))
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[1]),
                                            mm2inch(fig_size[0])])
            lbl = ['within', 'between']
            vals = [np.mean(mean_within), np.mean(mean_between)]
            errs = [np.std(mean_within), np.std(mean_between)]
            ax.bar(lbl[0], vals[0], yerr=errs[0], color="grey")
            ax.bar(lbl[1], vals[1], yerr=errs[1], color="k")
            ax.set_ylabel("Correlation distance")
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylim([0.5, max(vals)])
            plt.tight_layout()
            # Save file:
            plt.savefig(file_name, transparent=True, dpi=param["fig_res_dpi"])
            # Save to svg:
            file_name, file_extension = os.path.splitext(file_name)
            plt.savefig(file_name + ".svg", transparent=True)
            plt.close()

        # Plot the degree of agreement of features in time:
        if features_mat is not None:
            # Generate an array to store for each channel whether it was picked at a given time point or not:
            feat_unique = np.unique(features_mat)
            features_bin = np.zeros((features_mat.shape[0], features_mat.shape[1], np.unique(features_mat).shape[0]))
            feat_corr = []
            for samp in range(features_mat.shape[1]):
                if samp + 2 > features_mat.shape[1]:
                    continue
                else:
                    # Count how often each feature appears in each time point:
                    t0 = [np.sum(np.squeeze(features_mat[:, samp, :]) == ch) for ch in feat_unique]
                    t1 = [np.sum(np.squeeze(features_mat[:, samp + 1, :]) == ch) for ch in feat_unique]
                    feat_corr.append(pearsonr(t0, t1).correlation)
            # Convert to a numpy array:
            feat_corr = np.array(feat_corr)
            if gaussian_sig_samp_dflt > 0:
                feat_corr = gaussian_filter1d(feat_corr, gaussian_sig_samp_dflt, axis=-1)
            if n_pads is not None:
                feat_corr = np.pad(feat_corr, [(0, n_pads)], mode="constant", constant_values=np.nan)
            plot_time_series(feat_corr[np.newaxis, ...], tmin, tmax, err=None, colors=None, vlines=None, ylim=[0, 1],
                             xlabel="Time (s)", ylabel="Agreement proportion", err_transparency=0.2,
                             filename=Path(save_dir, "features_agreements.png"),
                             title=None, square_fig=False, conditions=None, do_legend=False,
                             patches=None, patch_color="r", patch_transparency=0.2)


if __name__ == "__main__":
    folders_list = [
        "gnw_face_identity_titr_1000ms_1500ms_all_to_all", "gnw_face_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "gnw_object_identity_titr_1000ms_1500ms_all_to_all",
        "gnw_object_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "gnw_letter_identity_titr_1000ms_1500ms_all_to_all",
        "gnw_letter_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "gnw_false_identity_titr_1000ms_1500ms_all_to_all", "gnw_false_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "gnw_face_orientation_titr_1500ms_all_to_all", "gnw_face_orientation_titr_1500ms_all_to_all_200_feat",
        "gnw_object_orientation_titr_1500ms_all_to_all", "gnw_object_orientation_titr_1500ms_all_to_all_200_feat",
        "gnw_letter_orientation_titr_1500ms_all_to_all", "gnw_letter_orientation_titr_1500ms_all_to_all_200_feat",
        "gnw_false_orientation_titr_1500ms_all_to_all", "gnw_false_orientation_titr_1500ms_all_to_all_200_feat",
        "gnw_face_vs_obj_ti_1500ms_all_to_all", "gnw_face_vs_obj_ti_1500ms_all_to_all_200_feat",
        "gnw_face_vs_obj_tr_1500ms_all_to_all", "gnw_face_vs_obj_tr_1500ms_all_to_all_200_feat",
        "gnw_letter_vs_sym_ti_1500ms_all_to_all", "gnw_letter_vs_sym_ti_1500ms_all_to_all_200_feat",
        "gnw_letter_vs_sym_tr_1500ms_all_to_all", "gnw_letter_vs_sym_ti_1500ms_all_to_all_200_feat",
        "iit_face_identity_titr_1000ms_1500ms_all_to_all", "iit_face_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "iit_object_identity_titr_1000ms_1500ms_all_to_all",
        "iit_object_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "iit_letter_identity_titr_1000ms_1500ms_all_to_all",
        "iit_letter_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "iit_false_identity_titr_1000ms_1500ms_all_to_all", "iit_false_identity_titr_1000ms_1500ms_all_to_all_200_feat",
        "iit_face_orientation_titr_1500ms_all_to_all", "iit_face_orientation_titr_1500ms_all_to_all_200_feat",
        "iit_object_orientation_titr_1500ms_all_to_all", "iit_object_orientation_titr_1500ms_all_to_all_200_feat",
        "iit_letter_orientation_titr_1500ms_all_to_all", "iit_letter_orientation_titr_1500ms_all_to_all_200_feat",
        "iit_false_orientation_titr_1500ms_all_to_all", "iit_false_orientation_titr_1500ms_all_to_all_200_feat",
        "iit_face_vs_obj_ti_1500ms_all_to_all", "iit_face_vs_obj_ti_1500ms_all_to_all_200_feat",
        "iit_face_vs_obj_tr_1500ms_all_to_all", "iit_face_vs_obj_tr_1500ms_all_to_all_200_feat",
        "iit_letter_vs_sym_ti_1500ms_all_to_all", "iit_letter_vs_sym_ti_1500ms_all_to_all_200_feat",
        "iit_letter_vs_sym_tr_1500ms_all_to_all", "iit_letter_vs_sym_ti_1500ms_all_to_all_200_feat"
    ]
    rsa_plot_handler(folders_list, preprocess_steps="desbadcharej_notfil_lapref",
                     save_root="/hpc/users/alexander.lepauvre/plotting_test/rsa", rdm_method="procrustes")
