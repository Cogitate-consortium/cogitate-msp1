"""
This script computes the correlation between the observed and predicted RSA
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
from pathlib import Path
import seaborn as sns

from general_helper_functions.data_general_utilities import zscore_mat
from rsa.rsa_helper_functions import *
from general_helper_functions.pathHelperFunctions import find_files, path_generator
from rsa.rsa_parameters_class import RsaParameters
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi
DO_ZSCORE = True
zscore_lim = [-5, 10]
FDR = "fdr_bh"
theory_corr_if_non_sig = True  # Whether or not to make sure that the RSA results are significant before computing


# theories correlations


def theories_correlations(configs, save_folder="super"):
    if len(configs) == 0 or configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")

    # ==================================================================================================================
    # Looping through all the passed configs:
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = \
            RsaParameters(config, sub_id=save_folder)
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
            stats_results = pd.DataFrame()

            # Getting the vmin and vmax for the matrices:
            vmin = []
            vmax = []
            for roi in param.rois:
                rsa_results = find_files(load_path_results, "*" + roi + "_rsa.", extension="npy")
                assert len(rsa_results) == 1, "More than one file was found for rsa results!"
                rsa = np.load(rsa_results[0])
                vmin.append(np.min(rsa))
                vmax.append(np.max(rsa))
            if DO_ZSCORE:
                vmin, vmax = zscore_lim[0], zscore_lim[1]
            else:
                vmin, vmax = np.min(vmin), np.max(vmax)

            # Looping through each subsampling coordinates:
            for time_win in analysis_parameters["matrix_subsampling_coordinates"].keys():
                # Get the subsampling coordinates:
                subsamp_coord = analysis_parameters["matrix_subsampling_coordinates"][time_win]

                for roi_ind, roi in enumerate(param.rois):
                    # ==================================================================================================
                    # Loading the data:
                    # Finding the results files:
                    rsa_results = find_files(load_path_results, "*" + roi + "_rsa.", extension="npy")
                    rsa_p_values = find_files(load_path_results, "*" + roi + "_rsa_p_values.", extension="npy")
                    rsa_label_shuffle = find_files(load_path_results, "*" + roi + "_rsa_label_shuffle",
                                                   extension=".npy")
                    sample_rdm = find_files(load_path_results, "*" + roi + "_sample_rdm", extension=".npy")
                    assert len(rsa_results) == 1, "More than one file was found for rsa results!"
                    assert len(rsa_p_values) == 1, "More than one file was found for rsa results!"
                    assert len(rsa_label_shuffle) == 1, "More than one file was found for rsa results!"
                    assert len(sample_rdm) == 1, "More than one file was found for sample_rdm!"
                    # Loading the results:
                    rsa = np.load(rsa_results[0])
                    rsa_p_values = np.load(rsa_p_values[0])
                    rsa_label_shuffle = np.load(rsa_label_shuffle[0])

                    # Averaging the rsa results along the first dimension:
                    avg_rsa = np.mean(np.array(rsa), axis=0)

                    if DO_ZSCORE:
                        avg_rsa = zscore_mat(avg_rsa, rsa_label_shuffle, axis=0)
                        rsa_label_shuffle = [zscore_mat(rsa_label_shuffle[i], rsa_label_shuffle)
                                             for i in range(rsa_label_shuffle.shape[0])]

                    if analysis_parameters["equate_offset"]:
                        start_n_end = analysis_parameters["epo_onset_offset"]
                    else:
                        start_n_end = analysis_parameters["crop_time"]

                    # ==================================================================================================
                    # Downsample the observed matrices:
                    subsamp_rsa_matrix, subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict = \
                        subsample_matrices(avg_rsa, start_n_end[0],
                                           start_n_end[1],
                                           subsamp_coord)
                    # Adjust the p values in case there was nothing:
                    if rsa_p_values.size == 0:
                        rsa_p_values = np.zeros(avg_rsa.shape) + 1
                    # Downsample the p values:
                    subsamp_pval_matrix, _, _, _ = \
                        subsample_matrices(rsa_p_values, start_n_end[0],
                                           start_n_end[1],
                                           subsamp_coord)
                    # Plot the downsampled matrices:
                    fig, ax = plt.subplots(figsize=[20, 15])
                    # Plot the matrix:
                    im = ax.imshow(subsamp_rsa_matrix, cmap=cmap, origin='lower',
                                   aspect='equal',
                                   vmin=vmin, vmax=vmax)
                    ax.set_title("{} RSA, ROI: {}".format(analysis_parameters["rsa_condition"], roi))
                    cbar = plt.colorbar(im, ax=ax)
                    if DO_ZSCORE:
                        cbar.ax.set_ylabel('zscore')
                    else:
                        cbar.ax.set_ylabel('Correlation difference within vs between')
                    cbar.ax.yaxis.set_label_position('left')
                    # Adding the axes titles:
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Time (s)')
                    # Adding the matrices demarcations in case of subsampling:
                    [ax.axhline(ind + 0.5, color='k', linestyle='--')
                     for ind in matrices_delimitations_ref]
                    [ax.axvline(ind + 0.5, color='k', linestyle='--')
                     for ind in matrices_delimitations_ref]
                    # Adding axis break to mark the difference:
                    d = 0.01
                    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                    # Looping through each demarcations to mark them::
                    for ind in matrices_delimitations_ref:
                        ind_trans = (ind + 1) / len(subsamp_rsa_matrix)
                        ax.plot((ind_trans - 0.005 - d, ind_trans
                                 - 0.005 + d), (-d, +d), **kwargs)
                        ax.plot((ind_trans + 0.005 - d, ind_trans
                                 + 0.005 + d), (-d, +d), **kwargs)
                        ax.plot((-d, +d), (ind_trans - 0.005 - d,
                                           ind_trans - 0.005 + d), **kwargs)
                        ax.plot((-d, +d), (ind_trans + 0.005 - d,
                                           ind_trans + 0.005 + d), **kwargs)
                    # Generate the ticks:
                    ticks_pos = np.linspace(0, subsamp_rsa_matrix.shape[0] - 1, 8)
                    # Generate the tick position and labels:
                    ticks_labels = [str(subsampled_time_ref[int(ind)]) for ind in ticks_pos]
                    ax.set_xticks(ticks_pos)
                    ax.set_yticks(ticks_pos)
                    ax.set_xticklabels(ticks_labels)
                    ax.set_yticklabels(ticks_labels)
                    plt.tight_layout()
                    # Save the figure:
                    if DO_ZSCORE:
                        file_name = Path(save_path_fig, param.files_prefix + "subsampled_matrix" + roi
                                         + "_" + time_win + "_zscore_rsa.png")
                    else:
                        file_name = Path(save_path_fig, param.files_prefix + "subsampled_matrix" + roi
                                         + "_" + time_win + "_rsa.png")
                    plt.savefig(file_name, transparent=True)
                    plt.close()

                    # Looping through each pairs of theories predictions:
                    for pred_pairs in analysis_parameters["theories_predictions"]:
                        # Generate a string to identify this pair of predictors:
                        pred_pair_id = "_".join(list(pred_pairs.keys()))

                        # Create the theories matrices:
                        theories_matrices = {}
                        for theory in pred_pairs:
                            theories_matrices[theory] = \
                                create_prediction_matrix(start_n_end[0],
                                                         start_n_end[1],
                                                         pred_pairs[theory],
                                                         len(avg_rsa[0]))
                            # Plot the full theories matrices:
                            fig, ax = plt.subplots(figsize=[20, 15])
                            ax.imshow(theories_matrices[theory], origin="lower", cmap=cmap, aspect="equal",
                                      extent=[start_n_end[0], start_n_end[-1], start_n_end[0], start_n_end[-1]],
                                      vmin=vmin, vmax=vmax)
                            ax.axvline(0, color='k')
                            ax.axhline(0, color='k')
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Time (s)')
                            plt.tight_layout()
                            file_name = Path(save_path_fig, param.files_prefix + theory + "_" + time_win +
                                             "_full_predicted_matrix.png")
                            plt.savefig(file_name, transparent=True)
                            plt.close()

                        # Creating list to store the time and delimitations of the other segmentations to make sure
                        # that they are correct:
                        subsample_time_test = []
                        matrices_delimitations_test = []

                        # Downsample the surrogate matrices:
                        subsamp_rsa_shuffle_matrix = []
                        for matrix in rsa_label_shuffle:
                            sub_samp_mat, subsamp_time, mat_lim, sub_matrix_dict = \
                                subsample_matrices(matrix, start_n_end[0],
                                                   start_n_end[1], subsamp_coord)
                            subsamp_rsa_shuffle_matrix.append(sub_samp_mat)
                            subsample_time_test.append(subsamp_time)
                            matrices_delimitations_test.append(mat_lim)

                        # Subsample the predicted matrices:
                        for theory in theories_matrices.keys():
                            theories_matrices[theory], subsamp_time, mat_lim, _ = \
                                subsample_matrices(theories_matrices[theory], start_n_end[0],
                                                   start_n_end[1], subsamp_coord)
                            subsample_time_test.append(subsamp_time)
                            matrices_delimitations_test.append(mat_lim)

                        # Making sure that all the time vectors are the same:
                        assert all([(subsampled_time_ref == time_array).all() for time_array in subsample_time_test]), \
                            "The times arrays are not the same between the different subsamplings!"
                        assert all([(matrices_delimitations_ref == mat_lim).all() for mat_lim in
                                    matrices_delimitations_test]), \
                            "The matrices delimitations are not the same between the different subsamplings!"
                        # If that works then we can proceed:
                        sub_samp_times = subsampled_time_ref
                        matrices_delimitations = matrices_delimitations_ref

                        # Finally, correlate the theories matrices with the observed matrices according to the different
                        # passed methods:
                        for corr_method in analysis_parameters["correlation_methods"]:
                            if (not theory_corr_if_non_sig and np.any(subsamp_pval_matrix <
                                                                      analysis_parameters["cluster_based_test_param"][
                                                                          "p_value_thresh"])) or theory_corr_if_non_sig:
                                # Compute the correlation to the observed matrix:
                                obs_corr, obs_corr_diff, perm_p_val, null_dist_diff, corr_pval = \
                                    rsa_shuffle_label_test(subsamp_rsa_matrix, subsamp_rsa_shuffle_matrix,
                                                           theories_matrices, corr_method)
                                # Appending the results to the results dictionary:
                                stats_results = stats_results.append(pd.DataFrame({
                                    "roi": roi,
                                    "time_win": time_win,
                                    "predictors": pred_pair_id,
                                    "correlation_method": corr_method,
                                    "correlation_difference": obs_corr_diff,
                                    "p-value": perm_p_val,
                                    **obs_corr,
                                    **corr_pval
                                }, index=[0]))

                                # Plotting the correlation results:
                                if null_dist_diff is not None:
                                    fig, ax = plt.subplots()
                                    ax.hist(null_dist_diff, density=True)
                                    ax.axvline(obs_corr_diff, color="red",
                                               label='p-val = {:.3f}'.format(perm_p_val))
                                    # Add the axis descs:
                                    if len(theories_matrices) == 2:
                                        ax.set_title(
                                            "Distribution correlation {0} - {1}".format(
                                                *list(theories_matrices.keys())))
                                    else:
                                        ax.set_title(
                                            "Distribution correlation {0}".format(theories_matrices.keys()))
                                    ax.set_ylabel("Probability")
                                    ax.set_xlabel("{0} correlation difference".format(corr_method))
                                    ax.legend()
                                    # Save the figure:
                                    plt.tight_layout()
                                    if DO_ZSCORE:
                                        plt.savefig(Path(save_path_fig, param.files_prefix +
                                                         "theories_corr_diff_" + roi + "_" + time_win + "_"
                                                         + pred_pair_id + "_" + corr_method + "_zscore.png"))
                                    else:
                                        plt.savefig(Path(save_path_fig, param.files_prefix +
                                                         "theories_corr_diff_" + roi + "_" + time_win + "_" +
                                                         pred_pair_id + "_" + corr_method + ".png"))
                                    plt.close()
                            else:
                                stats_results = stats_results.append(pd.DataFrame({
                                    "roi": roi,
                                    "time_win": time_win,
                                    "predictors": pred_pair_id,
                                    "correlation_method": corr_method,
                                    "correlation_difference": None,
                                    "p-value": 1,
                                    **{key: None for key in theories_matrices.keys()},
                                    **{key: 1 for key in theories_matrices.keys()}
                                }, index=[0]))

                        # ======================================================================
                        # 1. Plot the downsampled predicted matrices
                        if roi_ind == 0:  # Only once
                            for theory in theories_matrices.keys():
                                fig, ax = plt.subplots(figsize=[20, 15])
                                # Plot the matrix:
                                ax.imshow(theories_matrices[theory], cmap=cmap, origin='lower',
                                          aspect='equal')
                                ax.set_title("{} predicted rsa matrix".format(theory))
                                # Adding the axes titles:
                                ax.set_xlabel('Time (s)')
                                ax.set_ylabel('Time (s)')
                                # Adding the matrices demarcations in case of subsampling:
                                [ax.axhline(ind + 0.5, color='k', linestyle='--')
                                 for ind in matrices_delimitations]
                                [ax.axvline(ind + 0.5, color='k', linestyle='--')
                                 for ind in matrices_delimitations]
                                # Adding axis break to mark the difference:
                                d = 0.01
                                kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                                # Looping through each demarcations to mark them::
                                for ind in matrices_delimitations:
                                    ind_trans = (ind + 1) / len(subsamp_rsa_matrix)
                                    ax.plot((ind_trans - 0.005 - d, ind_trans
                                             - 0.005 + d), (-d, +d), **kwargs)
                                    ax.plot((ind_trans + 0.005 - d, ind_trans
                                             + 0.005 + d), (-d, +d), **kwargs)
                                    ax.plot((-d, +d), (ind_trans - 0.005 - d,
                                                       ind_trans - 0.005 + d), **kwargs)
                                    ax.plot((-d, +d), (ind_trans + 0.005 - d,
                                                       ind_trans + 0.005 + d), **kwargs)
                                # Generate the tick position and labels:
                                ticks_pos = np.linspace(0, theories_matrices[theory].shape[0] - 1, 8)
                                # Generate the tick position and labels:
                                ticks_labels = [str(sub_samp_times[int(ind)]) for ind in ticks_pos]
                                ax.set_xticks(ticks_pos)
                                ax.set_yticks(ticks_pos)
                                ax.set_xticklabels(ticks_labels)
                                ax.set_yticklabels(ticks_labels)
                                plt.tight_layout()
                                # Save the figure:
                                file_name = Path(save_path_fig, param.files_prefix + theory + "_" + time_win +
                                                 "_predicted_matrix.png")
                                plt.savefig(file_name, transparent=True)
                                plt.close()

                        # Plot correlation between the observed matrix and the predicted ones:
                        # Create a data frame to store flatten matrices:
                        obs_n_pred_df = pd.DataFrame({
                            "observed": subsamp_rsa_matrix.flatten(),
                            **{theory: theories_matrices[theory].flatten() for theory in theories_matrices.keys()}
                        })
                        sns.pairplot(obs_n_pred_df, kind='reg')
                        if DO_ZSCORE:
                            file_name = Path(save_path_fig, param.files_prefix + "_" + roi + "_" + time_win + "_"
                                             + pred_pair_id + "_theories_correlations_zscore.png")
                        else:
                            file_name = Path(save_path_fig, param.files_prefix + "_" + roi + "_" + time_win + "_"
                                             + pred_pair_id + "_theories_correlations.png")
                        plt.savefig(file_name, transparent=True)
                        plt.close()
            # Correcting the permutation p values for multiple comparison for each ROI separately:
            if len(stats_results) > 0:
                for roi in stats_results["roi"].unique():
                    for corr_method in stats_results["correlation_method"]:
                        _, stats_results.loc[
                            (stats_results["roi"] == roi) & (stats_results["correlation_method"] == corr_method),
                            "p-value"], _, _ = \
                            multitest.multipletests(stats_results.loc[(stats_results["roi"] == roi) & (
                                        stats_results["correlation_method"] == corr_method), "p-value"].to_list(),
                                                    method=FDR)
            # Save the stats results:
            if DO_ZSCORE:
                stats_results.to_csv(Path(load_path_results, param.files_prefix +
                                          "theories_correlation_results_zscore.csv"))
            else:
                stats_results.to_csv(Path(load_path_results, param.files_prefix +
                                          "theories_correlation_results.csv"))


if __name__ == "__main__":
    # Fetching all the config files:
    configs = find_files(Path(os.getcwd(), "super_subject_config"),
                         naming_pattern="*", extension="json")
    theories_correlations(configs, save_folder="super")
