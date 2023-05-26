"""
This script computes the RSA statistics
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
from pathlib import Path
import matplotlib
from general_helper_functions.data_general_utilities import zscore_mat, cluster_test
from general_helper_functions.plotters import MidpointNormalize
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
gaussian_sig = 4
cmap = 'RdYlBu_r'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi
DO_ZSCORE = True
y_lim = None
FDR = "fdr_bh"


def rsa_super_subject_statistics(configs, save_folder="super"):
    if len(configs) == 0 or configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")
    configs = [file for file in configs if "item_reliability" not in file]
    configs = [file for file in configs if "geometry_reliability" not in file]
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

            for roi_ind, roi in enumerate(param.rois):
                # ======================================================================================================
                # Loading the data:
                # Finding the results files:
                rsa_results = find_files(load_path_results, "*" + roi + "_rsa.", extension="npy")
                rsa_label_shuffle = find_files(load_path_results, "*" + roi + "_rsa_label_shuffle", extension=".npy")
                assert len(rsa_results) == 1, "More than one file was found for rsa results!"
                assert len(rsa_label_shuffle) == 1, "More than one file was found for rsa results!"
                # Loading the results:
                rsa = np.load(rsa_results[0])
                rsa_label_shuffle = np.load(rsa_label_shuffle[0])

                # Averaging the rsa results along the first dimension:
                avg_rsa = np.mean(np.array(rsa), axis=0)

                if analysis_parameters["equate_offset"]:
                    start_n_end = analysis_parameters["epo_onset_offset"]
                else:
                    start_n_end = analysis_parameters["crop_time"]

                # ======================================================================================================
                # RSA statistics:
                # 1. Compare the RSA values distribution against the null distribution
                if analysis_parameters["rsa_stat_test"] == "sliding_histogram_pval":
                    if DO_ZSCORE:
                        avg_rsa = zscore_mat(avg_rsa, rsa_label_shuffle, axis=0)
                        rsa_label_shuffle = [zscore_mat(rsa_label_shuffle[i], rsa_label_shuffle)
                                             for i in range(rsa_label_shuffle.shape[0])]
                    p_values, sig_mask, _ = \
                        label_shuffle_test_2d(avg_rsa, rsa_label_shuffle,
                                              **analysis_parameters["sliding_histogram_pval_param"])
                elif analysis_parameters["rsa_stat_test"] == "cluster_based_test":
                    avg_rsa_zscore, rsa_label_shuffle_zscore, clusters, cluster_pv, p_values, H0 = \
                        cluster_test(avg_rsa, rsa_label_shuffle,
                                     z_threshold=analysis_parameters["cluster_based_test_param"]["z_threshold"],
                                     adjacency=analysis_parameters["cluster_based_test_param"]["adjacency"],
                                     tail=analysis_parameters["cluster_based_test_param"]["tail"],
                                     max_step=analysis_parameters["cluster_based_test_param"]["max_step"],
                                     exclude=analysis_parameters["cluster_based_test_param"]["exclude"],
                                     t_power=analysis_parameters["cluster_based_test_param"]["t_power"],
                                     step_down_p=analysis_parameters["cluster_based_test_param"]["step_down_p"],
                                     do_zscore=analysis_parameters["cluster_based_test_param"]["do_zscore"])
                    # Create the sig_mask:
                    if DO_ZSCORE:
                        # Replace the avg rsa with the zscore one:
                        avg_rsa = avg_rsa_zscore
                        rsa_label_shuffle = rsa_label_shuffle_zscore
                    if p_values.size != 0:
                        # Create the significance mask:
                        sig_mask = avg_rsa.copy()
                        sig_mask[p_values > analysis_parameters["cluster_based_test_param"]["p_value_thresh"]] = np.nan
                    else:
                        sig_mask = np.zeros(avg_rsa.shape)
                        sig_mask[:, :] = np.nan
                else:
                    Exception("You have passed a rsa_stat_test that is not supported! Either sliding_histogram_pval or "
                              "cluster_based_test! Check spelling")
                if y_lim is None:
                    ylim = [np.percentile(avg_rsa, 5), np.percentile(avg_rsa, 95)]
                else:
                    ylim = y_lim
                norm = matplotlib.colors.TwoSlopeNorm(vmin=ylim[0], vcenter=0, vmax=ylim[1])
                # Save the pvalues:
                file_name = Path(load_path_results, param.files_prefix + roi + "_rsa_p_values.npy")
                if p_values.shape == 0:
                    p_values = np.zeros(avg_rsa.shape) + 1
                np.save(file_name, p_values)
                # 1.2. Plot the rsa temporal generalization matrices:
                fig, ax = plt.subplots(figsize=[20, 15])
                im = ax.imshow(avg_rsa, origin="lower", cmap=cmap, aspect="equal",
                               norm=norm, extent=[start_n_end[0], start_n_end[-1], start_n_end[0], start_n_end[-1]])
                ax.set_title("{} RSA, ROI: {}".format(analysis_parameters["rsa_condition"], roi,
                                                      ))
                ax.axvline(0, color='k')
                ax.axhline(0, color='k')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Time (s)')
                plt.tight_layout()
                cbar = plt.colorbar(im, ax=ax)
                if DO_ZSCORE:
                    cbar.set_label('Z score')
                else:
                    cbar.set_label('Correlation difference within vs between')
                cbar.ax.yaxis.set_label_position('left')
                if DO_ZSCORE:
                    plt.savefig(Path(save_path_fig, param.files_prefix +
                                     "rsa_matrix_zscore" + roi + ".png"))
                else:
                    plt.savefig(Path(save_path_fig, param.files_prefix +
                                     "rsa_matrix_" + roi + ".png"))
                plt.close()

                # ======================================================================
                # 1.3. Plot the rsa temporal generalization matrices with significance:
                # Doing the same but plotting the significance mask on top:
                fig, ax = plt.subplots(figsize=[20, 15])
                # Plot matrix with transparency:
                im = ax.imshow(avg_rsa, cmap=cmap, norm=norm,
                               extent=[start_n_end[0], start_n_end[-1], start_n_end[0], start_n_end[-1]],
                               origin="lower", alpha=0.4, aspect='equal')
                # Plot the significance mask on top:
                if not np.isnan(sig_mask).all():
                    # Plot only the significant bits:
                    im = ax.imshow(sig_mask, cmap=cmap, origin='lower', norm=norm,
                                   extent=[start_n_end[0], start_n_end[-1], start_n_end[0], start_n_end[-1]],
                                   aspect='equal')
                # Add the axis labels and so on:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Time (s)')
                ax.set_title("{} RSA, ROI: {}".format(analysis_parameters["rsa_condition"], roi,
                                                      ))
                ax.axvline(0, color='k')
                ax.axhline(0, color='k')
                plt.tight_layout()
                cb = plt.colorbar(im)
                if DO_ZSCORE:
                    cb.ax.set_ylabel('Z score')
                else:
                    cb.ax.set_ylabel('Correlation difference within vs between')
                cb.ax.yaxis.set_label_position('left')
                # Finally, adding the significance contour:
                if not np.isnan(sig_mask).all():
                    ax.contour(sig_mask > 0, sig_mask > 0, colors="k", origin="lower",
                               extent=[start_n_end[0], start_n_end[-1], start_n_end[0], start_n_end[-1]])
                # Save the figure:
                if DO_ZSCORE:
                    file_name = Path(save_path_fig, param.files_prefix + "sig" + "_" + roi + "_zscore_rsa.png")
                else:
                    file_name = Path(save_path_fig, param.files_prefix + "sig" + "_" + roi + "_rsa.png")
                plt.savefig(file_name, transparent=True)
                plt.close()


if __name__ == "__main__":
    # Fetching all the config files:
    configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")
    rsa_super_subject_statistics(configs, save_folder="super")
