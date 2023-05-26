"""
This script computes the RSA by subsampling N electrodes, to make sure that the results are not driven from a single
subject
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

from general_helper_functions.plotters import MidpointNormalize
from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from general_helper_functions.data_general_utilities import load_epochs
from rsa.rsa_parameters_class import RsaParameters
from rsa.rsa_helper_functions import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
N_REPEATS = 100
N_CHANNELS = 200
VERBOSE = False
cmap = "RdYlBu_r"


def super_subject_rsa(subjects_list, save_folder="super"):
    # Extract config from command line argument
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    # If no config was passed, just using them all
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = \
            RsaParameters(config, sub_id=save_folder)
        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "rsa")
        # Looping through the different analysis performed in the visual responsiveness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create an evoked object to append to:
            save_path_data = path_generator(param.save_root,
                                            analysis=analysis_name,
                                            preprocessing_steps=param.preprocess_steps,
                                            fig=False, stats=False, data=True)
            save_path_results = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=False, stats=True)
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)
            param.save_parameters(save_path_results)
            param.save_parameters(save_path_data)

            # Save the participants list to these directories too:
            with open(Path(save_path_fig, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_results, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_data, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")

            # ======================================================================================================
            # Loading the data:
            # Looping through each ROI:
            for roi in param.rois:
                print("Compute RSA in ROI {}".format(roi))
                sub_epochs = {}
                sub_mni_coords = {}
                # Loading the data of each subject:
                for subject in subjects_list:
                    sub_epochs[subject], sub_mni_coords[subject] = \
                        load_epochs(param.BIDS_root, analysis_parameters["signal"],
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
                                    select_vis_resp=False,
                                    vis_resp_folder=None,
                                    aseg=param.aseg,
                                    montage_space=param.montage_space,
                                    get_mni_coord=True,
                                    picks_roi=param.rois[roi],
                                    filtering_parameters=None)
                    if sub_epochs[subject] is None:
                        del sub_epochs[subject]
                if len(sub_epochs) == 0:
                    print("WARNING: There were no channels in ROI {}. \nIt will be skipped".format(roi))
                    continue
                # ======================================================================================================
                # Preprocessing:
                # Perform preprocessing in each:
                for subject in sub_epochs.keys():
                    # Subtract the evoked response:
                    if analysis_parameters["subtract_evoked"]:
                        sub_epochs[subject].subtract_evoked()
                    if analysis_parameters["regress_evoked"]:
                        sub_epochs[subject] = regress_evoked(sub_epochs[subject])
                    if analysis_parameters["equate_offset"]:
                        sub_epochs[subject] = equate_offset(sub_epochs[subject],
                                                            analysis_parameters["equate_offset_dict"])

                # ======================================================================================================
                # Compute RSA:
                # Compute the RSA with parallelization:
                rsa_results, sample_rdms, selected_channels = \
                    zip(*Parallel(n_jobs=param.njobs)(delayed(compute_super_subject_rsa)(
                        sub_epochs, analysis_parameters["rsa_condition"],
                        groups_condition=analysis_parameters["groups_condition"],
                        equalize_trials=analysis_parameters["equalize_trials"],
                        binning_ms=analysis_parameters["binning_ms"],
                        method=analysis_parameters["method"], shuffle_labels=False,
                        zscore=analysis_parameters["zscore"],
                        regress_groups=analysis_parameters["regress_groups"],
                        between_within_group=analysis_parameters["between_within_group"],
                        sample_rdm_times=analysis_parameters["sample_rdm_times"],
                        min_per_label=analysis_parameters["n_repeat"],
                        n_features=analysis_parameters["n_features"],
                        n_folds=analysis_parameters["n_folds"],
                        verbose=VERBOSE
                    )
                    for i in tqdm(range(N_REPEATS))))

                # ======================================================================================================
                # Save the results:
                # Convert the the rsa to a np array:
                rsa_results = np.asarray(rsa_results)
                # Saving the cross_temporal_rsa to file:
                file_name = Path(save_path_results, param.files_prefix + roi + "_rsa.npy")
                np.save(file_name, rsa_results)

                # Saving the sample RDM:
                sample_rdms = np.asarray(sample_rdms)
                file_name = Path(save_path_results, param.files_prefix + roi + "_sample_rdm.npy")
                np.save(file_name, sample_rdms)

                # ======================================================================================================
                # Plot the results:
                # Set the time correctly:
                if analysis_parameters["equate_offset"]:
                    start_n_end = analysis_parameters["epo_onset_offset"]
                else:
                    start_n_end = analysis_parameters["crop_time"]
                # Compute the average across the repeats:
                avg_rsa = np.mean(np.array(rsa_results), axis=0)
                ylim = [np.percentile(avg_rsa, 5), np.percentile(avg_rsa, 95)]
                norm = MidpointNormalize(vmin=ylim[0], vmax=ylim[1], midpoint=0)
                # 1.2. Plot the rsa temporal generalization matrices:
                fig, ax = plt.subplots(figsize=[20, 15])
                im = ax.imshow(avg_rsa, origin="lower", cmap=cmap, aspect="equal",
                               norm=norm, extent=[start_n_end[0], start_n_end[-1], start_n_end[0],
                                                  start_n_end[-1]])
                ax.set_title("{} RSA, ROI: {}".format(analysis_parameters["rsa_condition"], roi,
                                                      ))
                ax.axvline(0, color='k')
                ax.axhline(0, color='k')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Time (s)')
                plt.tight_layout()

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation difference within vs between')
                cbar.ax.yaxis.set_label_position('left')
                plt.savefig(Path(save_path_fig, param.files_prefix +
                                 "rsa_matrix_" + roi + ".png"))
                plt.close()

    return None


if __name__ == "__main__":
    super_subject_rsa(None, save_folder="super_robustness")
