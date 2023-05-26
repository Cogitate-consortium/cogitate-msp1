"""
This script computes the rsa on the super subject
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from general_helper_functions.data_general_utilities import load_epochs
from rsa.rsa_parameters_class import RsaParameters
from rsa.rsa_super_subject_statistics import rsa_super_subject_statistics
from rsa.theories_correlations import theories_correlations
from rsa.rsa_helper_functions import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VERBOSE = False


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
                rsa_results, rdm_diag, first_pres_labels, second_pres_labels, selected_channels, sel_features, \
                split_half_rdm = \
                    zip(*Parallel(n_jobs=param.njobs)(delayed(compute_super_subject_rsa)(
                        sub_epochs, analysis_parameters["rsa_condition"],
                        groups_condition=analysis_parameters["groups_condition"],
                        equalize_trials=analysis_parameters["equalize_trials"],
                        binning_ms=analysis_parameters["binning_ms"],
                        method=analysis_parameters["method"], shuffle_labels=False,
                        zscore=analysis_parameters["zscore"],
                        min_per_label=analysis_parameters["n_repeat"],
                        n_features=analysis_parameters["n_features"],
                        n_folds=analysis_parameters["n_folds"],
                        verbose=VERBOSE,
                        feat_sel_diag=analysis_parameters["feat_sel_diag"],
                        store_intermediate=analysis_parameters["store_intermediate"],
                        metric=analysis_parameters["metric"]
                    )
                                                      for i in tqdm(range(analysis_parameters["n_resampling"]))))
                # Compute RSA but shuffling the labels to generate a null distribution:
                rsa_label_shuffle, _, _, _, _, _, _ = zip(*Parallel(n_jobs=param.njobs)(
                    delayed(compute_super_subject_rsa)(
                        sub_epochs, analysis_parameters["rsa_condition"],
                        groups_condition=analysis_parameters["groups_condition"],
                        equalize_trials=analysis_parameters["equalize_trials"],
                        binning_ms=analysis_parameters["binning_ms"],
                        method=analysis_parameters["method"], shuffle_labels=True,
                        zscore=analysis_parameters["zscore"],
                        min_per_label=analysis_parameters["n_repeat"],
                        n_features=analysis_parameters["n_features"],
                        n_folds=analysis_parameters["n_folds"],
                        verbose=VERBOSE,
                        feat_sel_diag=analysis_parameters["feat_sel_diag"]
                    )
                    for i in tqdm(range(analysis_parameters["n_perm"]))))

                # ======================================================================================================
                # Save the results:
                # Convert the the rsa to a np array:
                rsa_results = np.asarray(rsa_results)
                # Saving the cross_temporal_rsa to file:
                file_name = Path(save_path_results, param.files_prefix + roi + "_rsa.npy")
                np.save(file_name, rsa_results)

                # Same for the label shuffle results:
                rsa_label_shuffle = np.asarray(rsa_label_shuffle)
                # Saving the cross_temporal_rsa to file:
                file_name = Path(save_path_results, param.files_prefix + roi + "_rsa_label_shuffle.npy")
                np.save(file_name, rsa_label_shuffle)

                # Saving the sample RDM:
                rdm_diag = np.asarray(rdm_diag)
                file_name = Path(save_path_results, param.files_prefix + roi + "_sample_rdm.npy")
                np.save(file_name, rdm_diag)

                # Saving the first and second presentation labels to be able to track things down:
                first_pres_labels = np.asarray(first_pres_labels)
                file_name = Path(save_path_results, param.files_prefix + roi + "_first_pres_labels.npy")
                np.save(file_name, first_pres_labels)
                if second_pres_labels[0] is not None:
                    second_pres_labels = np.asarray(second_pres_labels)
                    file_name = Path(save_path_results, param.files_prefix + roi + "_second_pres_labels.npy")
                    np.save(file_name, second_pres_labels)

                # Save the features:
                if analysis_parameters["n_features"] is not None and analysis_parameters["feat_sel_diag"]:
                    try:
                        sel_features = np.array(sel_features)
                        file_name = Path(save_path_results, param.files_prefix + roi + "_features.npy")
                        np.save(file_name, sel_features)
                    except:
                        print("No Features to save!")

                if split_half_rdm is not None:
                    rdm_diag = np.asarray(split_half_rdm)
                    file_name = Path(save_path_results, param.files_prefix + roi + "_split_half_rdm.npy")
                    np.save(file_name, rdm_diag)


                # Saving the channels info:
                selected_channels = np.unique(np.concatenate(selected_channels, axis=0))
                mni_coords = pd.concat([sub_mni_coords[key] for key in sub_mni_coords.keys()])
                channels_info = mni_coords.loc[mni_coords["channels"].isin(selected_channels)]
                file_name = Path(save_path_results, param.files_prefix + roi + "_channels_info.csv")
                channels_info.to_csv(file_name)

    print("RSA was successfully computed for all the subjects!")
    print("Now computing the statistics on all the results")
    rsa_super_subject_statistics(configs, save_folder=save_folder)
    theories_correlations(configs, save_folder=save_folder)

    return None


if __name__ == "__main__":
    super_subject_rsa(None, save_folder="super")
