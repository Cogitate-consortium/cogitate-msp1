"""
This script performs decoding of stimulus duration within single electrode
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from mne.stats.cluster_level import _pval_from_histogram
from Experiment1ActivationAnalysis.activation_analysis_helper_function import *
from general_helper_functions.data_general_utilities import load_epochs

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
from Experiment1ActivationAnalysis.plot_duration_decoding_results import plotting_duration_decoding_results
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def duration_decoding_analysis(subjects_list, save_folder="super"):
    """
    This function performs the duration tracking analysis, replicating the analysis from Gerber et al. 2017
    (https://www.sciencedirect.com/science/article/pii/S1053811917306754) to investigate IIT predictions.
    :param subjects_list: (list of strings) list of the subjects on whom to run the analysis
    :return:
    """
    # ==================================================================================================================
    # Extract config from command line argument
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    # If no config was passed, just using them all
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "duration_decoding_config"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = DurationAnalysisParameters(config, sub_id=save_folder)
        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "activation_analysis")
        # Looping through the different analysis performed in the visual responsiveness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create an evoked object to append to:
            save_path_results = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=False, stats=True)
            save_path_data = path_generator(param.save_root,
                                            analysis=analysis_name,
                                            preprocessing_steps=param.preprocess_steps,
                                            fig=False, data=True)
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

            # Looping through each ROI:
            for roi in param.rois:
                # Prepare the storage of the results:
                decoding_scores = []
                decoding_shuffle = []
                mni_coords = []
                # Loading in parallel the data of each subject:
                for subject in subjects_list:
                    # Loading the data:
                    epochs, mni_coord = \
                        load_epochs(param.BIDS_root, analysis_parameters["signal"],
                                    subject, session=param.session, task_name=param.task_name,
                                    preprocess_folder=param.preprocessing_folder,
                                    preprocess_steps=param.preprocess_steps,
                                    channel_types={"seeg": True, "ecog": True},
                                    condition=analysis_parameters["conditions"],
                                    baseline_method=analysis_parameters["baseline_correction"],
                                    baseline_time=analysis_parameters["baseline_time"],
                                    crop_time=analysis_parameters["crop_time"],
                                    aseg=param.aseg,
                                    montage_space=param.montage_space,
                                    get_mni_coord=True,
                                    picks_roi=param.rois[roi],
                                    select_vis_resp=analysis_parameters["select_vis_resp"],
                                    vis_resp_folder=analysis_parameters["vis_resp_folder"],
                                    filtering_parameters=
                                    analysis_parameters["multitaper_parameters"])
                    if epochs is None:
                        print("There were no electrodes for sub-{} in {} roi".format(subject, roi))
                        continue
                    mni_coords.append(mni_coord)
                    # Reset the metadata indices:
                    trials_metadata = epochs.metadata.reset_index(drop=True)

                    # ======================================================================
                    # Compute the duration decoding, parallelized across channels:
                    subject_decoding_scores = Parallel(n_jobs=param.n_jobs)(
                        delayed(duration_decoding)(
                            epochs, channel, trials_metadata,
                            shuffle_label=False,
                            n_folds=analysis_parameters[
                                "n_folds"],
                            time_win=analysis_parameters["time_win"],
                            classifier=analysis_parameters["classifier"],
                            binning_ms=analysis_parameters["binning_ms"],
                            do_diff=analysis_parameters["do_diff"]
                        )
                        for channel in epochs.ch_names)
                    # Append to the results:
                    decoding_scores.append(pd.concat(subject_decoding_scores, ignore_index=True))

                    # Now computing the permutations:
                    # Looping through each channel:
                    for channel in epochs.ch_names:
                        subject_decoding_shuffle = Parallel(n_jobs=param.n_jobs)(
                            delayed(duration_decoding)(
                                epochs, channel, trials_metadata,
                                labels_condition=analysis_parameters[
                                    "labels_condition"],
                                shuffle_label=True,
                                n_folds=analysis_parameters[
                                    "n_folds"],
                                time_win=analysis_parameters["time_win"],
                                classifier=analysis_parameters["classifier"],
                                binning_ms=analysis_parameters["binning_ms"],
                                do_diff=analysis_parameters["do_diff"]
                            )
                            for i in range(analysis_parameters["nperm"]))
                        decoding_shuffle.append(pd.concat(subject_decoding_shuffle, ignore_index=True))

                # Combine the results into one data frame:
                decoding_scores = pd.concat(decoding_scores, ignore_index=True)
                decoding_shuffle = pd.concat(decoding_shuffle, ignore_index=True)

                # ======================================================================
                # Compute the statistics:
                stats_results = pd.DataFrame()
                for channel in decoding_scores["channel"].unique():
                    # Get the results of this one channel
                    observed_value = decoding_scores.loc[decoding_scores["channel"] == channel,
                                                         "decoding_accuracy"].to_numpy()
                    null_distribution = decoding_shuffle.loc[
                        decoding_shuffle["channel"] ==
                        channel, "decoding_accuracy"].to_numpy()
                    null_distribution = np.concatenate([null_distribution, observed_value])
                    # Compute the p value:
                    p_val = _pval_from_histogram(observed_value, null_distribution, analysis_parameters["tail"])
                    # We are only interested into 1 tailed test, so converting the percentile into p value:
                    stats_results = stats_results.append(pd.DataFrame({
                        "subject": channel.split("-")[0],
                        "channel": channel,
                        "decoding_score": observed_value,
                        "p-value": p_val
                    }))
                # Performing FDR if needed:
                if analysis_parameters["fdr_method"] is not None:
                    stats_results["reject"], stats_results["p-value"], _, _ = \
                        multipletests(stats_results["p-value"].to_numpy(), alpha=analysis_parameters["alpha"],
                                      method=analysis_parameters["fdr_method"])
                else:
                    stats_results["reject"] = stats_results["p-value"].to_numpy() < analysis_parameters["alpha"]

                # ======================================================================
                # Save the results:
                file_name = Path(save_path_results, param.files_prefix + roi + "_duration_decoding_accuracy_stats.csv")
                stats_results.to_csv(file_name)

                # Finally, generate and save the channels info:
                mni_coords = pd.concat(mni_coords, ignore_index=True)
                channels_info = pd.DataFrame()
                for channel in list(stats_results["channel"].unique()):
                    # Get the results of this channel:
                    ch_results = stats_results.loc[stats_results["channel"] == channel]
                    # Look for the x y and z coordinates of that channel:
                    coords = np.squeeze(mni_coords.loc[mni_coords["channels"] == channel, ["x", "y", "z"]].to_numpy())
                    ch_type = mni_coords.loc[mni_coords["channels"] == channel, "ch_types"].item()
                    # Append to the data frame:
                    channels_info = channels_info.append(pd.DataFrame({
                        "subject": channel.split("-")[0],
                        "channel": channel,
                        "ch_types": ch_type,
                        "x": coords[0],
                        "y": coords[1],
                        "z": coords[2],
                        "decoding_score": ch_results["decoding_score"],
                        "reject": ch_results["reject"]
                    }, index=[0]))
                channels_info_file = param.files_prefix + roi + "channels_info.csv"
                channels_info.to_csv(Path(save_path_fig, channels_info_file))

    print("Plotting the results")
    plotting_duration_decoding_results(configs, save_folder=save_folder)


if __name__ == "__main__":
    duration_decoding_analysis(None, save_folder="super")
