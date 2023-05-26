"""
This script attempts to detect channels showing sustained activation consistent with the stimulus duration
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from pathlib import Path
from scipy import stats
from joblib import Parallel, delayed
from Experiment1ActivationAnalysis.activation_analysis_helper_function import *
from general_helper_functions.data_general_utilities import load_epochs

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
from Experiment1ActivationAnalysis.plot_duration_tracking_results import plotting_duration_tracking_results
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def duration_tracking_analysis(subjects_list, save_folder="super"):
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
        configs = find_files(Path(os.getcwd(), "duration_tracking_config"), naming_pattern="*", extension=".json")
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
            for roi in param.rois:
                # Prepare the storage of the results:
                results = []
                tracking_proportion = []
                results_shuffle = []
                tracking_proportion_shuffle = []
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
                    # Smooth the data if needed:
                    if analysis_parameters["moving_average_ms"] is not None:
                        epochs = epochs_mvavg(epochs, analysis_parameters["moving_average_ms"])
                    # Reset the metadata indices:
                    trials_metadata = epochs.metadata.reset_index(drop=True)

                    # ======================================================================
                    # Compute the duration tracking:
                    # Parallelize electrodes:
                    subject_results, subject_tracking_proportion = zip(*Parallel(n_jobs=param.n_jobs)(
                        delayed(compute_tracking_inaccuracy)(
                            epochs, subject, channel, trials_metadata,
                            threshold_condition=analysis_parameters[
                                "threshold_condition"],
                            baseline_time=analysis_parameters["baseline"],
                            activation_time=analysis_parameters["activation"],
                            shuffle_label=False,
                            inaccuracy_threshold=
                            analysis_parameters["inaccuracy_threshold"],
                            fast_track=analysis_parameters["fast_track"]
                        )
                        for channel in epochs.ch_names))
                    # Append to the results:
                    results.append(pd.concat(subject_results, ignore_index=True))
                    tracking_proportion.append(pd.concat(subject_tracking_proportion, ignore_index=True))

                    # Now computing the permutations:
                    for channel in epochs.ch_names:
                        subject_results_shuffle, subject_tracking_proportion_shuffle = \
                            zip(*Parallel(n_jobs=param.n_jobs)(
                                delayed(compute_tracking_inaccuracy)(
                                    epochs, subject, channel, trials_metadata,
                                    threshold_condition=analysis_parameters[
                                        "threshold_condition"],
                                    baseline_time=analysis_parameters["baseline"],
                                    activation_time=analysis_parameters["activation"],
                                    shuffle_label=True,
                                    inaccuracy_threshold=
                                    analysis_parameters["inaccuracy_threshold"],
                                    fast_track=analysis_parameters["fast_track"]
                                )
                                for i in range(analysis_parameters["nperm"])))
                        results_shuffle.append(pd.concat(subject_results_shuffle, ignore_index=True))
                        tracking_proportion_shuffle.append(pd.concat(subject_tracking_proportion_shuffle,
                                                                     ignore_index=True))

                # Combine the results into one data frame:
                results = pd.concat(results, ignore_index=True)
                tracking_proportion = pd.concat(tracking_proportion, ignore_index=True)
                results_shuffle = pd.concat(results_shuffle, ignore_index=True)
                tracking_proportion_shuffle = pd.concat(tracking_proportion_shuffle, ignore_index=True)

                # ======================================================================
                # Compute the statistics:
                stats_results = pd.DataFrame()
                for channel in tracking_proportion["channel"].unique():
                    # Get the results of this one channel
                    observed_value = tracking_proportion.loc[tracking_proportion["channel"] ==
                                                             channel, "tracking_proportion"].to_numpy()
                    null_distribution = tracking_proportion_shuffle.loc[
                        tracking_proportion_shuffle["channel"] ==
                        channel, "tracking_proportion"].to_numpy()
                    # Locating on the null distribution
                    prctile = stats.percentileofscore(np.concatenate([null_distribution, observed_value]),
                                                      observed_value) / 100
                    # We are only interested into 1 tailed test, so converting the percentile into p value:
                    stats_results = stats_results.append(pd.DataFrame({
                        "subject": channel.split("-")[0],
                        "channel": channel,
                        "tracking_accuracy": observed_value,
                        "p-value": 1 - prctile
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
                # Save the "data", i.e. what we preprocessing the stats on:
                file_name = Path(save_path_data, param.files_prefix + roi + "_single_trial_duration_tracking.csv")
                results.to_csv(file_name)
                file_name = Path(save_path_data, param.files_prefix + roi + "_tracking_proportion.csv")
                tracking_proportion.to_csv(file_name)
                # Save the results, i.e. the stats:
                file_name = Path(save_path_results, param.files_prefix + roi + "_tracking_proportion_stats.csv")
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
                        "tracking_accuracy": ch_results["tracking_accuracy"],
                        "reject": ch_results["reject"]
                    }, index=[0]))
                channels_info_file = param.files_prefix + roi + "channels_info.csv"
                channels_info.to_csv(Path(save_path_fig, channels_info_file))

    print("Plotting the results")
    plotting_duration_tracking_results(configs, save_folder=save_folder)


if __name__ == "__main__":
    duration_tracking_analysis(None, save_folder="super")
