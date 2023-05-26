"""
This script tests for activation in time window following stimulus onset and offset
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from pathlib import Path
from Experiment1ActivationAnalysis.activation_analysis_helper_function import *
from general_helper_functions.data_general_utilities import load_epochs

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
from Experiment1ActivationAnalysis.plot_onset_offset_results import plotting_onset_offset_results
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def onset_offset_super_subject(subjects_list, save_folder="super"):
    """
    This function performs the onset offset analysis, i.e. comparing the onset period and the offset period against
    a baseline to see if there are any differences to investigate GNW predictions.
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
        configs = find_files(Path(os.getcwd(), "onset_offset_config"), naming_pattern="*", extension=".json")
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
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True, stats=False)
            param.save_parameters(save_path_fig)
            param.save_parameters(save_path_results)

            # Save the participants list to these directories too:
            with open(Path(save_path_fig, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_results, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")

            for roi in param.rois:
                data_df = {condition: [] for condition in analysis_parameters["conditions"]}
                mni_coords = []
                # Loading in parallel the data of each subject:
                for subject in subjects_list:
                    epochs, mni_coord = load_epochs(param.BIDS_root, analysis_parameters["signal"],
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
                                                    aseg=param.aseg,
                                                    montage_space=param.montage_space,
                                                    get_mni_coord=True,
                                                    picks_roi=param.rois[roi],
                                                    select_vis_resp=analysis_parameters["select_vis_resp"],
                                                    vis_resp_folder=analysis_parameters["vis_resp_folder"],
                                                    filtering_parameters=
                                                    analysis_parameters["multitaper_parameters"]
                                                    )
                    if epochs is None:
                        continue
                    mni_coords.append(mni_coord)
                    # Otherwise, format the data for the test:
                    for condition in analysis_parameters["conditions"]:
                        data_df[condition].append(format_tim_win_comp_data(epochs.copy()[condition], subject,
                                                                           analysis_parameters["baseline_window"],
                                                                           analysis_parameters["test_window"]))

                # Looping through the different conditions:
                results = []
                for condition in analysis_parameters["conditions"]:
                    # Convert the list to a single data frame:
                    data_df[condition] = pd.concat(data_df[condition], axis=0, ignore_index=True)
                    # Performing the moving window test:
                    test_results = moving_window_test(data_df[condition], analysis_parameters["test_window"][0],
                                                      groups="channel", thresh=analysis_parameters["alpha"],
                                                      dur_thresh=analysis_parameters["dur_thresh"],
                                                      alternative=analysis_parameters["alternative"],
                                                      sfreq=512, fdr_method=analysis_parameters["fdr_method"],
                                                      stat_test=analysis_parameters["stat_test"])
                    test_results["condition"] = condition
                    results.append(test_results)

                # Now, compiling the results into one table:
                results = pd.concat(results, ignore_index=True)
                results_reformat = pd.DataFrame()
                # Handling the results table to account for electrodes being significant in both conditions:
                for channel in results["channel"].unique():
                    # Get the data of this unique channel:
                    ch_results = results.loc[results["channel"] == channel]
                    # Getting the significant ones:
                    ch_sig_results = ch_results.loc[ch_results["reject"] == True]
                    if len(ch_sig_results) == 0:
                        results_reformat = results_reformat.append(pd.DataFrame({
                            "subject": ch_results["subject"].to_list()[0],
                            "channel": channel,
                            "metric": ch_results["metric"].to_list()[0],
                            "reject": False,
                            "stat": [ch_results["stat"].to_list()],
                            "pval": [ch_results["pval"].to_list()],
                            "onset": None,
                            "offset": None,
                            "condition": "no_diff"
                        }, index=[0]))
                    elif len(ch_sig_results) == 1:
                        results_reformat = results_reformat.append(pd.DataFrame({
                            "subject": ch_results["subject"].to_list()[0],
                            "channel": channel,
                            "metric": ch_results["metric"].to_list()[0],
                            "reject": True,
                            "stat": ch_sig_results["stat"].to_list()[0],
                            "pval": ch_sig_results["pval"].to_list()[0],
                            "onset": ch_sig_results["onset"].to_list()[0],
                            "offset": ch_sig_results["offset"].to_list()[0],
                            "condition": ch_sig_results["condition"].to_list()[0]
                        }, index=[0]))
                    elif len(ch_results) == 2:
                        # Compute onset and effect sizes as being the average:
                        onset = np.mean(ch_results["onset"])
                        offset = np.mean(ch_results["offset"])
                        results_reformat = results_reformat.append(pd.DataFrame({
                            "subject": ch_results["subject"].to_list()[0],
                            "channel": channel,
                            "metric": ch_results["metric"].to_list()[0],
                            "reject": True,
                            "stat": [ch_results["stat"].to_list()],
                            "pval": [ch_results["pval"].to_list()],
                            "onset": onset,
                            "offset": offset,
                            "condition": "both"
                        }, index=[0]))
                    else:
                        raise Exception("Something went wrong!")

                # Dropping the index:
                results_reformat = results_reformat.reset_index(drop=True)
                # Save to file:
                file_name = param.files_prefix + roi + "_onset_offset_results.csv"
                results_reformat.to_csv(Path(save_path_results, file_name), index=False)

                # Generate a summary table:
                summary_table = pd.DataFrame()
                cond_list = list(results_reformat["condition"].unique())
                for subject in results_reformat["subject"].unique():
                    # Extracting the data from this specific subject from the results table:
                    sub_results = results_reformat.loc[results_reformat["subject"] == subject]
                    sub_results_sig = sub_results.loc[sub_results["reject"] == True]
                    count_per_cond = {cond: len(sub_results.loc[sub_results["condition"] == cond])
                                      for cond in cond_list if cond is not None}
                    summary_table = summary_table.append({
                        "subject": subject,
                        "signal": analysis_parameters["signal"],
                        "percent-sig": len(sub_results_sig) / len(sub_results),
                        "n-sig-electrodes": len(sub_results_sig),
                        "total-n-electrodes": len(sub_results),
                        **count_per_cond
                    }, ignore_index=True)
                file_name = param.files_prefix + roi + "_onset_offset_summary.csv"
                summary_table.to_csv(Path(save_path_results, file_name), index=False)

                # Finally, generate and save the channels info:
                mni_coords = pd.concat(mni_coords, ignore_index=True)
                channels_info = pd.DataFrame()
                for channel in list(results_reformat["channel"].unique()):
                    # Get the results of this channel:
                    ch_results = results_reformat.loc[results_reformat["channel"] == channel]
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
                        "condition": ch_results["condition"].item()
                    }, index=[0]))
                channels_info_file = param.files_prefix + roi + "channels_info.csv"
                channels_info.to_csv(Path(save_path_fig, channels_info_file))

    print("Onset offset computations done!")
    print("Plotting the results")
    plotting_onset_offset_results(configs, save_folder=save_folder)


if __name__ == "__main__":
    onset_offset_super_subject(None, save_folder="super")
