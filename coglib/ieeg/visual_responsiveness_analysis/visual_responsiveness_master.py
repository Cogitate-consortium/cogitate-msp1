""" This script identifies channels showing responsiveness to the experimental task.
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Katarina Bendtz, Simon Henin
    katarina.bendtz@tch.harvard.edu
    Simon.Henin@nyulangone.org
"""
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from visual_responsiveness_analysis.visual_responsiveness_helper_functions import *
from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from visual_responsiveness_analysis.visual_responsivness_parameters_class import VisualResponsivnessAnalysisParameters
from visual_responsiveness_analysis.plot_visual_responsiveness_results import plot_single_electrodes, plot_roi_evoked
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def visual_responsiveness(subjects_list, save_folder="super"):
    # ==================================================================================================================
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
            VisualResponsivnessAnalysisParameters(
                config, sub_id=save_folder)
        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "visual_responsiveness")
        # Looping through the different analysis performed in the visual responsivness:
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # Create an evoked object to append to:
            save_path_results = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=False, stats=True)
            save_path_data = path_generator(param.save_root,
                                            analysis=analysis_name,
                                            preprocessing_steps=param.preprocess_steps,
                                            fig=False, stats=False, data=True)
            save_path_figures = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=True, stats=False, data=False)
            param.save_parameters(save_path_data)
            param.save_parameters(save_path_results)
            results = []

            # Save the participants list to these directories too:
            with open(Path(save_path_results, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_data, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            # ==========================================================================================================
            # Single condition:
            # If only on condition is supposed to be investigated:
            if len(analysis_parameters["conditions"]) == 1:
                # Loading in parallel the data of each subject:
                data, sfreqs, mni_coords = zip(*Parallel(n_jobs=param.njobs)(
                    delayed(prepare_test_data)(
                        param.BIDS_root, analysis_parameters["signal"],
                        analysis_parameters["baseline_correction"],
                        analysis_parameters["baseline_window"],
                        analysis_parameters["test_window"],
                        analysis_parameters[analysis_parameters["test"]][
                            "metric"],
                        analysis_parameters["test"],
                        subject,
                        baseline_time=analysis_parameters["baseline_time"],
                        crop_time=analysis_parameters["crop_time"],
                        condition=analysis_parameters["conditions"],
                        session=param.session,
                        task_name=param.task_name,
                        preprocess_folder=param.preprocessing_folder,
                        preprocess_steps=param.preprocess_steps,
                        channel_types={"seeg": True, "ecog": True},
                        get_mni_coord=True,
                        montage_space="T1",
                        picks_rois=None,
                        multitaper_parameters=analysis_parameters["multitaper_parameters"],
                        aseg=param.aseg
                    ) for subject in subjects_list))
                # Make sure that all subjects had the same sampling frequency:
                if len(set(sfreqs)) > 1:
                    raise Exception("The subjects have different sampling rates! That doesn't work!")
                else:
                    sfreq = list(set(sfreqs))[0]
                # Concatenating all the data together:
                data = pd.concat(data, ignore_index=True)
                mni_coords = pd.concat(mni_coords, ignore_index=True)
                # Saving the data to file:
                data.to_csv(Path(save_path_data, param.files_prefix + "data.csv"))
                # Performing the expected test:
                if analysis_parameters["test"] == "t_test" \
                        or analysis_parameters["test"] == "wilcoxon_signed_rank" \
                        or analysis_parameters["test"] == "wilcoxon_rank_sum" \
                        or analysis_parameters["test"] == "bayes_t_test":
                    # Performing the test:
                    results = \
                        aggregated_stat_test(data, analysis_parameters["test_window"],
                                             groups="channel",
                                             test=analysis_parameters["test"],
                                             p_val=analysis_parameters[analysis_parameters["test"]]["p_val"],
                                             alternative=
                                             analysis_parameters[analysis_parameters["test"]]["alternative"])
                    results["condition"] = analysis_parameters["conditions"][0]
                elif analysis_parameters["test"] == "sustained_zscore":
                    results = sustained_zscore_test(data, analysis_parameters["test_window"][0],
                                                    groups="channel",
                                                    z_thresh=analysis_parameters[analysis_parameters["test"]][
                                                        "z_thresh"],
                                                    dur_thresh=
                                                    analysis_parameters[analysis_parameters["test"]][
                                                        "dur_thresh"],
                                                    alternative=
                                                    analysis_parameters[analysis_parameters["test"]][
                                                        "alternative"],
                                                    sfreq=sfreq)
                    results["condition"] = analysis_parameters["conditions"][0]
                elif analysis_parameters["test"] == "cluster_based":
                    # Perform the statistical test:
                    results = \
                        cluster_based_permutation_test(data,
                                                       sfreq,
                                                       groups="channel",
                                                       onset=analysis_parameters["test_window"][0],
                                                       n_perm=analysis_parameters[analysis_parameters["test"]][
                                                           "n_perm"],
                                                       p_val=analysis_parameters[analysis_parameters["test"]][
                                                           "p_val"])
                    results["condition"] = analysis_parameters["conditions"][0]
                else:
                    raise Exception("ERROR: YOU HAVE PASSED A TEST THAT IS NOT SUPPORTED")

                # Perform FDR if required:
                if analysis_parameters["fdr_correction"] is not None and \
                        analysis_parameters["test"] != "sustained_zscore":
                    # Store the initial p_values:
                    results["orig_pval"] = results["pval"]
                    results["orig_reject"] = results["reject"]
                    # Perform FDR across all electrodes in population:
                    results["reject"], results["pval"], _, _ = \
                        multipletests(results["pval"], alpha=0.05,
                                      method=analysis_parameters["fdr_correction"])

                # Generate the summary table:
                summary_table = pd.DataFrame()
                for subject in results["subject"].unique():
                    # Extracting the data from this specific subject from the results table:
                    sub_results = results.loc[results["subject"] == subject]
                    sub_results_sig = sub_results.loc[sub_results["reject"] == True]
                    summary_table = summary_table.append({
                        "subject": subject,
                        "signal": analysis_parameters["signal"],
                        "percent-sig": len(sub_results_sig) / len(sub_results),
                        "n-sig-electrodes": len(sub_results_sig),
                        "total-n-electrodes": len(sub_results)
                    }, ignore_index=True)

                # Saving the results to file:
                all_res_file = param.files_prefix + "all_results.csv"
                sig_res_file = param.files_prefix + "sig_results.csv"
                summary_file = param.files_prefix + "summary.csv"
                results.to_csv(Path(save_path_results, all_res_file))
                results.loc[results["reject"] == True].to_csv(Path(save_path_results, sig_res_file))
                summary_table.to_csv(Path(save_path_results, summary_file))

                # Finally, generate and save the channels info:
                channels_info = pd.DataFrame()
                for ind, row in results.iterrows():
                    # Look for the x y and z coordinates of that channel:
                    coords = np.squeeze(
                        mni_coords.loc[mni_coords["channels"] == row["channel"], ["x", "y", "z"]].to_numpy())
                    ch_type = mni_coords.loc[mni_coords["channels"] == row["channel"], "ch_types"].item()
                    # Append to the data frame:
                    channels_info = channels_info.append(pd.DataFrame({
                        "subject": row["channel"].split("-")[0],
                        "channel": row["channel"],
                        "ch_types": ch_type,
                        "x": coords[0],
                        "y": coords[1],
                        "z": coords[2],
                        "reject": row["reject"],
                        "effect_strength": row["effect_strength"]
                    }, index=[0]))
                channels_info_file = param.files_prefix + "channels_info.csv"
                channels_info.to_csv(Path(save_path_figures, channels_info_file))

            # ==========================================================================================================
            # Two conditions
            # Else if two conditions were passed:
            elif len(analysis_parameters["conditions"]) == 2:
                for cond in analysis_parameters["conditions"]:
                    print("=" * 40)
                    print("Loading and preparing subject data")
                    print("Test: {}".format(analysis_parameters["test"]))
                    # Loading in parallel the data of each subject:
                    data, sfreqs, mni_coords, epochs = zip(*Parallel(n_jobs=param.njobs)(
                        delayed(prepare_test_data)(
                            param.BIDS_root, analysis_parameters["signal"],
                            analysis_parameters["baseline_correction"],
                            analysis_parameters["baseline_window"],
                            analysis_parameters["test_window"],
                            analysis_parameters[analysis_parameters["test"]][
                                "metric"],
                            analysis_parameters["test"],
                            subject,
                            baseline_time=analysis_parameters["baseline_time"],
                            crop_time=analysis_parameters["crop_time"],
                            condition=cond,
                            session=param.session,
                            preprocess_folder=param.preprocessing_folder,
                            preprocess_steps=param.preprocess_steps,
                            get_mni_coord=True,
                            montage_space="T1",
                            picks_rois=None,
                            multitaper_parameters=analysis_parameters["multitaper_parameters"],
                            aseg=param.aseg,
                            scal=analysis_parameters["scal"]
                        ) for subject in subjects_list))
                    # Make sure that all subjects had the same sampling frequency:
                    if len(set(sfreqs)) > 1:
                        raise Exception("The subjects have different sampling rates! That doesn't work!")
                    else:
                        sfreq = list(set(sfreqs))[0]
                    # Concatenating all the data together:
                    data = pd.concat(data, ignore_index=True)
                    mni_coords = pd.concat(mni_coords, ignore_index=True)
                    # Handling the cond name to remove any unwanted chars:
                    cond_name = cond.replace("/", "")
                    cond_name = cond_name.replace(" ", "_")
                    # Saving the data to file:
                    data.to_csv(Path(save_path_data, param.files_prefix + "data_" + cond_name + ".csv"))
                    # Loading the single subjects data:
                    if analysis_parameters["test"] == "t_test" \
                            or analysis_parameters["test"] == "wilcoxon_signed_rank" \
                            or analysis_parameters["test"] == "wilcoxon_rank_sum" \
                            or analysis_parameters["test"] == "bayes_t_test":
                        # Performing the test:
                        test_results = \
                            aggregated_stat_test(data, analysis_parameters["test_window"],
                                                 groups="channel",
                                                 test=analysis_parameters["test"],
                                                 p_val=analysis_parameters[analysis_parameters["test"]][
                                                     "p_val"],
                                                 alternative=
                                                 analysis_parameters[analysis_parameters["test"]]["alternative"])
                        test_results["condition"] = cond

                    elif analysis_parameters["test"] == "sustained_zscore":
                        # Extracting the data of that subject:
                        test_results = sustained_zscore_test(data,
                                                             analysis_parameters["test_window"][0],
                                                             groups="channel",
                                                             z_thresh=
                                                             analysis_parameters[analysis_parameters["test"]][
                                                                 "z_thresh"],
                                                             dur_thresh=
                                                             analysis_parameters[analysis_parameters["test"]][
                                                                 "dur_thresh"],
                                                             alternative=
                                                             analysis_parameters[analysis_parameters["test"]][
                                                                 "alternative"],
                                                             sfreq=sfreq)
                        test_results["condition"] = cond
                    elif analysis_parameters["test"] == "cluster_based":
                        test_results = \
                            cluster_based_permutation_test(data,
                                                           sfreq,
                                                           groups="channel",
                                                           onset=analysis_parameters["test_window"][0],
                                                           n_perm=analysis_parameters[analysis_parameters["test"]][
                                                               "n_perm"],
                                                           p_val=analysis_parameters[analysis_parameters["test"]][
                                                               "p_val"])
                        test_results["condition"] = cond
                    else:
                        raise Exception("ERROR: YOU HAVE PASSED A TEST THAT IS NOT SUPPORTED")
                    # Perform FDR if required:
                    if analysis_parameters["fdr_correction"] is not None and \
                            analysis_parameters["test"] != "sustained_zscore":
                        # Store the initial p_values:
                        test_results["orig_pval"] = test_results["pval"]
                        test_results["orig_reject"] = test_results["reject"]
                        # Perform FDR across all electrodes in population:
                        test_results["reject"], test_results["pval"], _, _ = \
                            multipletests(test_results["pval"], alpha=0.05,
                                          method=analysis_parameters["fdr_correction"])
                    # Finally, compute the response delays:
                    test_results = compute_latencies(epochs, test_results,
                                                     baseline=[-0.45, 0],
                                                     onset=[0.05, 0.5], sig_window_sec=0.020,
                                                     alpha=0.05)
                    # Append the results to the rest:
                    results.append(test_results)
                # Concatenating the results into 1 data frame:
                results = pd.concat(results, ignore_index=True)

                # Extract the significant results only:
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
                            "effect_strength-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["effect_strength"].to_list()[0]],
                            "effect_strength-{}".format(ch_results["condition"].to_list()[1]):
                                [ch_results["effect_strength"].to_list()[1]],
                            "condition": None,
                            "latency-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["latency"].to_list()[0]],
                            "latency-{}".format(ch_results["condition"].to_list()[1]): [
                                ch_results["latency"].to_list()[1]]
                        }, index=[0]))
                    elif len(ch_sig_results) == 1:
                        results_reformat = results_reformat.append(pd.DataFrame({
                            "subject": ch_results["subject"].to_list()[0],
                            "channel": channel,
                            "metric": ch_results["metric"].to_list()[0],
                            "reject": False,
                            "stat": ch_sig_results["stat"].to_list()[0],
                            "pval": ch_sig_results["pval"].to_list()[0],
                            "onset": ch_sig_results["onset"].to_list()[0],
                            "offset": ch_sig_results["offset"].to_list()[0],
                            "effect_strength-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["effect_strength"].to_list()[0]],
                            "effect_strength-{}".format(ch_results["condition"].to_list()[1]):
                                [ch_results["effect_strength"].to_list()[1]],
                            "condition": ch_sig_results["condition"].to_list()[0],
                            "latency-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["latency"].to_list()[0]],
                            "latency-{}".format(ch_results["condition"].to_list()[1]): [
                                ch_results["latency"].to_list()[1]]
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
                            "effect_strength-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["effect_strength"].to_list()[0]],
                            "effect_strength-{}".format(ch_results["condition"].to_list()[1]):
                                [ch_results["effect_strength"].to_list()[1]],
                            "condition": "both",
                            "latency-{}".format(ch_results["condition"].to_list()[0]):
                                [ch_results["latency"].to_list()[0]],
                            "latency-{}".format(ch_results["condition"].to_list()[1]): [
                                ch_results["latency"].to_list()[1]]
                        }, index=[0]))
                    else:
                        raise Exception("Something went wrong!")
                results_reformat = results_reformat.reset_index(drop=True)
                # Generate the summary table:
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

                # Saving the results to file:
                all_res_file = param.files_prefix + "all_results.csv"
                sig_res_file = param.files_prefix + "sig_results.csv"
                summary_file = param.files_prefix + "summary.csv"
                results_reformat.to_csv(Path(save_path_results, all_res_file))
                results_reformat.loc[results_reformat["reject"] == True].to_csv(Path(save_path_results, sig_res_file))
                summary_table.to_csv(Path(save_path_results, summary_file))

                # Finally, generate and save the channels info:
                channels_info = pd.DataFrame()
                for ind, row in results_reformat.iterrows():
                    # Look for the x y and z coordinates of that channel:
                    coords = np.squeeze(mni_coords.loc[mni_coords["channels"] == row["channel"],
                                                       ["x", "y", "z"]].to_numpy())
                    ch_type = mni_coords.loc[mni_coords["channels"] == row["channel"], "ch_types"].item()
                    # Append to the data frame:
                    channels_info = channels_info.append(pd.DataFrame({
                        "subject": row["channel"].split("-")[0],
                        "channel": row["channel"],
                        "ch_types": ch_type,
                        "x": coords[0],
                        "y": coords[1],
                        "z": coords[2],
                        "condition": row["condition"],
                        "reject": row["reject"],
                        "effect_strength-{}".format(analysis_parameters["conditions"][0].split("/")[1]):
                            row["effect_strength-{}".format(analysis_parameters["conditions"][0])],
                        "effect_strength-{}".format(analysis_parameters["conditions"][1].split("/")[1]):
                            row["effect_strength-{}".format(analysis_parameters["conditions"][1])],
                        "latency-{}".format(analysis_parameters["conditions"][0].split("/")[1]):
                            row["latency-{}".format(analysis_parameters["conditions"][0])],
                        "latency-{}".format(analysis_parameters["conditions"][1].split("/")[1]):
                            row["latency-{}".format(analysis_parameters["conditions"][1])]
                    }, index=[0]))
                channels_info_file = param.files_prefix + "channels_info.csv"
                channels_info.to_csv(Path(save_path_figures, channels_info_file))
            elif len(analysis_parameters["conditions"]) > 2:
                raise Exception("You have passed more than two conditions for visual responsiveness. "
                                "There is no easy way to deal with that, therefore this option is not available!")

    print("Visual responsiveness was successfully computed for all the subjects!")
    print("Now plotting the results")
    # if len(subjects_list) > 1:  # This plotting concantenates across subjects!
    #     plot_roi_evoked(configs, save_folder=save_folder)
    # plot_single_electrodes(configs, save_folder=save_folder)


if __name__ == "__main__":
    visual_responsiveness(None, save_folder="super")
