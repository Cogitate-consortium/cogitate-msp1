"""
This script investigates condition selectivity in single channels. In the cogitate, category selectivity is mainly
investigated
    authors: Alex Lepauvre
    contributors: Katarina Bendtz, Simon Henin
    katarina.bendtz@tch.harvard.edu
    Simon.Henin@nyulangone.org
"""
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed

from statsmodels.stats.multitest import multipletests

from category_selectivity_analysis.category_selectivity_helper_function import *

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from category_selectivity_analysis.category_selectivity_parameters_class import CategorySelectivityAnalysisParameters
from category_selectivity_analysis.plot_category_selectivity_results import plot_category_selectivity
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def category_selectivity_master(subjects_list, save_folder="super"):
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
            CategorySelectivityAnalysisParameters(
                config, sub_id=save_folder)
        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "category_selectivity")
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
            # Save the parameters to these two directories:
            param.save_parameters(save_path_data)
            param.save_parameters(save_path_results)
            param.save_parameters(save_path_figures)

            # Save the participants list to these directories too:
            with open(Path(save_path_figures, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_results, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")
            with open(Path(save_path_data, 'subjects_list.txt'), 'w') as f:
                for subject in subjects_list:
                    f.write(f"{subject}\n")

            # ==========================================================================================================
            # Load and prepare the data:
            # Loading in parallel the data of each subject:
            data, sfreqs, mni_coords = zip(*Parallel(n_jobs=param.njobs)(
                delayed(prepare_test_data)(
                    param.BIDS_root, analysis_parameters["signal"],
                    analysis_parameters["baseline_correction"],
                    analysis_parameters["test_window"],
                    analysis_parameters["metric"],
                    analysis_parameters["to_compare"],
                    subject,
                    baseline_time=analysis_parameters["baseline_time"],
                    crop_time=analysis_parameters["crop_time"],
                    sel_conditions=analysis_parameters["conditions"],
                    session=param.session,
                    task_name=param.task_name,
                    preprocess_folder=param.preprocessing_folder,
                    preprocess_steps=param.preprocess_steps,
                    channel_types={"seeg": True, "ecog": True},
                    get_mni_coord=True,
                    select_vis_resp=analysis_parameters["select_vis_resp"],
                    vis_resp_folder=analysis_parameters["vis_resp_folder"],
                    aseg=param.aseg,
                    montage_space="T1",
                    picks_roi=None,
                    multitaper_parameters=analysis_parameters["multitaper_parameters"],
                    scal=analysis_parameters["scal"]
                ) for subject in subjects_list))
            # Remove the nones from the list:
            data = [df for df in data if df is not None]
            sfreqs = [freq for freq in sfreqs if freq is not None]
            mni_coords = [coords for coords in mni_coords if coords is not None]
            # Concatenating the data frames:
            data = pd.concat(data, ignore_index=True)
            mni_coords = pd.concat(mni_coords, ignore_index=True)
            # Save the data to file:
            data.to_csv(Path(save_path_data, "data.csv"))
            # Perform the test:
            if analysis_parameters["test"] == "highest_vs_all":
                results = highest_vs_all(data, groups="channel", test=analysis_parameters["stats_fun"],
                                         p_val=analysis_parameters["p_val"])
            elif analysis_parameters["test"] == "highest_vs_second":
                results = highest_vs_second(data, groups="channel", test=analysis_parameters["stats_fun"],
                                            p_val=analysis_parameters["p_val"])
            elif analysis_parameters["test"] == "dprime_test":
                results = dprime_test(data, groups="channel", n_perm=analysis_parameters["dprime_param"]["n_perm"],
                                      tail=analysis_parameters["dprime_param"]["tail"],
                                      p_val=analysis_parameters["p_val"], n_jobs=param.njobs)
            else:
                raise Exception("You have passed a test that is not supported!")

            # Perform FDR if required:
            if analysis_parameters["fdr_correction"] is not None:
                # Store the initial p_values:
                results["orig_pval"] = results["pval"]
                results["orig_reject"] = results["reject"]
                # Perform FDR separately for each category:
                results_fdr = pd.DataFrame()
                for cond in list(results["condition"].unique()):
                    # Extract the results of that  one condition:
                    cond_results = results.loc[results["condition"] == cond]
                    # Perform FDR:
                    cond_results["reject"], cond_results["pval"], _, _ = \
                        multipletests(cond_results["pval"], alpha=analysis_parameters["p_val"],
                                      method=analysis_parameters["fdr_correction"])
                    # Append to the data frame:
                    results_fdr = results_fdr.append(cond_results, ignore_index=True)
                # We can now overwrite the original results:
                results = results_fdr

            # The dprime test has the particularity of testing each category as opposed to only 1. This requires
            # special handling:
            if analysis_parameters["test"] == "dprime_test":
                result_reformat = pd.DataFrame()
                for channel in list(results["channel"].unique()):
                    # Extract the results for this specific channel:
                    ch_results = results.loc[results["channel"] == channel]
                    # Extract those that are significant:
                    ch_sig_results = ch_results.loc[ch_results["reject"] == True]
                    # If more than one category was found to be significant, marking this channel as none significant:
                    if len(ch_sig_results) > 1 or len(ch_sig_results) == 0:
                        result_reformat = result_reformat.append(pd.DataFrame({
                            "subject": channel.split("-")[0],
                            "channel": channel,
                            "metric": None,
                            "reject": False,
                            "condition": None,
                            "stat": None,
                            "pval": 1,
                            "effect_strength": None
                        }, index=[0]))
                    else:  # But if there is indeed only one, then we are in business
                        result_reformat = result_reformat.append(pd.DataFrame({
                            "subject": channel.split("-")[0],
                            "channel": channel,
                            "metric": None,
                            "reject": True,
                            "condition": ch_sig_results["condition"].item(),
                            "stat": ch_sig_results["stat"].item(),
                            "pval": ch_sig_results["pval"].item(),
                            "effect_strength": ch_sig_results["effect_strength"].item()
                        }, index=[0]))
                results = result_reformat

            # Generate the summary table:
            summary_table = pd.DataFrame()
            # Get the unique conditions:
            cond_list = list(results["condition"].unique())
            for subject in results["subject"].unique():
                # Extracting the data from this specific subject from the results table:
                sub_results = results.loc[results["subject"] == subject]
                sub_results_sig = sub_results.loc[sub_results["reject"] == True]
                count_per_cond = {cond: len(sub_results_sig.loc[sub_results_sig["condition"] == cond])
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
            results.to_csv(Path(save_path_results, all_res_file))
            results.loc[results["reject"] == True].to_csv(Path(save_path_results, sig_res_file))
            summary_table.to_csv(Path(save_path_results, summary_file))

            # Finally, generate and save the channels info:
            channels_info = pd.DataFrame()
            for ind, row in results.iterrows():
                # Look for the x y and z coordinates of that channel:
                coords = np.squeeze(mni_coords.loc[mni_coords["channels"] == row["channel"],
                                                   ["x", "y", "z"]].to_numpy())
                ch_type = mni_coords.loc[mni_coords["channels"] == row["channel"], "ch_types"].item()
                # Append to the data frame:
                channels_info = channels_info.append(pd.DataFrame({
                    "subject": row["channel"].split("-")[0],
                    "channel": row["channel"],
                    "x": coords[0],
                    "y": coords[1],
                    "z": coords[2],
                    "ch_types": ch_type,
                    "condition": row["condition"],
                    "reject": row["reject"],
                    "effect_strength": row["effect_strength"]
                }, index=[0]))
            channels_info_file = param.files_prefix + "channels_info.csv"
            channels_info.to_csv(Path(save_path_figures, channels_info_file))

    print("Category selectivity was successfully computed for all the subjects!")
    print("Now plotting the results")
    plot_category_selectivity(configs, save_folder=save_folder)


if __name__ == "__main__":
    category_selectivity_master(None, save_folder="super")
