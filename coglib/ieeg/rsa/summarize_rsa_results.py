"""
This script summarizes the RSA results
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import os
from pathlib import Path

from rsa.rsa_helper_functions import *
from general_helper_functions.pathHelperFunctions import find_files, path_generator
from rsa.rsa_parameters_class import RsaParameters
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
correlation_method = "kendall"


def summarize_rsa(configs, save_folder="super"):
    if len(configs) == 0 or configs is None:
        configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*", extension=".json")

    results_summary_df = pd.DataFrame(columns=["analysis_name", "roi", "compared_predictors", "time_win", "validated",
                                               "iit_correlation", "gnw_correlation", "correlation_difference",
                                               "p-value"])
    # Looping through each:
    for config in configs:
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
            # Load the results:
            results_file = (Path(load_path_results, param.files_prefix +
                                 "theories_correlation_results_zscore.csv"))
            # Load the results:
            try:
                results_df = pd.read_csv(results_file)
            except FileNotFoundError:
                print("WARNING: no results file found for " + analysis_name)
                continue
            for time_win in results_df["time_win"].unique():
                time_win_results = results_df.loc[results_df["time_win"] == time_win]
                # Parse the results:
                for roi in list(results_df["roi"].unique()):
                    roi_results = time_win_results.loc[time_win_results["roi"] == roi]
                    for pred_pairs in list(roi_results["predictors"].unique()):
                        pred_results = roi_results.loc[roi_results["predictors"] == pred_pairs]
                        # Extract the relevant info:
                        correlation_diff = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                            "correlation_difference"].item()
                        p_val = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                 "p-value"].item()
                        p_val_iit = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                     "iit_p-val"].item()
                        if pred_pairs == "iit_gnw_no_offset":
                            corr_gnw = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                        "gnw_no_offset"].item()
                            p_val_gnw = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                         "gnw_no_offset_p-val"].item()
                        else:
                            corr_gnw = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                        "gnw"].item()
                            p_val_gnw = pred_results.loc[pred_results["correlation_method"] == correlation_method,
                                                         "gnw_p-val"].item()
                        validated = False  # Guilty until proven innocent
                        if "iit" in roi:
                            if correlation_diff > 0 and p_val < 0.05:
                                validated = True
                        elif "gnw" in roi:
                            if correlation_diff < 0 and p_val < 0.05:
                                validated = True
                        else:
                            continue

                        results_summary_df = results_summary_df.append(pd.DataFrame({
                            "analysis_name": analysis_name,
                            "roi": roi,
                            "compared_predictors": pred_pairs,
                            "time_win": time_win,
                            "validated": validated,
                            "iit_correlation": pred_results.loc[
                                pred_results["correlation_method"] == correlation_method, "iit"].item(),
                            "iit_p-value": p_val_iit,
                            "gnw_p-value": p_val_gnw,
                            "gnw_correlation": corr_gnw,
                            "correlation_difference": pred_results.loc[pred_results[
                                                                           "correlation_method"] == correlation_method,
                                                                       "correlation_difference"].item(),
                            "p-value": pred_results.loc[
                                pred_results["correlation_method"] == correlation_method, "p-value"].item()
                        }, index=[0]))

    results_summary_df = results_summary_df.reset_index(drop=True)
    save_path = load_path_results.parents[1]
    results_summary_df.to_csv(Path(save_path, "rsa_results_summary.csv"))


if __name__ == "__main__":
    configs = find_files(Path(os.getcwd(), "super_subject_config"), naming_pattern="*",
                         extension=".json")
    summarize_rsa(configs, save_folder="super")
