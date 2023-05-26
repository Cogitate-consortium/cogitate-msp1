"""
This script computes linear mixed model to test the theories predictions regarding activation patterns associated
with sustained perception
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from Experiment1ActivationAnalysis.activation_analysis_helper_function import *

from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
from Experiment1ActivationAnalysis.plot_lmm_results import plotting_lmm_results
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def activation_analysis_super_subject(subjects_list, save_folder="super"):
    """
    This function performs the linear mixed models, i.e. investigate the theories predictions realization in the data
    by fitting linear mixed models with predictors based on the theories predictions and comparing the different
    theories models to see what fits best
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
        configs = find_files(Path(os.getcwd(), "lmm_configs"), naming_pattern="*", extension=".json")
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
            save_path_data = path_generator(param.save_root,
                                            analysis=analysis_name,
                                            preprocessing_steps=param.preprocess_steps,
                                            fig=False, data=True)
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
                lmm_df = []
                mni_coords = []
                # Loading in parallel the data of each subject:
                for subject in subjects_list:
                    sub_df, _, mni_coord = prepare_test_data(param.BIDS_root, analysis_parameters["signal"],
                                                             analysis_parameters["lmm_parameters"][
                                                                 "additional_predictors"],
                                                             analysis_parameters["metric"],
                                                             analysis_parameters["time_bins"],
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
                                                             moving_average_ms=analysis_parameters["moving_average_ms"],
                                                             multitaper_parameters=
                                                             analysis_parameters["multitaper_parameters"],
                                                             scal=analysis_parameters["scal"]
                                                             )
                    if sub_df is not None:
                        lmm_df.append(sub_df)
                        mni_coords.append(mni_coord)

                # Convert the list of data frame to one single data frame:
                lmm_df = pd.concat(lmm_df, axis=0, ignore_index=True)
                mni_coords = pd.concat(mni_coords, ignore_index=True)
                # Save the lmm data:
                lmm_df.to_csv(Path(save_path_data, param.files_prefix + roi + "_lmm_data.csv"), index=False)

                # Fitting the linear mixed models in parallel:
                lmm_results, anova_results = \
                    zip(*Parallel(n_jobs=param.n_jobs)(
                        delayed(fit_lmm)(lmm_df.loc[lmm_df["channel"] == channel],
                                         analysis_parameters["lmm_parameters"]["models"],
                                         lmm_df.loc[lmm_df["channel"] == channel][
                                             "epoch"],
                                         group=channel,
                                         alpha=analysis_parameters["lmm_parameters"]["p_value"],
                                         package=analysis_parameters["lmm_parameters"]["package"])
                        for channel in lmm_df["channel"].unique()))
                # Concatenating the results into 1 data frame:
                lmm_results = pd.concat(lmm_results, axis=0, ignore_index=True)
                # Saving the results to file:
                file_name = param.files_prefix + roi + "_lmm_results.csv"
                lmm_results.to_csv(Path(save_path_results, file_name), index=False)

                if analysis_parameters["lmm_parameters"]["package"] == "lmer":
                    anova_results = pd.concat(anova_results, axis=0, ignore_index=True)
                    file_name = param.files_prefix + roi + "_anova_results.csv"
                    anova_results.to_csv(Path(save_path_results, file_name), index=False)

                # Reject the models that didn't converge:
                lmm_results = lmm_results.loc[lmm_results["converged"] == True].reset_index(drop=True)
                if analysis_parameters["lmm_parameters"]["package"] == "lmer":
                    anova_results = anova_results.loc[anova_results["converged"] == True].reset_index(drop=True)

                # Selecting the best models using the selected criterion:
                best_models = model_comparison(lmm_results,
                                               criterion=analysis_parameters["lmm_parameters"][
                                                   "model_selection_criterion"],
                                               test=analysis_parameters["lmm_parameters"]["test"])
                # Saving the results to file:
                file_name = param.files_prefix + roi + "_best_lmm_results.csv"
                best_models.to_csv(Path(save_path_results, file_name), index=False)

                # Same for the anova:
                if analysis_parameters["lmm_parameters"]["package"] == "lmer":
                    best_models_anova = model_comparison(anova_results,
                                                         criterion=analysis_parameters["lmm_parameters"][
                                                             "model_selection_criterion"],
                                                         test=analysis_parameters["lmm_parameters"]["test"])
                    # Saving the results to file:
                    file_name = param.files_prefix + roi + "_best_models_anovas.csv"
                    best_models_anova.to_csv(Path(save_path_results, file_name), index=False)

                # Writing up a summary table:
                # ------------------------------------------------------------------------------------------------------
                summary_columns = ["subject", "#electrodes"] + list(best_models["model"].unique())
                # Create summary data frame:
                summary_table = pd.DataFrame(index=range(len(list(lmm_df["subject"].unique()))),
                                             columns=summary_columns)
                for ind, sub in enumerate(list(lmm_results["subject"].unique())):
                    # Add subject ID:
                    summary_table.loc[ind, "subject"] = sub
                    # Get total number of channels:
                    summary_table.loc[ind, "#electrodes"] = \
                        len(lmm_results.loc[lmm_results["subject"] == sub, "group"].unique())
                    # Extract the model results of that specific subject:
                    sub_model_results = best_models.loc[best_models["subject"] == sub]
                    # Loop through the significant models of that subject:
                    for model in list(sub_model_results["model"].unique()):
                        # Add how many electrodes are significant for a given model:
                        summary_table.loc[ind, model] = \
                            len(sub_model_results.loc[sub_model_results["model"] == model, "group"].unique())
                # Saving the results to file:
                file_name = param.files_prefix + roi + "_lmm_summary.csv"
                summary_table.to_csv(Path(save_path_results, file_name), index=False)

                # Finally, generate and save the channels info:
                channels_info = pd.DataFrame()
                for channel in list(best_models["group"].unique()):
                    # Get the model of this channel:
                    ch_model = best_models.loc[best_models["group"] == channel, "model"].unique()[0]
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
                        "model": ch_model
                    }, index=[0]))
                channels_info_file = param.files_prefix + roi + "channels_info.csv"
                channels_info.to_csv(Path(save_path_fig, channels_info_file))

    print("Linear mixed model computations done!")
    print("Plotting the results")
    plotting_lmm_results(configs, save_folder=save_folder)


if __name__ == "__main__":
    activation_analysis_super_subject(None, save_folder="super")
