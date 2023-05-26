"""
This script plots the results of the onset offset analysis
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""

import shutil
import os
from pathlib import Path
from rsa.rsa_helper_functions import *
from general_helper_functions.pathHelperFunctions import find_files, path_generator
from Experiment1ActivationAnalysis.activation_analysis_parameters_class import DurationAnalysisParameters
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def plotting_onset_offset_results(configs, save_folder="super"):
    """
    This function plots the results of the onset and offset analysis.
    :param configs: (list of strings) list of the config files to use to plot
    :return:
    """
    # Looping through the configs:
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)
        # Generating the analysis object with the current config:
        param = DurationAnalysisParameters(config, sub_id=save_folder)
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

            # Looping through each ROI:
            for roi in param.rois:
                # Finding the results file:
                onset_offset_results_file = find_files(load_path_results, "*" + roi + "_onset_offset_results",
                                                       extension=".csv")
                assert len(onset_offset_results_file) == 1, "More than one file was found for rsa results!"
                # Load the file:
                onset_offset_results = pd.read_csv(onset_offset_results_file[0])
                # Now looping through each condition:
                for condition in list(onset_offset_results["condition"].unique()):
                    if condition == "no_diff":
                        continue
                    # Extracting the results of that condition only:
                    condition_results = onset_offset_results.loc[onset_offset_results["condition"] == condition]
                    # Generating the path to the save dir of that specific condition:
                    cond_fig_path = Path(save_path_fig, condition)
                    if not os.path.isdir(cond_fig_path):
                        os.makedirs(cond_fig_path)
                    # Looping through each channel of that condition:
                    for channel in list(condition_results["channel"].unique()):
                        # Find the figure of that specific electrode:
                        channel_figs = find_files(param.figures_root, "*" + channel + "_", extension=".png")
                        # Copying these figures to the directory of that condition:
                        for fig_file in channel_figs:
                            shutil.copy(fig_file, cond_fig_path)

    print("Done!")


if __name__ == "__main__":
    # Fetching all the config files:
    configs = find_files(Path(os.getcwd(), "config"), naming_pattern="*", extension=".json")
    plotting_onset_offset_results(configs, save_folder="super")
