"""
This scripts loops through all configs to perform the visual responsiveness
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import subprocess
import os
from general_helper_functions.pathHelperFunctions import find_files


def visual_responsiveness_batch_runner():
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Getting the current dir
    pwd = os.getcwd()

    # Get the different config
    config_files = find_files(os.path.join(pwd, "super_subject_config"),
                              naming_pattern="*", extension=".json")
    # Launching a job for each:
    for config in config_files:
        # Run the rsa analysis script using the customized config file
        run_command = "sbatch " + "visual_responsiveness_job.sh"\
                      + " --config=" \
                      + '"{}"'.format(config)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    visual_responsiveness_batch_runner()
