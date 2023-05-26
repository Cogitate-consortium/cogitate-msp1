"""
This script loops through all single config to perform the category selectivity as slurm jobs
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import subprocess
from general_helper_functions.pathHelperFunctions import find_files


def category_selectivity_batch_runner():
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
        run_command = "sbatch " + "category_selectivity_job.sh"\
                      + " --config=" \
                      + '"{}"'.format(config)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    category_selectivity_batch_runner()
