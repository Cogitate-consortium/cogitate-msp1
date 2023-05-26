"""
This script loops over the different rsa configs to launch a slurm job each
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import subprocess
from general_helper_functions.pathHelperFunctions import find_files


def rsa_batch_runner(robustness_checks=False):
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
        if robustness_checks:
            run_command = "sbatch " + "rsa_robustness_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
        else:
            run_command = "sbatch " + "rsa_super_subject_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    rsa_batch_runner(robustness_checks=False)
