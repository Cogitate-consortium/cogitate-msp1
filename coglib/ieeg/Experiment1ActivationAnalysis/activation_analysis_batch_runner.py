"""
This scripts loops through all single config to perform the different activation analyses as slurm jobs
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import subprocess
from general_helper_functions.pathHelperFunctions import find_files


def activation_analysis_batch_runner(lmm=False, onset_offset=False, duration_tracking=False, duration_decoding=False):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Getting the current dir
    pwd = os.getcwd()
    if lmm:
        # Get the different config
        config_files = find_files(os.path.join(pwd, "lmm_configs"),
                                  naming_pattern="*", extension=".json")
        # Launching a job for each:
        for config in config_files:
            # Run the rsa analysis script using the customized config file
            run_command = "sbatch " + "lmm_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
            subprocess.Popen(run_command, shell=True)
    if onset_offset:
        # Get the different config
        config_files = find_files(os.path.join(pwd, "onset_offset_config"),
                                  naming_pattern="*", extension=".json")
        # Launching a job for each:
        for config in config_files:
            # Run the rsa analysis script using the customized config file
            run_command = "sbatch " + "onset_offset_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
            subprocess.Popen(run_command, shell=True)

    if duration_tracking:
        # Get the different config
        config_files = find_files(os.path.join(pwd, "duration_tracking_config"),
                                  naming_pattern="*", extension=".json")
        # Launching a job for each:
        for config in config_files:
            # Run the rsa analysis script using the customized config file
            run_command = "sbatch " + "duration_tracking_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
            subprocess.Popen(run_command, shell=True)

    if duration_decoding:
        # Get the different config
        config_files = find_files(os.path.join(pwd, "duration_decoding_config"),
                                  naming_pattern="*", extension=".json")
        # Launching a job for each:
        for config in config_files:
            # Run the rsa analysis script using the customized config file
            run_command = "sbatch " + "duration_decoding_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
            subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    activation_analysis_batch_runner(lmm=True, onset_offset=False, duration_tracking=False, duration_decoding=False)
