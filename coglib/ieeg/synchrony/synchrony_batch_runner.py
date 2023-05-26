"""
This script loops over the different decoding configs to launch a slurm job each
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
import argparse
import subprocess
from general_helper_functions.pathHelperFunctions import find_files


def synchrony_batch_runner(robustness_checks=False, naming_pattern="*", exclude=None):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Getting the current dir
    pwd = os.getcwd()
    # Get the different config
    config_files = find_files(os.path.join(pwd, "configs"),
                              naming_pattern=naming_pattern, extension=".json")
    
    if exclude is not None:
        for exc in exclude:
            print( 'excluding files matching: %s' % exc )
            config_files = [x for x in config_files if exc not in x]
            
    print('\n-------\nlaunching config files:')
    # Launching a job for each:
    for config in config_files:
        print( config )
        # Run the rsa analysis script using the customized config file
        if robustness_checks:
            run_command = "sbatch " + "synchrony_robustness_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
        else:
            run_command = "sbatch " + "synchrony_master_job.sh" \
                          + " --config=" \
                          + '"{}"'.format(config)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implements batch runner")
    parser.add_argument('--pattern', type=str, default="*",
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('-e', '--exclude', nargs='+', default=None,
                        help="Config file for analysis parameters (file name + path)")                        
    args = parser.parse_args()
    synchrony_batch_runner(robustness_checks=False, naming_pattern=args.pattern, exclude=args.exclude)
