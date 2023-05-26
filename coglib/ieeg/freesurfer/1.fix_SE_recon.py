"""Batch runner for SE_recon_fix_job."""
import subprocess
from general_helper_functions.pathHelperFunctions import find_files
from pathlib import Path


def recon_all(subjects_list, subjects_dir):
    """
    This function runs slurm jobs for the SE_recon_fix_job on all subjects passed
    :param subjects_list: (list of strings) list of subjects for whom to preprocessing the freesurfer recon all pipeline!
    :param subjects_dir: (path string) freesurfer directory
    :return:
    """
    # Looping through each subject:
    for subject in subjects_list:
        # Run the SE_recon_fix_job analysis script using the customized config file
        run_command = "sbatch " + "SE_recon_fix_job.sh" \
                      + " --FREESURFER_PATH=" \
                      + '"{}"'.format(subjects_dir) \
                      + " --subject=" \
                      + '"{}"'.format("sub-" + subject)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    recon_all(["SE103", "SE107", "SE108", "SE109", "SE110", "SE112", "SE113", "SE115", "SE118", "SE119"],
              "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs")
