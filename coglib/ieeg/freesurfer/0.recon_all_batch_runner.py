"""Batch runner for recon_all."""
import subprocess
from general_helper_functions.pathHelperFunctions import find_files
from pathlib import Path


def recon_all(subjects_list, subjects_dir):
    """
    This function runs slurm jobs for the freesurfer recon all on all subjects submitted
    :param subjects_list: (list of strings) list of subjects for whom to preprocessing the freesurfer recon all
    pipeline!
    :param subjects_dir: (path string) freesurfer directory
    :return:
    """
    # Looping through each subject:
    for subject in subjects_list:
        # Finding the T1 scan from this subject:
        T1_scan = find_files(Path(subjects_dir, subject, "mri", "origin"), naming_pattern="*", extension=".mgz")
        assert len(T1_scan) == 1, "There was not exactly one T1 scan for subject {}!".format(subject)
        # Run the recon_all_job analysis script using the customized config file
        run_command = "sbatch " + "recon_all_job.sh" \
                      + " --FREESURFER_PATH=" \
                      + '"{}"'.format(subjects_dir) \
                      + " --subject=" \
                      + '"{}"'.format(subject) \
                      + " --T1_SCAN={}".format(T1_scan[0])
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    recon_all(["CE103", "CE107", "CE108"],
              "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs")
