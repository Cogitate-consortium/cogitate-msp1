"""Batch runner for wang_mapping_job."""
import subprocess
from os import listdir
from os.path import isdir, join


def wang_mapping_batch(fs_path, subjects_list=None):
    """
    This function runs slurm jobs for the wang atlas mapping on all subjects founds in the free surfer directory
    :param fs_path: (string) path to the freesurfer directory containing the freesurfer data of all subjects
    :param subjects_list: (list of strings) name of the subjects on whom to preprocessing the script. If none, it will be
    executed for all subjects
    :param singularity_path: (string path) path to the wang singularity!
    :return:
    """
    # List the subjects in the directory
    if subjects_list is None:
        subjects_list = [f for f in listdir(fs_path) if isdir(join(fs_path, f)) and "sub-" in f]
    # Launching a job for each:
    for subject in subjects_list:
        # Run the wang_mapping_job analysis script using the customized config file
        run_command = "sbatch " + "wang_mapping_job.sh" \
                      + " --FREESURFER_PATH=" \
                      + '"{}"'.format(fs_path) \
                      + " --subject=" \
                      + '"{}"'.format(subject)
        subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    wang_mapping_batch("/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs",
                       subjects_list=None)
