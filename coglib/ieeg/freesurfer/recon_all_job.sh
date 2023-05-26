#!/bin/bash
#SBATCH --partition=octopus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs/recon_all-%A_%a.out
#SBATCH --job-name=recon_all

subject=""
FREESURFER_PATH=""
T1_SCAN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --FREESURFER_PATH=*)
      FREESURFER_PATH="${1#*=}"
      ;;
    --T1_SCAN=*)
      FREESURFER_PATH="${1#*=}"
      ;;
    --subject=*)
      subject="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument: ${1}*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done
echo $subject
echo $FREESURFER_PATH

# Get the current working directory, as we will change dir later on to be in the path of the docker:
SCRIPTS_DIR=$(pwd)
# Preparing the free surfer module and setting directories:
module purge
module load FreeSurfer/6.0.1-centos6_x86_64; source ${FREESURFER_HOME}/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$FREESURFER_PATH

# Launch recon all:
recon-all -s ${subject} -i ${T1_SCAN} -all
