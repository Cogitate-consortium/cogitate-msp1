#!/bin/bash
#SBATCH --partition=octopus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/fs/wang_atlas_mapping-%A_%a.out
#SBATCH --job-name=SE_recon_fix

# Sorting out inputs:
subject=""
FREESURFER_PATH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --FREESURFER_PATH=*)
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

# Preparing the free surfer module and setting directories:
echo "Loading freesurfer"
module purge
module load FreeSurfer/6.0.1-centos6_x86_64; source ${FREESURFER_HOME}/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$FREESURFER_PATH

mris_make_surfaces -orig_white white -orig_pial white -aseg ../mri/aseg.presurf -nowhite -mgz -T1 brain.finalsurfs $subject lh
mris_make_surfaces -orig_white white -orig_pial white -aseg ../mri/aseg.presurf -nowhite -mgz -T1 brain.finalsurfs $subject rh
