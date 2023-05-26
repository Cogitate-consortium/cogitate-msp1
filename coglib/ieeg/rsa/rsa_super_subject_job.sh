#!/bin/bash
#SBATCH --partition=octopus
#SBATCH --exclude=cn10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de
#SBATCH --time 120:00:00
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/rsa/slurm-%A_%a.out
#SBATCH --job-name=rsa
config=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config=*)
      config="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument: ${1}*\n"
      printf "***************************\n"
      exit 1

  esac
  shift
  echo ${participant_id}
  echo ${config}
done

cd /hpc/users/alexander.lepauvre/sw/github/ECoG

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /hpc/users/$USER/.conda/envs/mne_ecog02

export PYTHONPATH=$PYTHONPATH:/hpc/users/alexander.lepauvre/sw/github/ECoG

python ./rsa/rsa_master.py --config "${config}"
