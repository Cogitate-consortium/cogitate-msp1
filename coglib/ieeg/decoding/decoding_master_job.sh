#!/bin/bash
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=simon.henin@nyumc.org
#SBATCH --time 4-24:00:00
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/decoding_analysis/slurm-%A_%a.out
#SBATCH --job-name=decoding
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

cd /hpc/users/$USER/sw/github/ECoG

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; 
conda activate /hpc/users/$USER/.conda/envs/cogitate_ecog

export PYTHONPATH=$PYTHONPATH:/hpc/users/$USER/sw/github/ECoG

python decoding/decoding_master.py --config "${config}"
