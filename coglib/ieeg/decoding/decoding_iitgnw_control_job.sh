#!/bin/bash
#SBATCH --partition=xnat
#SBATCH --nodelist=cn10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=80000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=simon.henin@nyumc.org
#SBATCH --time 4-24:00:00
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/decoding_analysis/slurm-%A_%a.out
#SBATCH --job-name=decoding


cd /hpc/users/$USER/sw/github/ECoG

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; 
conda activate /hpc/users/$USER/.conda/envs/cogitate_ecog

export PYTHONPATH=$PYTHONPATH:/hpc/users/$USER/sw/github/ECoG

python decoding/decoding_control_iit_vs_iitgnw.py
