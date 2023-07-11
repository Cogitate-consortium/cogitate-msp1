# MEG
This folder contains all the code created by Oscar Ferrante and Ling Liu in the frame of the COGITATE project

## Setup:
Create a new conda environment by running the following:
```
conda env create --file=requirements_cogitate_meg.yaml
```
For the linear mixed midel (LMM) analysis used in the activation analysis, create a specific LMM conda environment by running the following:
```
conda env create --file=requirements_cogitate_meg_lmm.yaml
```
The environments are tailored for Linux and the HPC, so some things might break a little if you use windows or Mac 
(not tested very thoroughly).

**Installation time ~= 90min**

## Change root path:
To run the analysis described below on the sample data, make sure to change the bids root path in /meeg/config/config.py:
*$ROOT/sample_data/bids*

### Running 3rd level quality checks:
In the command line, enter:
sbatch --array=<subjectID> srun_qc.sh <SA/SB> V1

### Running preprocessing:
In the command line, enter:
```
python REPO_ROOT/cogitate-msp1/meeg/preprocessing/99_run_preproc.py --sub SA124 --visit v1 --record run --step 1
```
When the first preprocessing step is finished, enter:
```
python REPO_ROOT/cogitate-msp1/meeg/preprocessing/P99_run_preproc.py --sub SA124 --visit v1 --record run --step 2
```
Expected output: the script should generate a directory under:
*$ROOT/sample_data/bids/derivatives/preprocessing/sub-SA124*
containing several subfolders, one for each preprocessing steps. The epoching files contain the final state of 
the data ready for the next analysis steps.

**Run time ~= 90min**

### Running analyses:
For each analysis, run the scripts in the corresponding analysis folder (e.g., /meeg/activation) following the order
reported in the file name (e.g., first run "S01_source_loc.py", then "S02_source_loc_ga.py" and so on).
To run any of the individual-level analysis, enter:
```
python REPO_ROOT/cogitate-msp1/meeg/ANALYSIS_FOLDER/ANALYSIS_CODE.py --sub SA124 --visit v1
```
Replace ANALYSIS_FOLDER with the name of the folder corresponding to the analysis you want to run (e.g., activation)
and ANALYSIS_CODE with the name of the script you want to exacute (e.g., S01_source_loc.py).
To run any of the group-level analysis (i.e., these analyses are marked in the script file name with the suffix "ga"), enter:
```
python REPO_ROOT/cogitate-msp1/meeg/ANALYSIS_FOLDER/ANALYSIS_CODE.py
```

## List of analysis and corresponding run time
- activation:
**Individual-level analysis run time ~= 60min per participant**
**Group-level analysis run time ~= 240min**
- connectivity
**Individual-level analysis run time ~= 90min per participant**
**Group-level analysis run time ~= 30min**
- ged (to be run before the connectivtiy analysis)
**Individual-level analysis run time ~= 15min per participant**
**Group-level analysis run time ~= 10min**
- roi_mvpa
**Individual-level analysis run time ~= XXmin per participant**
**Group-level analysis run time ~= XXmin**
- source_modelling
**Individual-level analysis run time ~= 210min per participant**
**Group-level analysis run time ~= 60min**
