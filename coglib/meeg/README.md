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
