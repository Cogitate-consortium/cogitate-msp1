# ECoG
This folder contains all the code created by Katarina Bendtz and Alex Lepauvre in the frame of the COGITATE project

## Setup:
Create a new conda environment by running the following:
```
conda env create --file=requirements_cogitate_ecog.yaml
```
The environments are tailored for Linux and the HPC, so some things might break a little if you use windows or Mac 
(not tested very thoroughly).

The different analyses work with a master script that then calls functions from different places. As the utilities
weren't set as packages, the entire repository must be set as a new PYTHONPATH for python to know it can search for
functions in the different folders. You should therefore execute:
```
export PYTHONPATH=$PYTHONPATH:YOUR_PATH/ECoG
```

## General architecture:
The repository is organized in folders corresponding to different analyses. In each folder, you will find a subfolder 
called *_config, containing json files which contains all the parameters required for a given analysis. These are being 
loaded by the master scripts (see below) to perform the analysis according to the parameters. The only exception is the 
general_helper_functions folder, which contains functions called by several analyses. The ending of the .py files
indicate the "function" of the script:

### *_master.py OR *Master.py:
Main scripts for a given analysis, calling functions from elsewhere (see below) to load the data, perform the 
computations, save the results and plot the results. These can be called in several ways:
*From the command line*: python {analyis}_master.py --"config.json"
*From a python shell or editor*: super_subject_rsa(["Subject1", "Subject2"], save_folder="super") while passing the 
config as a parameter
The scripts can be modified at the bottom to select subjects on whom to run as well as the name of the folder in which
to save the results, or by calling the scripts from a python shell directly with the corresponding parameters.

### *_helper_function.py:
Contains various functions relevant for a particular analysis and are being called by the master scripts.

### *_batch_runner:
Python scripts listing all the configs found in the config of each respective analyses and launching  SLURM jobs for 
each. In some analyses folders, there are several different analysis and controls. In this case, the batch runners
have flags controlling which ones to run. 

### plot_*:
Contains functions to plot the results of  a given analysis. These are automatically called at the end of the master 
scripts but can also be launched separately, provided the respective analysis was ran already.

### *_job.sh:
Slurms jobs to run the analysis on an HPC with scheduler.

### config file structure
The analyses parameters are controlled by the config file. To replicate our analysis, only the parameters in bold below
should be adjusted. 

- raw_root: The path to the raw data before bids conversion. This is not really relevant, this is just to keep track
of where the initial data live
- **BIDS_root**: root of the bids converted data. This is very important as it will let the scripts know where to load 
the data. Because we are adhering to our own set of conventions, it is sufficient to know the root to know where to 
fetch things
- session&task_name: these are BIDS fields that specify what session and what task to load
- preprocessing_folder&preprocess_steps: the preprocessing saves data in a specific preprocessing folder, corresponding 
to a specific preprocessing operation (i.e. the data right after notch filter, the data after epoching, for all 
analyses, this is set to epoching). But on top of that, the preprocessing script creates an additional subfolder to keep
track of which preprocessing steps were performed on the data before. This is very handy, because say you want to load 
the epoched data, but specifically those that had as preprocessing step: notch filter, followed by common average 
referencing, followed by epoching but without bad channels rejection, you can do that (assuming you ran preprocessing
with this combination of steps).
- Additional fields: there are a set of additional fields that is mostly consistent across analyses but might differ a 
little, though they are mostly self explanatory
- analysis_parameters: this sets all the run time parameters specific to the actual test you are running. So for example
you want to run the visual responsiveness specifically on the high gamma signal, using a wilcoxon signed rank test as
opposed to t-test, using AUC as opposed to averaging, this is where you will set it. The analysis parameter contains
one or several dictionaries. The name of the dictionary dictates the name of the folder in which the results will be 
saved. So say you want to run the visual responsiveness in two different ways, you can have the following:
```
"high_gamma": {
  "signal": "high_gamma",
  "baseline_correction": null,
  "baseline_time": [-0.375, -0.125],
  "crop_time": [-0.3, 2],
  "multitaper_parameters": null,
  "do_zscore": true,
  ...
}
"erp": {
  "signal": "erp",
  "baseline_correction": null,
  "baseline_time": [-0.375, -0.125],
  "crop_time": [-0.3, 2],
  "multitaper_parameters": null,
  "do_zscore": true,
  ...
}
```   
The master script will loop through both these set of parameters, for the first, it will save the results under 
/high_gamma for the one and under /erp for the other


## Single folders description:
The following folders each correspond to a specific processing step in the frame of the cogitate project

###data_preparation: 
Contains scripts that prepares the data coming from the different hospital to the BIDS format for
further processing steps. This includes loading the files, finding the triggers channels, extracting the triggers and 
aligning to the log files, sanity checks, setting of channels types and montage and BIDS conversion as a final step
###preprocessing: 
contains the script to clean the signal up (bad channels removal, notch filter, rereferencing), 
computations of signals of interest (HGB, ERP...) and epoching. In addition, the different scripts plot the electrodes 
on the brain surface
###visual_responsiveness_analysis: 
contains scripts to determine which channels are showing a significant response
following stimulus onset. This can be done in many different ways, several of which are implemented there. The relevant
channels are plotted in several different ways to highlight the significance.
###category_selectivity_analysis: 
contains the script to determine which electrode show a response tuning to a specific 
experimental category. The scripts enable to do so in 3 different ways, only one of which we are interested in in the 
cogitate project. The electrodes are further plotted in different ways to highlight the tuning.
###Experiment1ActivationAnalysis: 
this folder contains several different analyses that all investigate the theories
predictions regarding the neural activation associated with sustained perception. One script computes linear mixed 
model, averaging signal in predifined time windows and modelling the response with the theories predictions and 
additional factors. Another computes sliding t-test at onset and offset to test GNW onset offset prediction. Finally,
one script computes duration tracking, a method to see how accurately a given electrode tracks stimulus duration.
**Important**: To run the linear mixed model, a separate environment is required, as the pymer4 package that we use
is not compatible with the main environment. You should install this separate environment as follows:
```
conda env create --file=requirements_lmm.yaml
```

###rsa:
Contains functions computing correlation between vectors of channels in cross temporal fashion to investigate 
multivariate patterns associated with sustained perception.

###Decoding:
@Simon, you can add a quick description here if needed

###Synchrony:
@Simon, you can add a quick description here if needed


Other folders contain general functions
###freesurfer:
Contains different freesurfer functions that do different reconstructions and whatnot.
###general_helper_function:
Contains collections of functions that are called in several different places throughout the repo to perform specific 
tasks that are repeated. For example, loading the data has to be done throughout so there is a function that does that
in the specific way I want.
