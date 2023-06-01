# Behavior and Eye Tracking Procesisng and Analyses
This folder contains all the code created by Rony Hirschhorn, Csaba Kozma, and Abdo Sharaf for the COGITATE project's behavior and eye-tracking analyses

## Setup:
Create a new conda environment by running the following:
```
conda env create --file=requirements_cogitate_beh_et.yml
```
The environment was not tailored for Mac (might break).

## General architecture:
The repository is organized in folders corresponding to different analyses: one called "behavior" which contains all the scripts to execute and plot the behavioral data pre-processing and analysis, and one called "eyetracking" to execute and plot all the eyetracking data preprocessing and analysis.
Notably, all pre-processing scripts are Python scripts, and the analyses are one R script each (one for the behavioral analysis and one for the et analysis).

### behavior

#### quality_checks.py
This is the script running all the behavioral data reading and QCs. It uses data_reader.py to load the data, data_saver.py to save the processed data, and quality_checks_criteria.py to test whether the processed data is valid.

#### exp1_lmms.R
All the modelling of the grouped, processed data for Eyelink-based subject data is done in this script. 

### eyetracking

#### ET_qc_manager.py
This is the script running all the ET data reading and QCs for Eyelink-based data (not TOBII, that is a separate script, see below). 
The QC manager uses ET_data_extraction.py (which utilizes DataParser.py) to extract and parse the raw ascii files (converted, anonymized versions of the EDF files usually outputted from Eyelink devices).
In this process, it parases blinks (using the based_noise_blinks_detection.py), saccades, and fixations. The AnalysisHelpers.py is used mainly for side-functions calculating degrees visual angle from screen and distance data.
The ET_qc_manager.py script has fixed variables and information that is related to the experiment, used in the ET processing.

Then, all subject's fixation, blink and saccade data is aggregated using the ET_data_processing.py script. This is the place that aggregates all data types at the group level, and utilizes plotter.py to generate all the plots.

#### exp1_et_lmms.R
All the modelling of the grouped, processed data for Eyelink-based subject data is done in this script. 

#### tobii_et_handler_matlab_limited.py
For the TOBII-data subjects, there is only coordinate data - without pupil diameter, information about blinks and so on. That, combined with many periods of missing data, does not allow for proper identification of blinks, saccades and fixations. 
Therefore, this data is used only for sanity-check to see that subjects looked at the screen center during the experiment's trials. The data is loaded, parsed to trials, BL corrected, and then plotted in a heatmap. 
It is not part of the analysis, and cannot be loaded into the R script. 