# Behavior and Eye Tracking Processing and Analyses
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

Notably, as eye-tracking data parsing and processing relies on segmenting the data into trials, the behavioral processing must be done first before performing the ET data processing. This is important, as a sanity check compares ET triggers with behavioral record logs - a comparison which cannot be made unless quality_checks.py was already run.

#### ET_qc_manager.py
This is the script running all the ET data reading and QCs for Eyelink-based data (not TOBII, that is a separate script, see below). 
The QC manager uses ET_data_extraction.py (which utilizes DataParser.py) to extract and parse the raw ascii files (converted, anonymized versions of the EDF files usually outputted from Eyelink devices).
In this process, it parases blinks (using the based_noise_blinks_detection.py), saccades, and fixations. The AnalysisHelpers.py is used mainly for side-functions calculating degrees visual angle from screen and distance data.
The ET_qc_manager.py script has fixed variables and information that is related to the experiment, used in the ET processing.

Then, all subject's fixation, blink and saccade data is aggregated using the ET_data_processing.py script. This is the place that aggregates all data types at the group level, and utilizes plotter.py to generate all the plots.

#### pre-processing steps explained
- Step 1: Ascii files are aggregated, event triggers are identified and samples are transformed into workable tables. Lab-based parameters (screen size, resolution, distance from screen) are saved for each subject (information was given by the labs at the testing stage and we rely on that - this is derived from the param_manager script). 
- Step 2: Blink identification: as all future taggings (fixations, saccades) depend on their overlap with blinks, we start with blink identification. We use the algorithm by Hershman et al which identifies blinks based on pupillometric noise. We use the Python code that accompanies their paper to do that. Therefore, real blink (not just missing data) = Hershman blinks
Hershman, R., Henik, A., & Cohen, N. (2018). A novel blink detection method based on pupillometry noise. Behavior research methods, 50, 107-114.
- Step 3: We then run the algorithm for saccade detection, originally suggested by Engbert & Kliegl (2003), with the parameters from Engbert & Mergenthaler (2006) (as they say that’s better). Notably, saccades identified by this algorithm that overlap with a padded blink - are not treated as real saccades and accordingly are not included. Therefore, Real saccades = E & K (E & M params) saccades that do not overlap with a padded real blink. 
Engbert, R., & Mergenthaler, K. (2006). Microsaccades are triggered by low retinal image slip. Proceedings of the National Academy of Sciences, 103(18), 7192-7197.
Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. Vision research, 43(9), 1035-1045.
- Step 4: then, fixations are identified according to Eyelink tagging, as long as they do not overlap with padded real blinks.
Real fixations = Eyelink (SR research, Eyelink parser) fixations that do not overlap with a padded real blink. 
- Step 5: then, we do the same for pupils: we nullify pupil size that overlaps with a padded blink, so that it won’t affect future pupil size analyses. 
Real pupil = Eyelink-reported pupil size (arbitrary units), that do not overlap with a padded real blink.
- Step 6: at this point in time, each sample is tagged and accounted for - we identified real blinks, saccades, fixations, and pupil. Now we are adding the behavioral data into the gaze data in two ways: First, based on the EyeLink triggers - each sample is tagged based on the trigger events that were saved on the Eyelink log (miniblock number, stimulus features etc).
Then, this data is compared to the behavioral data. I then compare the behavioral data to the ET trigger data. I match each behavioral trial information with its corresponding gaze data (not the other way around, as in some cases triggers were missing!). This means that for each subject, only trials that had corresponding triggers were analyzed in the ET analysis (because these are the ones we can account for). This constitutes the vast majority of trials (only very few triggers (<10) were missed). 


#### exp1_et_lmms.R
All the modelling of the grouped, processed data for Eyelink-based subject data is done in this script. 

#### tobii_et_handler_matlab_limited.py
For the TOBII-data subjects, there is only coordinate data - without pupil diameter, information about blinks and so on. That, combined with many periods of missing data, does not allow for proper identification of blinks, saccades and fixations. 
Therefore, this data is used only for sanity-check to see that subjects looked at the screen center during the experiment's trials. The data is loaded, parsed to trials, BL corrected, and then plotted in a heatmap. 
It is not part of the analysis, and cannot be loaded into the R script. 


### Information
- [Rony Hirschhorn](https://github.com/RonyHirsch), Sagol School of Neuroscience, Tel-Aviv University
- [Csaba Kozma](https://github.com/csaba-a), Newcastle University
- [Abdo Sharaf](https://github.com/AbdoSharaf98), Georgia Institute of Technology
- Center PI: Prof. Liad Mudrik, Sagol School of Neuroscience and School of Psychological Sciences, Tel-Aviv University

Contact: rony.hirschhorn[at]gmail[dot]com // mudrikli[at]tauex[dot]tau[dot]ac[dot]il
