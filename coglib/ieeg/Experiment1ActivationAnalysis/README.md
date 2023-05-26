# Experiment1ActivationAnalysis
This folder contains code to perform the univariate duration analysis

Contributors: Alex Lepauvre, Katarina Bendtz, Simon Henin

## How to run:
There are several different subanalysis in this folder to investigate the neural activation associated with sustained
perception in a univariate fashion. 

###linear_mixed_model_master.py
This script averages single trials data in specified time windows and perform linear mixed models according to the 
different models in the config.
Under /lmm_configs, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). Add the list of the subjects you want to investigate at line 219. 
You should then open a command line and enter the following:
```
python Experiment1ActivationAnalysis/linear_mixed_model_master.py --config "path_to_your_config/lmm_configs/your_config.json"
```

###duration_tracking_master.py
This script computes for each channel a threshold to discriminate "activated-state" from "non-activated-state" and 
for each trial computes for how long the amplitude is above the activation threshold and compares this to the duration 
for which a stimulus was present on the screen for this specific trial. A channel will be considered to accurately 
tracking duration if the difference between the stim duration and the "activation duration" is less than a specified 
threshold (+-150ms). This should therefore detects the channels that remains activated for the duration of the stim
and therefore fulfill IIT predictions.
Under /duration_tracking_config, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). Add the list of the subjects you want to investigate at line 215. 
You should then open a command line and enter the following:
```
python Experiment1ActivationAnalysis/duration_tracking_master.py --config "path_to_your_config/duration_tracking_config/your_config.json"
```

###onset_offset_master.py
This script crops single trials data in a time window following stimulus onset and offset. A sliding t-test is computed 
between stimulus onset vs baseline and stimulus offset vs baseline (in that case, defined as pre-offset period). 
A conjunction is then performed to see which channels show both an activation onset following stimulus onset and offset,
which show only onset, which show only offset. This is performed specifically on frontal regions to find channels that 
fulfill GNW predictions (showing ignition both at onset and offsets) as well as channels that show only an onset and 
channels showing only an offset, for sake of completeness (maybe a channel showing only offset bursts could be 
interesting for GNW as there could be a distributed code in PFC, different channels showing an offset than those
showing an onset). 
Under /onset_offset_config, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). Add the list of the subjects you want to investigate at line 189. 
You should then open a command line and enter the following:
```
python Experiment1ActivationAnalysis/onset_offset_master.py --config "path_to_your_config/onset_offset_config/your_config.json"
```

## Files and their use

### linear_mixed_model_master.py
Scripts handling the call of the different functions in the right order to perform the linear mixed modelling. First, 
this script calls the activation_analysis_parameters_class to create the parameters object. It then loads the epoched 
data of the passed subjects. The epoched data are then averaged in the relevant time windows (0.8-1.0, 1.3-1.5, 1.8-2.0 
or whatever it is you pass in the config json) and converted to a data frame. Then, the different selected linear mixed
models set in the config are fitted to each channel. The different models are then compared using the criterion set
in the config file (BIC or AIC) to find the model explaining the data the best. If one of the model of one specific 
theory explain the data the best, then this channel is considered to be consistent with the theories predictions. All of
this is done using functions found under activation_analysis_helper_function

### duration_tracking_master.py
Scripts handling the call of the different functions in the right order to perform the duration tracking analysis. 
First, this script calls the activation_analysis_parameters_class to create the parameters object. 
It then loads the epoched data of the passed subjects. For each channel, an activation threshold is computed as the 
median between the measured activation in the baseline period and in a time window in the longest trials closest toward
the end of the stimulus. Then, for every trial, the time point at which the measured activation drops below the 
threshold is computed. This time point is compared to the time point at which stimulus offset occurs. If the difference
between the two is found to be less than +-150ms (or whatever is set in the config file), the trial is considered to be 
accurately tracked. Then, for every channel, the proportion of trials for which there is accurate duration tracking is 
computed. Then, the same procedure is repeated 10 000 times, shuffling the duration labels to generate a null 
distribution to estimate whether the proportion of duration tracking trials is significant. A moving average can be 
applied first to smooth the data. All the functions to perform these operations can be found under 
activation_analysis_helper_function

### onset_offset_master.py
Scripts handling the call of the different functions in the right order to perform the onset offset analysis. 
First, this script calls the activation_analysis_parameters_class to create the parameters object. 
It then loads the epoched data of the passed subjects time locked to the stimulus onset and offset. For each, the 
data are cropped (not averaged) in specified time windows (-200 to -0 and 300-500ms following stim onset and offset).
A sliding t-test is performed, giving a p-value for each time point (can be FDR corrected or not). The number of 
successive samples with successive samples is then compared to a threshold (say 50ms). If and only if more than 50ms 
of data are significantly different are the differences considered significant. Since this is done for each channel with
onset and offset, one can isolate channels showing only onset, only offset or both, highlighting channels consistent 
with gnw or partially so. All of this is done using functions found under activation_analysis_helper_function

###plot_lmm_results.py
Can be called standalone but is called at the end of the linear mixed modelling script to plot the results of the linear
mixed model, ordering things as a function of which model was significant. 

###plot_duration_tracking_results.py
Can be called standalone but is called at the end of the duration tracking script to plot the results.

###plot_onset_offset_results.py
Can be called standalone but is called at the end of the onset offset script to plot the results.

### lmm_job.sh
Slurm job for lmm. It basically executes 
```
python Experiment1ActivationAnalysis/linear_mixed_model_master.py --config "${config}"
```
with a specified passed config as a slurm job that will be scheduled and executed as soon as possible.

### duration_tracking_job.sh
Slurm job for duration tracking. It basically executes 
```
python Experiment1ActivationAnalysis/duration_tracking_master.py --config "${config}"
```
with a specified passed config as a slurm job that will be scheduled and executed as soon as possible.

### onset_offset_job.sh
Slurm job for onset offset. It basically executes 
```
python Experiment1ActivationAnalysis/onset_offset_master.py --config "${config}"
```
with a specified passed config as a slurm job that will be scheduled and executed as soon as possible.

### activation_analysis_batch_runner.py
This function enables to run the different master functions described  above as slurm jobs, listing all the 
config json found in the respective analyses config folders.

