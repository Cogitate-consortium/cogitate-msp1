# Decoding Analysis
This folder contains code to perform the decoding analyses

Contributors: Alex Lepauvre, Simon Henin

## How to run:
Under /configs, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). In decoding_master, you should add a list of subjects to run the analysis on or None to run analysis on all available subjects
You should then open a command line and enter the following:
```
python decoding/decoding_master.py --config "path_to_your_config/your_config.json"
```

## Files and their use

### decoding_master.py
Main decoding analysis script.

### decoding_master_job.sh
Slurm job for running decoding scripts on an HPC


## Control Analyses
### Robustness checks
#### decoding_robustness_test.py
Performs decoding analysis using 100 repeats, randomly selecting 200 electrodes from the pool of available electrodes
```
python decoding/decoding_robustness_test.py --config "path_to_your_config/your_config.json"
```

###  IIT vs. IIT+GNW analysis
#### decoding_control_iit_vs_iitgnw.py
Standalone script to run analysis to compare performance of classifier models using electrodes from IIT only -vs- IIT+GNW classifier models.
It only performs the analysis on basic decoding over time (e.g. using the 'decoding_category_roi_basic.json' config script), excluding cross-task generalization analyses

```
python decoding/decoding_control_iit_vs_iitgnw.py
```


## Batch Runner 
### decoding_batch_runner.py
Wrapper function to queue batch jobs for decoding analysis. It lists all the files found under configs and create one job each.
e.g., 

```
python decoding_batch_runner()
```
Also, useful for running robustness tests with robustness_checks=True inside the main function