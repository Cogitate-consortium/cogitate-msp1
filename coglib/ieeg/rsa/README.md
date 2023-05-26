# rsa
This folder contains code to perform the rsa

Contributors: Alex Lepauvre, Simon Henin

## How to run:
Under /super_subject_config, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). In rsa_master, you should add a list of subjects to run the analysis on at the 
line 214. You should then open a command line and enter the following:
```
python rsa/rsa_master.py --config "path_to_your_config/your_config.json"
```

## Files and their use

### rsa_master.py
Scripts handling the different processes to compute the rsa. It will first create the parameters
object from the config file with the rsa_parameters_class.py, then load the data of the different 
subjects passed in the input. It will then call several different functions from 
rsa_helper_functions.py to format the data, average the data, compute different things...
And it will finally save the different outputs: data (the data that went into the statistical test in the exact same 
format), the results , and figures. Note that because the computation times on this one are much heavier, the 
statstical test have been separated, under rsa_super_subject_statistics.py
This script is also a bit harder to read than normal, as parallelization had to be add to have reasonable times. 
This means that some function had to be extra nested to be able to call them as a bundle to have the parallelization 
work

###rsa_parameters_class.py
Class that reads in the parameters json to create an object that contains all the different parameters for the analysis.
It also comes with a couple useful methods. For example the object can be serialized and saved to file in one call...

### rsa_helper_functions.py
Contains all the functions performing the relevant operations for the rsa. 

### rsa_super_subject_statistics.py
Is called  at the very end of the rsa_master but can also be called on its own. It loads
back what was computed in rsa_master to perform the relevant statistical tests on the data. The computation of the
dependent variables take forever, but the stats are quick, which is why I have splitted things down. This scripts
also calls a lot of functions from rsa_helper_functions and performs the plotting as well.

### rsa_super_subject_job.sh
Slurm job for rsa. It basically executes 
```
python rsa/rsa_master.py --config "${config}"
```
with a specified passed config as a slurm job that will be scheduled and executed as soon as possible.

### rsa_batch_runner.py
This function lists all the files found under super_subject_config and create one job each.

### plot_feature_optimization.py
This is a script that is not really in use anymore, but that I use to plot how the rsa values change as a function
of the number of selected features. It is a bit tricky though because with correlation, the lower the number of samples
the higher the correlations tend to get, there is a confound (i.e. very few features would seem as much better
but not truly the case, just artifact of correlation)

### config_archive
Collection of old configs. For RSA, we have to run so many combinations of things and the configs are so long, I kept
several that I thought might be useful in the future just in case. Can be deleted before publi



