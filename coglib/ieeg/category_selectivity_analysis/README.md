# category_selectivity_analysis
This folder contains code to identify category selective electrodes from a visual experiment.

Contributors: Alex Lepauvre, Katarina Bendtz, Simon Henin

## How to run:
Under /super_subject_config, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). In category_selectivity_master, you should add a list of subjects to run the analysis on at the 
line 185. You should then open a command line and enter the following:
```
python category_selectivity_analysis/category_selectivity_master.py --config "path_to_your_config/your_config.json"
```

## Files and their use

### category_selectivity_master.py
Scripts handling the different processes to compute the category selectivity. It will first create the parameters
object from the config file with the visual_responsivness_parameters_class.py, then load the data of the different 
subjects passed in the input. It will then call several different functions from 
category_selectivity_parameters_class.py to format the data, average the data, compute different statistical tests...
And it will finally save the different outputs: data (the data that went into the statistical test in the exact same 
format), results (the actual results of the test) and figures. 

###category_selectivity_parameters_class.py
Class that reads in the parameters json to create an object that contains all the different parameters for the analysis.
It also comes with a couple useful methods. For example the object can be serialized and saved to file in one call...

### category_selectivity_helper_function.py
Contains all the functions performing the relevant operations for the visual responsiveness test. 

### plot_category_selectivity_results.py
Is called  at the very end of the category selectivity master but can also be called on its own. It loads
back the results of the category selectivity and plots the data in different ways to display the results

### category_selectivity_job.sh
Slurm job for visual responsiveness. It basically executes 
```
python category_selectivity_analysis/category_selectivity_master.py --config "${config}"
```
with a specified passed config as a slurm job that will be scheduled and executed as soon as possible.

### category_selectivity_batch_runner.py
This function lists all the files found under super_subject_config and create one job each.



