# Preprocessing
This folder contains code to perform the preprocessing

Contributors: Alex Lepauvre, Katarina Bendtz, Simon Henin

## How to run:
Under /configs, create a config file according to your own settings (make sure to set the path to your 
bids root correctly). You should then open a command line and enter the following:
```
python Preprpcessing/PreprocessingMaster.py "path_to_your_config/your_config.json" --subject "SUBID" --interactive
```

## Files and their use

### PreprocessingMaster.py
Scripts handling the call of the different preprocessing functions in the right order. First, this script calls
the PreprocessingParametersClass to create the parameters object. It then loads the data of the subject passed
as subject and creates a subject info object with the SubjectInfo class. This object keeps track of information that
are specific to the subject (bad channels...). The data are then downsampled to 512Hz and detrended.
It then performs the different preprocessing steps according to the order set in the config file. 
The --interactive flag controls whether the script is run on an interactive node or a compute node. Some of the 
preprocessing functions require manual input (i.e. plotting and checking which channels are bad manually). Moreover,
there are complexities with 3D plotting in python and MNE and while it can theoretically work on a headless server, we
have never been able to get that to work with Siegfried. If the preprocessing steps do not contain any steps requiring
interaction, it will run without problems on a compute node. But if you try to run with the functions requiring 
interactive on a compute node, those steps will be ignored and skipped. 

Right now, it is best to only run it with the interactive flag on an interactive node. I keep this function however just
in case we want to run the preprocessing as SLURM jobs in the future, but right now, not working so well

###PreprocessingParametersClass.py
Class that reads in the parameters json to create an object that contains all the different parameters for the analysis.
It also comes with a couple useful methods. For example the object can be serialized and saved to file in one call...

###SubjectInfo.py
Class that keeps track of subject specific information. 

### PreprocessingHelperFunctions.py
Contains all the functions performing the relevant operations for the preprocessing. 

###test
Contains a couple of unittests for the preprocessing functions that are especially risky (lot of data handling, 
indexing...). There are tests for laplace referencing, custom car, high gamma computations.



