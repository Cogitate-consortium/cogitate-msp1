#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 1st, 2nd or 3rd level analyses using FSL FEAT.

First, fsf files are created based on template fsf files (one per analysis).
For each analysis from these templates one fsf file is created for each 
participant and run (1st level only).
Analyses are defined in analysis_definitions_* dicts, including task and output 
space definitions.
    analysis_definitions_1stLevel: analysis definitons for first level feat 
    analyses (run level)
    analysis_definitions_2ndLevel: analysis definitons for second level feat 
    analyses (subject level)
    analysis_definitions_3rdLevel: analysis definitons for third level feat 
    analyses (group level)

Second, feat analyses are submitted as separate jobs using the created fsf 
files.

Assumed are:
    1. a bids compliant dataset preprocessed with fmriprep, following default 
    fmriprep output folder structures and filenames
    2. fsf templates per analysis/task type and output space (e.g. 
    sub-PXX_DurXX_analysis-2ndQC_space-T1w.fsf)
        - fsf templates are further assumed to contain place holders for: 
        A: subject IDs (sub-PXX), 
        B: number of volumes (XXX_NTPS; only 1st level analyses) 
        C: task/run place holders that are task specific (e.g. DurXX for 
        task-Dur of exp.1)
        D: analysis space place holder (space-SXX) defining the analysis space 
        (e.g. T1w, MNI152NLin2009cAsym)
        Analysis labels + suffix must correspond to the labels defined in 
        analysis_definitions_* below.

Handles 2nd level feat analyses for 1st level feat dirs which are already
normalized/registered to the desired template (MNI or T1); i.e., bypasses
2nd level normalization usually run by feat. Note: it is essential that 
1st level feat dirs are NOT modified by parallel 2nd level analyses! Use
time_wait_after_analysis to set a delay before submitting subsequent 2nd level
analyses.
Also handles 2nd level feat analyses which may involve contrasts that do not
contain trials of that contrast in each 1st level analysis; i.e. removes
these feat dirs from 2nd level analysis, if so specificied (missingCon).
This option is particularly relevant for the task-independent visual processing
analysis as it requires analysis of target trials per stimulus category which
do not contain at least 1 trial in each run. Hence these analyses have to be
defined as separate 2nd level analyses with the missingCon option enabled.

Group level (3rd level) feat analyses can be defined for many copes of the same
2nd level gfeat. Applicable subjects must (currently) be defined in fsf 
template.

Additional custom group level (3rd level) feat analysis option added for 
task-independent visual processing analysis with special fsf file handling.

Requires FSL functions!
I.e., in the shell run, before executing the python script:
module load FSL

@author: David Richter, Yamil Vidal
Email: hvidaldossantos@gmail.com
Created on Fri Mar 29 15:11:06 2021
"""

import os, time, sys
import pandas as pd


#%% Paths and Parameters

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# FSL output path 
fsl_output_dir = bids_dir + '/derivatives/fslFeat'

# Determine which analyses are run:
# Set to 1 to run 1st level analyses; i.e. run level (defined in analysis_definitions_1stLevel)
# Set to 2 to run 2nd level analyses; i.e. across runs (defined in analysis_definitions_2ndLevel)
# Set to 3 to run 3nd level analyses; i.e. across subjects (defined in analysis_definitions_3rdLevel)
run_analysis = 3

# Whether to submit feat jobs using qsub. If False only fsf files are created
submit_jobs = True

# Optional wait time between submitting successive jobs. This can be useful to 
# avoid too many parallel jobs due to FEAT's large temporary files, if storage 
# space is a concern
wait_between_jobs_in_sec = 600

# Wait time after successive analyses. Useful for 2nd level analyses modifying 
# the same 1st level feat dir (i.e. for 2nd level analyses where this applies; 
# this should be at least the duration of the 2nd level 
# analyses to finish before proceeding!)
time_wait_after_analysis = 600

# Which subject list to use (set to 'all' to process all availabel participants; 'phase2_2.3' to process only those of phase 2 2/3 (list from DMT)))
subject_list_type = 'phase3_V1'
#subject_list_type = 'phase2_V1'
#subject_list_type = 'debug'


# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/code'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import get_subject_list


#%% Define analyses to be performed !
"""
Dict per analysis, specifying analysis label, suffix and number of runs. 
Analysis labels must match the fsf file template names!

desc: (optional) description of analysis
label: determines the analysis label, including session (ses-xx) and task 
    (e.g. task-Dur) key-value pairs to be analyzed
suffix: suffix for the analysis label; should describe the type of analysis 
    (e.g. QC = quality control) and space (e.g. T1w = T1 weighted space) 
    key-value pair. Space has to correspond to an output space of fmriprep
    2nd level analyses are assumed to be flagged by "analysis-2nd..." key/value
    THIS DETERMINES WHETHER AN ANALYSIS IS CONSIDERED TO BE 1st OR 2nd LEVEL!
fsf_file: path to the fsf file templates (assumed to be located in bids dir).
    The fsf file templates must match the analysis label + suffix 
runs: number of runs of this type. Note: 2nd level analyses do not have a run 
    key/value pair!
copes: cope labels/numbers used exclusively for 3rd (group) level analyses.
walltime: hours required to process run (submitted as job)
memory: gb memory required to process run (submitted as job)
missingCon: (optional) whether to check 1st level contrasts for runs with 
    missing data (e.g. some exp1 runs do not contain target face or target 
    letter trials) and if applicable remove these runs from 2nd level analysis
    (required by feat). Only applies for 2nd level analyses.
"""

# First level analyses (run level)
analysis_definitions_1stLevel = {
        # 'ROI_EVCLoc_T1w': {
        # 'desc'      : '1st level ROI definition analysis for EVC localizer exp2 data',
        # 'label'     : 'ses-V2_task-EVCLoc_run-',
        # 'suffix'    : 'analysis-1stROI_space-T1w',
        # 'runs'      : 1,
        # 'fsf_file'  : '/code/glm/fsf_templates/1st_level',
        # 'walltime'  : 3,
        # 'memory'    : 3
        # },
        'GLM_Dur_MNI': {
        'desc'      : '1st level GLM for exp 1',
        'label'     : 'ses-V1_task-Dur_run-',
        'suffix'    : 'analysis-1stGLM_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/1st_level',
        'walltime'  : 8,
        'memory'    : 8
        }
        }

# Second level analyses (subject level)  
analysis_definitions_2ndLevel = {
        'GLM_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_TarFace_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (only target face contrast; separate processing necessary since some runs do not contain trials for this contrast, hence must be excluded from 2nd level in feat)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_TarFace_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8,
        'missingCon': 1,
        },
        'GLM_TarObject_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (only target object contrast; separate processing necessary since some runs do not contain trials for this contrast, hence must be excluded from 2nd level in feat)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_TarObject_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8,
        'missingCon': 1,
        },
        'GLM_TarLetter_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (only target letter contrast; separate processing necessary since some runs do not contain trials for this contrast, hence must be excluded from 2nd level in feat)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_TarLetter_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8,
        'missingCon': 1,
        },
        'GLM_TarFalseFont_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (only target false font contrast; separate processing necessary since some runs do not contain trials for this contrast, hence must be excluded from 2nd level in feat)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_TarFalseFont_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8,
        'missingCon': 1,
        },
        'GLM_leave_run1_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 1 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run1_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run2_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 2 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run2_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run3_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 3 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run3_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run4_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 4 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run4_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run5_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 5 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run5_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run6_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 6 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run6_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run7_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 7 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run7_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        },
        'GLM_leave_run8_out_Dur_MNI': {
        'desc'      : '2nd level GLM for exp 1 (all contrasts, leaving run 8 out)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-2ndGLM_leave_run8_out_space-MNI152NLin2009cAsym',
        'runs'      : 8,
        'fsf_file'  : '/code/glm/fsf_templates/2nd_level',
        'walltime'  : 8,
        'memory'    : 8
        }
        }

# Third level analyses (group level)
analysis_definitions_3rdLevel = {
        'GLM_Dur_MNI': {
        'desc'      : '3rd (group) level GLM for exp 1 (all contrasts, except for target stimuli)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-3rdGLM_space-MNI152NLin2009cAsym',
        'copes'     : ['5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'],
        'fsf_file'  : '/code/glm/fsf_templates/3rd_level',
        'walltime'  : 3,
        'memory'    : 8
        },
        'GLM_TarFace_Dur_MNI': {
        'desc'      : '3rd (group) level GLM for exp 1 (only target face contrast)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-3rdGLM_TarFace_space-MNI152NLin2009cAsym',
        'copes'     : ['1'],
        'fsf_file'  : '/code/glm/fsf_templates/3rd_level',
        'walltime'  : 3,
        'memory'    : 8
        },
        'GLM_TarObject_Dur_MNI': {
        'desc'      : '3rd (group) level GLM for exp 1 (only target object contrast)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-3rdGLM_TarObject_space-MNI152NLin2009cAsym',
        'copes'     : ['2'],
        'fsf_file'  : '/code/glm/fsf_templates/3rd_level',
        'walltime'  : 3,
        'memory'    : 8
        },
        'GLM_TarLetter_Dur_MNI': {
        'desc'      : '3rd (group) level GLM for exp 1 (only target letter contrast)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-3rdGLM_TarLetter_space-MNI152NLin2009cAsym',
        'copes'     : ['3'],
        'fsf_file'  : '/code/glm/fsf_templates/3rd_level',
        'walltime'  : 3,
        'memory'    : 8
        },
        'GLM_TarFalseFont_Dur_MNI': {
        'desc'      : '3rd (group) level GLM for exp 1 (only target false font contrast)',
        'label'     : 'ses-V1_task-Dur',
        'suffix'    : 'analysis-3rdGLM_TarFalseFont_space-MNI152NLin2009cAsym',
        'copes'     : ['4'],
        'fsf_file'  : '/code/glm/fsf_templates/3rd_level',
        'walltime'  : 3,
        'memory'    : 8
        }
        }


# get analysis definitions
if run_analysis == 1:
    analysis_definitions = analysis_definitions_1stLevel
elif run_analysis == 2:
    analysis_definitions = analysis_definitions_2ndLevel
elif run_analysis == 3:
    analysis_definitions = analysis_definitions_3rdLevel


# %% Functions returning analysis names/types and space and paths
def get_fmriprep_processed_nifti_file(bids_dir, sub, session_label, analysis, space):
    """
    get nifti file name, given path + pattern (defined here), and input parameters
    bids_dir = bids directory
    sub: subject ID (e.g. SC101)
    session_label: session label ("key-value" pair; e.g. ses-V1)
    analysis: analysis label (e.g. task-Dur)
    space: analysis space (e.g. T1w)
    Returns nifti file path+filename
    """
    # nifti data path
    nifti_path_pattern = bids_dir + os.sep + 'derivatives' + os.sep + 'fmriprep' + os.sep + '%(sub)s' + os.sep + '%(ses)s' + os.sep + 'func' + os.sep
    # nifti file suffix (appended after subject ID and analysis/task label)
    nifti_file_pattern = '%(sub)s_%(analysis)s_%(space)s_desc-preproc_bold.nii.gz'
    nifti_file = nifti_path_pattern%{'sub':sub, 'ses':session_label} + nifti_file_pattern%{'sub':sub, 'analysis':analysis, 'space':space}
    return nifti_file

def get_analysis_labels(analysis_dict):
    """
    combine analysis label, analysis suffix and run numbers to for analyses 
    labels per run (if 1st level). Only return analysis label + suffix for 
    higher (2nd level) analyses.
    Returns:
    analyses: analysis labels
    analyses_suffix: analysis suffix
    fsf_templates_dir: path to fsf template
    run_missing_contrast_check: whether or not to run missing contrast check before 2nd level
    """
    analyses = []
    analyses_suffix = analysis_dict['suffix']
    fsf_templates_dir = bids_dir + analysis_dict['fsf_file']
    # process first level (i.e. not 2nd or 3rd level) analyses
    if 'analysis-1st' in analyses_suffix:
        for run in range(analysis_dict['runs']):
            analyses.append(analysis_dict['label'] + str(run+1))
    # process 2nd level analyses
    else:
        analyses.append(analysis_dict['label'])
    # check if missingCon key exists (indicating that additional processing before 2nd level is required)
    run_missing_contrast_check = 'missingCon' in analysis_dict
    return analyses, analyses_suffix, fsf_templates_dir, run_missing_contrast_check

def write_error_log(bids_dir, error_df):
    """
    write log file listing possible errors encountered during job submission
    """
    timestr = time.strftime("date-%Y%m%d_time-%H%M%S")
    file_name = os.sep + 'log_job_submission_' + str(run_analysis) + '-level_' + timestr + '.csv'
    output_dir = bids_dir + os.sep + 'derivatives' + os.sep + 'fslFeat'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fname = output_dir + file_name
    error_df.to_csv(fname, sep=',', index=False, na_rep='Null')
    print('Error saved to ' + output_dir)


#%% functions to modify fsf templates 
def replace_text(fname, string_to_replace, replacement_string):
    """
    replace text (string_to_replace) with new string (replacement_string) in 
    input file (fname)
    """
    import fileinput
    with fileinput.FileInput(fname, inplace=True) as file:
        for line in file:
            print(line.replace(string_to_replace, replacement_string), end='')
            
def get_n_volumes(nifti_file):
    """
    get number of volumes in input nifti file
    returns number of volumes (v_vols)
    """
    import subprocess
    full_cmd = ['fslhd', nifti_file]
    # run fslhd to get number of volumes
    result = subprocess.run(full_cmd, stdout=subprocess.PIPE)
    fslHd_output = result.stdout.decode('utf-8')
    # extract only n vols (dim4)
    dim4_target_string = '\ndim4\t\t'
    n_vols = fslHd_output[fslHd_output.find(dim4_target_string)+len(dim4_target_string):fslHd_output.find('\ndim5')]
    return n_vols

def make_this_fsf_template(fsf_file, sub, analysis, nifti_file, place_holder, space):
    """
    Take fsf template and replace subject, analysis and number of volumes 
    place holders with args below. 
    Subject ID place holder is assumed to be PXX number of volumes XXX_NTPS and 
    analysis place holders are the task key's value + XX.
    fsf_file: fsf file name
    sub: subject ID (can be empty for 3rd level analyses). Assumed to contain 
    sub key-value pair
    analysis: analysis label or cope label (for 3rd level analyses)
    nifti_file: nifti file 
    place_holder: string of analysis place holder in template
    space: target analysis space (key-value pair; e.g. space-T1w)
    """
    print(' . Creating fsf files for analysis: ' + analysis)
    # replace PXX in fsf template with subject ID
    replace_text(fsf_file, 'PXX', sub[4::])
    # replace XXX_NTPS in fsf template with number of volumes
    if not nifti_file == 'is_2nd_level':
        n_vols = get_n_volumes(nifti_file)
        replace_text(fsf_file, 'XXX_NTPS', n_vols)
    # replace run/analysis place holder with analysis
    replace_text(fsf_file, place_holder, analysis)
    # replace analysis space place holder with target space
    replace_text(fsf_file, 'space-SXX', space)
    

# %% function to modify 2nd level fsf template in case of missing contrasts in first level feat dirs, necessary if some contrasts have no trails in some runs (e.g. Dur Targets)  
def adjust_fsf_template_for_missing_contrasts(fsf_file):
    """
    Adjust 2nd level feat fsf file such that runs with empty contrasts are 
    removed from analysis. First gets list of 1st level dirs and list of 
    contrast labels. Then checks for the current contrast of interest (defined 
    by the fsf_file namen) whether any runs have an empty contrast of interest. 
    If so removes these runs from the 2nd level contrast definitions in the fsf 
    file. Note: the 2nd level fsf template must contain the same label for the
    contrast of interest as the 1st level contrast name. Thus, if the contrast
    on the first level is called TarFace (experiment 1, target, face) then the 
    2nd level fsf file template must contain TarFace in its filename.
    fsf_file: path and fname of fsf file to be edited
    """
    # get first level feat dirs
    feat_1st_level_dirs = get_1st_level_feat_dirs_from_fsf_file(fsf_file)
    # get contrast labels
    contrast_labels = get_contrast_labels(feat_1st_level_dirs[0])
    # get currently relevant contrast based on contrast labels (from 1st level feat contrast names) and 2nd level fsf file name. Note: fsf template file names must contain the exact contrast name as specified in the first level feat analysis for the contrast of interest
    for con_no, con_label in contrast_labels.items():
        if con_label[0] in fsf_file:
            contrast_idx = con_no-1
    # get which runs for current contrast of interest have empty contrasts
    empty_runs = check_if_contrast_is_empty(feat_1st_level_dirs, contrast_idx)
    # remove empty runs from list of 1nd level fsf files
    valid_1st_level_dirs = get_only_valid_1st_level_feat_dirs(feat_1st_level_dirs, empty_runs)
    # set correct number of valid runs in fsf file
    n_valid_runs = str(len(valid_1st_level_dirs))
    replace_text(fsf_file, 'XXX_NRUNS', n_valid_runs)
    # replace all feat paths in 2nd level fsf file with place holders (to be replaced later)
    run_counter = 0
    for feat_dir in feat_1st_level_dirs:
        run_counter += 1
        replace_text(fsf_file, feat_dir, 'XXX_RUN'+str(run_counter))
    # replace place holders with valid runs
    run_counter = 0
    for valid_feat_dir in valid_1st_level_dirs:
        run_counter += 1
        replace_text(fsf_file, 'XXX_RUN'+str(run_counter), valid_feat_dir)
    # disable other references to invalid runs (technically not necessary as feat will ignore these, but it results in a cleaner design.fsf file)
    cleanup_fsf_file_with_missing_contrasts(fsf_file,feat_1st_level_dirs,valid_1st_level_dirs)
    
    
def cleanup_fsf_file_with_missing_contrasts(fsf_file,feat_1st_level_dirs,valid_1st_level_dirs):
    """
    Perform some additional cleanup on 2nd level fsf files with missing 
    contrasts by prepending # to unnecessary rows in the fsf file.
    fsf_file: path and fname of fsf file to be edited
    feat_1st_level_dirs: list of 1st level feat dirs
    valid_1st_level_dirs: list of only valid 1st level feat dirs (i.e. without)
    missing contrasts
    """
    runs_to_disable = len(feat_1st_level_dirs) - len(valid_1st_level_dirs)
    for run_idx in range(runs_to_disable):
        run_no = (len(feat_1st_level_dirs)-run_idx)
        # replace 3 target strings that are dependent on the number of runs
        # each string is prepended with a # to denote a comment and the first 
        # letter is removed to avoid accidental string finds looking e.g. for 
        # the run data
        string_to_replace = 'set fmri(groupmem.' + str(run_no) + ') 1'
        replace_text(fsf_file, string_to_replace, '# '+string_to_replace[1::])
        string_to_replace = 'set fmri(evg' + str(run_no) + '.1) 1'
        replace_text(fsf_file, string_to_replace, '# '+string_to_replace[1::])
        string_to_replace = 'set feat_files(' + str(run_no) + ')'
        replace_text(fsf_file, string_to_replace, '# '+string_to_replace[1::])
                     
def get_only_valid_1st_level_feat_dirs(feat_1st_level_dirs, empty_runs):
    """
    Get a list with 1st level feat dirs that do not contain runs without the 
    contrast of interest
    feat_1st_level_dirs: list of 1st level feat dirs
    empty_runs: list with runs that do not contain trials of the contrast of 
    interest
    Returns
    valid_1st_level_dirs: list with 1st level feat dirs that all contain runs
    for the contrast of interest (i.e. after removing empty runs from list)
    """
    valid_1st_level_dirs = []
    for feat_dir in feat_1st_level_dirs:
        cur_run = feat_dir[feat_dir.find('run-'):feat_dir.find('run-')+len('run-')+1]
        if not cur_run in empty_runs:
            valid_1st_level_dirs.append(feat_dir)
    return valid_1st_level_dirs   

def check_if_contrast_is_empty(feat_1st_level_dirs, contrast_idx):
    """
    Check if all 1st level feat dirs contain the current contrast of interest.
    Specifically, whether their design.con file contains a 1 for the contrast
    of interest. If not, this indicates that the run does not contain trials
    for any regressor forming the contrast of interest. Such empty runs are
    checked and returned in a list of empty_runs.
    feat_1st_level_dirs: list of 1st level feat dirs
    contrast_idx: index of the current contrast of interest in the list of 
    contrasts
    Returns 
    empty_runs: list with runs that do not contain trials of the contrast of 
    interest
    """
    empty_runs=[]
    for feat_dir in feat_1st_level_dirs:
        run_label = feat_dir[feat_dir.find('run-'):feat_dir.find('run-')+len('run-')+1]
        # load design.con file
        con_fname = feat_dir + os.sep + 'design.con'
        with open(con_fname, 'r') as file:
            con_text = file.read()
        # clean up text and get one row per contrast
        con_text_clean = con_text[con_text.find('/Matrix\n')+len('/Matrix\n')::]
        contrasts = con_text_clean.split('\n')
        if not '1' in contrasts[contrast_idx] and len(contrasts[contrast_idx])>0:
            empty_runs.append(run_label)
    return empty_runs

def get_contrast_labels(feat_dir):
    """
    Get labels and numbers of contrasts defined in design.con in feat dir
    feat_dir: path to a feat dir
    Returns
    contrast_labels: dict with constrast numbers and associated contrast labels
    """
    # load design.con file
    con_fname = feat_dir + os.sep + 'design.con'
    with open(con_fname, 'r') as file:
        con_text = file.read()
    con_text_list = con_text.split('\n')
    contrast_labels = {}
    for con_label in con_text_list:
        if 'ContrastName' in con_label:
            con_parts = con_label.split('\t')
            con_no = int(con_parts[0][con_label.find('ContrastName')+len('ContrastName')::])
            con_name = con_parts[1][0:-1]
            contrast_labels[con_no] = []
            contrast_labels[con_no].append(con_name)
    return contrast_labels

def get_1st_level_feat_dirs_from_fsf_file(fsf_file):
    """
    Get list of 1st level feat dir paths from 2nd level fsf file
    fsf_file: fsf file to be submitted as feat job
    Returns
    feat_1st_level_dirs: list of 1st level feat dirs
    """
    with open(fsf_file, 'r') as file:
        fsf_file_text = file.read()
    # check number of runs based on fsf file
    feat_1st_level_dirs = []
    check_run = 1
    while check_run < 20: # check for a max of 20 runs (all tasks have significantly less; even if combining tasks within a session)
        target_string = 'set feat_files(' + str(check_run) + ') '
        if target_string in fsf_file_text:
            feat_dir_1st_level = fsf_file_text[fsf_file_text.find(target_string) + len(target_string): fsf_file_text.find('# Add confound EVs text file')]
            feat_1st_level_dirs.append(feat_dir_1st_level[1:feat_dir_1st_level.find('"\n')])
        check_run += 1
    return feat_1st_level_dirs

def check_if_feat_output_exists(fsf_file):
    """
    Checks if feat output specified in fsf file already exists
    fsf_file: fsf file to be submitted as feat job
    Returns
    feat_dir_exists: bool; true if output already exists
    """
    with open(fsf_file, 'r') as file:
        fsf_file_text = file.read()
    # get output path
    target_string = 'set fmri(outputdir) "'
    feat_dir = fsf_file_text[fsf_file_text.find(target_string) + len(target_string):fsf_file_text.find('# TR(s)') - 3]
    feat_dir_exists = os.path.isdir(feat_dir)
    return feat_dir_exists
                      
# %% functions for submitting feat jobs 
def write_tmp_file_with_sbatch_cmd(fsf_file):
    """
    Create file with command to run feat on fsf file to be submitted as job
    Return path + name of tmp file
    """
    if 'analysis-3rd' in fsf_file:
        tmp_file = os.path.split(fsf_file)[0] + os.sep + 'tmpscript_' + fsf_file[fsf_file.find('cope'):fsf_file.find('.fsf')]
    else:
        tmp_file = os.path.split(fsf_file)[0] + os.sep + 'tmpscript_' + fsf_file[fsf_file.find('sub-'):fsf_file.find('.fsf')]
    file = open(tmp_file, 'w')
    # write sbatch command lines to beginning of script, then add feat and path to fsf file
    file.write('#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --partition=xnat\nfeat ' + fsf_file)
    file.write('\n')
    file.close()
    return tmp_file

def check_1st_level_feat_dirs(fsf_file):
    """
    Checks before 2nd level analyses whether 1st level feat dirs exist and 
    whether report.html files of 1st level contain no errors.
    fsf_file: path to fsf file
    Returns
    check_flag_feat_dirs: true if all 1st level feat dirs exist and have no 
    errors, false if not all dirs exist or any errors are detected in 
    report.html
    error_msg: lists error type and run number for errors detected in 1st level 
    feat dirs, if any
    """
    error_msg = []
    check_run = 0
    feat_1st_level_dirs = get_1st_level_feat_dirs_from_fsf_file(fsf_file)
    for feat_dir_1st_level in feat_1st_level_dirs:
        check_run += 1
        # throw warning if feat dir is not found and set flag to false
        if not os.path.isdir(feat_dir_1st_level):
            error_msg.append(['1st level feat dir not found',str(check_run+1)])
            print(' . ! Error for 2nd or 3rd level analysis! FEAT dir NOT FOUND for : ' + feat_dir_1st_level[feat_dir_1st_level.find('ses-')::] + ' ! ')
        #if feat dir exists check report log for errors
        else:
            # read report_log.html file
            report_log = feat_dir_1st_level + os.sep + 'report.html'
            with open(report_log, 'r') as file:
                data_log = file.read()
            # throw warning if error is found and set flag to false
            if ('Error' in data_log) or ('ERROR' in data_log):
                error_msg.append(['error in 1st level feat report',str(check_run+1)])
                print(' . ! Error for 2nd or 3rd level analysis! Error in FEAT log for  : ' + feat_dir_1st_level[feat_dir_1st_level.find('ses-')::] + ' !')
    check_flag_feat_dirs = not any(error_msg)
    return check_flag_feat_dirs, error_msg

def submit_feat_job(fsf_file,walltime,memory):
    """
    Submit feat jobs using fsf files, unless outputs already exist.
    In case of 2nd level analyses, also checks if 1st level feat reports 
    contain any errors and prepares 1st level feat dirs to avoid repeating 
    registration.
    fsf_file: fsf file to be submitted as feat job
    walltime: walltime in hours for job
    memory: memory in gb for job
    Returns
    error_msg: lists run and error type detected in 1st level feat dirs
    """
    # get expected output feat dir from fsf file
    output_dir_str = 'set fmri(outputdir) "'
    with open(fsf_file, 'r') as file:
        fsf_file_text = file.read()
    # get feat dir
    feat_dir = fsf_file_text[fsf_file_text.find(output_dir_str) + len(output_dir_str):fsf_file_text.find('# TR(s)') - 3]
    # check if output already exists, if so, skip
    if os.path.isdir(feat_dir):
        print(' . ! FEAT dir ALREADY EXISTS: ' + os.path.split(feat_dir)[1] + ' ! Skipping !')
        error_msg = [['_ Analysis output already existed - Skipped _', '_']]
    else:
        # if output doesn exist yet, prepare feat dir and submit jobs
        if 'analysis-2nd' in fsf_file:
            # if its a 2nd level analysis, make sure that all 1 level feat dirs exist and modify reg dir
            check_flag_feat_dirs, error_msg = check_1st_level_feat_dirs(fsf_file)
            if check_flag_feat_dirs:
                process_1st_level_reg_dirs_before_2nd_level(fsf_file)
        elif 'analysis-3rd' in fsf_file:
            # if its a 3rd level also check that all 2nd level dirs exists, but dont modify anything
            check_flag_feat_dirs, error_msg = check_1st_level_feat_dirs(fsf_file)
        else:
            error_msg = []
            check_flag_feat_dirs = True
        # check if no error in feat dirs has been detected or throw error 
        if check_flag_feat_dirs:
            # write tmp file to submit as job using qsub
            tmp_file = write_tmp_file_with_sbatch_cmd(fsf_file)
            submit_cmd = ['sbatch -N1 -n1 -t %(walltime)s:00:00 --mem=%(memory)sgb '%{'walltime':walltime,'memory':memory} + tmp_file]
            # submit job
            print(' . . Submitting FEAT job --> ' + tmp_file[tmp_file.find('tmpscript')::])
            os.system(submit_cmd[0])
            # wait after each submission
            time.sleep(wait_between_jobs_in_sec)
        else:
            print(' . ! Analysis NOT submitted ! Unexpected error ! Check inputs ')
    return error_msg


# %% functions for avoiding reregistration in feat before running 2nd level
def process_1st_level_reg_dirs_before_2nd_level(fsf_file):
    """
    Process 1st level feat/reg dirs before running 2nd level feat analysis to 
    avoid repeating registration during higher level feat analysis.
    Finds all 1st level feat dir associated with current 2nd level analysis 
    (fsf_file) then submits each 1st level dir to be modified.
    Assumes that bold data is already registered to common space (MNI or T1w)
    due to e.g. preprocessing data with fmriprep.
    fsf_file: 2nd level fsf file
    """
    # find 1st level feat dirs
    feat_1st_level_dirs = get_1st_level_feat_dirs_from_fsf_file(fsf_file)
    for feat_dir_1st_level in feat_1st_level_dirs:
        #modify 1st level feat dir
        modify_1st_level_reg_dir(feat_dir_1st_level)

def modify_1st_level_reg_dir(feat_dir_1st_level):
    """
    Modifies 1st level feat /reg dirs before running 2nd level feat analysis to 
    avoid repeating registration during higher level feat analysis.
    Removes any files that may result in repeated registration reg_standard dir
    and any .mat file in /reg. Then replaces example_func2standard with 
    identity matrix and reference image standard.nii.gz with mean_func.
    feat_dir_1st_level: path to first level feat
    """
    import shutil, glob, os
    print('. modifying /reg dir of 1st level feat: ' + feat_dir_1st_level[feat_dir_1st_level.find('fslFeat')+25::])
    reg_standard_dir = feat_dir_1st_level + os.sep + 'reg_standard'
    reg_dir = feat_dir_1st_level + os.sep + 'reg'
    # remove reg_standard and contents if it already exists
    if os.path.isdir(reg_standard_dir):
        shutil.rmtree(reg_standard_dir)
    # delete mat files in reg dir
    for f in glob.glob(reg_dir + os.sep + '*.mat'):
        os.remove(f)
    # copy idenitty matrix to reg dir
    fsl_dir = os.environ["FSLDIR"]
    ident_mat = fsl_dir + os.sep + 'etc' + os.sep + 'flirtsch' + os.sep + 'ident.mat'
    reg_mat = reg_dir + os.sep + 'example_func2standard.mat'
    shutil.copy(ident_mat,reg_mat)
    # delete existing standard image
    standard_image = reg_dir + os.sep + 'standard.nii.gz'
    if os.path.isfile(standard_image):
        os.remove(standard_image)
    # copy mean_func as new standard reference image
    mean_func = feat_dir_1st_level + os.sep + 'mean_func.nii.gz'
    shutil.copy(mean_func,standard_image)
    

# %% main function to loop over subjects & analyses
def run_fsf_creation_and_submit_feat_job(bids_dir, subjects, analyses, analyses_suffix, fsf_templates_dir, fsf_output_dir, run_missing_contrast_check, error_df, walltime, memory):
    """
    Run fsf file creation for all subjects and runs for current analysis specification
    bids_dir: path to bids folder
    subjects: subject list
    analyses: list of labels for all runs of current analysis, including task and session keys (e.g. ses-V2_task-Replay_run-1, ...)
    analyses_suffix: suffix specifying analysis further, including analysis type and space key (e.g. analysis-1stROI_space-T1w)
    fsf_templates_dir: path to fsf templates
    fsf_output_dir: path to fsf output dir
    run_missing_contrast_check: whether to run checks for missing contrasts on 1st level before running 2nd level analysis
    error_df: data frame for error logs during job submission
    walltime: walltime for feat jobs in hours
    memory: memory for feat jobs in gb
    """
    from shutil import copy as copyFile
    # loop over subjects
    for sub in subjects:
        print('Subject: ' + sub + ' | processing...')
        # loop over analyses (runs)
        for analysis in analyses:
            error_msg = []
            # get session label from analysis name
            session_label = analysis[analysis.find('ses-'):analysis.find('task-')-1]
            # get analysis space from analysis suffix
            space = analyses_suffix[analyses_suffix.find('space-')::]
            # check if it's first level analysis
            if 'analysis-1st' in analyses_suffix:
                # get nifti file name if it is not 2nd level analysis
                nifti_file = get_fmriprep_processed_nifti_file(bids_dir, sub, session_label, analysis, space)
                # get place holder in template based on analysis name
                place_holder = analysis[analysis.find('task-')+5:analysis.find('_run-')] + 'XX' 
            else:
                # if it is 2nd or 3rd level analysis set nifti file to 2nd level flag
                nifti_file = 'is_2nd_level'
                # get place holder in template based on analysis name
                place_holder = analysis[analysis.find('task-')+5::] + 'XX'
            place_holder_analysis_label = analyses_suffix[0:analyses_suffix.find('_space')]
            # check if nifti file exists, if so, create fsf file
            if os.path.isfile(nifti_file) or nifti_file == 'is_2nd_level':
                # create fsf copy of template for current subject & analysis
                input_file = fsf_templates_dir + os.sep + 'sub-PXX_' + place_holder + '_' + place_holder_analysis_label + '_space-SXX.fsf'
                fsf_file = fsf_output_dir + os.sep + sub + '_' + analysis + '_' + analyses_suffix + '.fsf'
                copyFile(input_file,fsf_file)
                # check if output already exists
                if check_if_feat_output_exists(fsf_file):
                    print(' ! Subject: ' + sub + ' | Analysis: ' + analysis + ' in ' + space + ' space. FEAT output already exists ! Skipping ! ')
                    continue
                # make fsf template for current subject & analysis
                make_this_fsf_template(fsf_file, sub, analysis, nifti_file, place_holder, space)
                # modify fsf template in case of runs missing contrasts
                if run_missing_contrast_check:
                    adjust_fsf_template_for_missing_contrasts(fsf_file)
                # submit feat job
                if submit_jobs:
                    error_msg = submit_feat_job(fsf_file, walltime, memory)
                # log error messages
                if any(error_msg):
                    for err_row in error_msg:
                        error_df.loc[len(error_df)] = [sub, analysis, analyses_suffix, 'run-' + err_row[1], err_row[0]]
                else:
                    error_df.loc[len(error_df)] = [sub, analysis, analyses_suffix, '-', '_ Job submitted _']
            else:
                print(' ! Subject: ' + sub + ' | Analysis: ' + analysis + ' in ' + space + ' space. Nifti FILE NOT FOUND !')
                error_msg = ['nifti file not found','x']
                error_df.loc[len(error_df)] = [sub, analysis, analyses_suffix, 'run-' + error_msg[1], error_msg[0]]


def run_fsf_creation_and_submit_feat_job_group_analyses(bids_dir, copes, analysis_prefix, analyses_suffix, fsf_templates_dir, fsf_output_dir, run_missing_contrast_check, error_df, walltime, memory):
    """
    Run fsf file creation for all subjects and runs for current analysis 
    specification. Uses modified functions for GLM analyses, detected
    based on GLM string in analyses_suffix.
    bids_dir: path to bids folder
    copes: copes list
    analysis_prefix: analysis_prefix label (prefix; e.g. ses-V1_task-Dur)
    analyses_suffix: suffix specifying analysis further, including analysis type and space key (e.g. analysis-3rdFIR_space-MNI152NLin2009cAsym)
    fsf_templates_dir: path to fsf templates
    fsf_output_dir: path to fsf output dir
    run_missing_contrast_check: whether to run checks for missing contrasts on 1st level before running 2nd level analysis
    error_df: data frame for error logs during job submission
    walltime: walltime for feat jobs in hours
    memory: memory for feat jobs in gb
    """
    from shutil import copy as copyFile
    # loop over copes
    for cope in copes:
        print('Cope: ' + cope + ' | processing...')
        cope = 'cope' + cope
        error_msg = []
        # place holder
        place_holder = 'copeXX'
        # get analysis space from analysis suffix
        space = analyses_suffix[analyses_suffix.find('space-')::]
        # analysis label
        analysis_label = analyses_suffix[0:analyses_suffix.find('_space')]
        # 3rd level analysis uses nifti file to 2nd level flag
        nifti_file = 'is_2nd_level'
        # create fsf copy of template for current subject & analysis
        if 'GLM' in analyses_suffix:
            input_file = fsf_templates_dir + os.sep + analysis_prefix + '_analysis-3rdGLMXX_space-SXX_desc-copeXX.fsf'
            fsf_file = fsf_output_dir + os.sep + analysis_prefix + '_' + analysis_label + '_' + space + '_desc-' + cope + '.fsf'
            copyFile(input_file,fsf_file)
            # make fsf template for current subject & analysis
            make_GLM_3rd_level_fsf_template(fsf_file, analysis_label, subjects, cope, space)
        else:
            input_file = fsf_templates_dir + os.sep + analysis_prefix + '_' + analysis_label + '_space-SXX' + '_desc-' + place_holder + '.fsf'
            fsf_file = fsf_output_dir + os.sep + analysis_prefix + '_' + analyses_suffix + '_desc-' + cope + '.fsf'
            copyFile(input_file,fsf_file)
            # make fsf template for current subject & analysis
            make_this_fsf_template(fsf_file, '', cope, nifti_file, place_holder, space)
        # check if output already exists
        if check_if_feat_output_exists(fsf_file):
            print(' ! Cope: ' + cope + ' | Analysis: ' + analysis_prefix + ' in ' + space + ' space. FEAT output already exists ! Skipping ! ')
            continue
        # submit feat job
        if submit_jobs:
            error_msg = submit_feat_job(fsf_file, walltime, memory)
        # log error messages
        if any(error_msg):
            for err_row in error_msg:
                error_df.loc[len(error_df)] = [cope, analysis_prefix, analyses_suffix, 'run-' + err_row[1], err_row[0]]
        else:
            error_df.loc[len(error_df)] = [cope, analysis_prefix, analyses_suffix, '-', '_ Job submitted _']
        

#%% functions specific for GLM analysis
def make_GLM_3rd_level_fsf_template(fsf_file, analysis_label, subjects, cope, space):
    """
    Take fsf template and replace analysis, number of subjects, cope and space 
    place holders with args below. This specific version handles fsf templates
    specific to the GLM group level (3rd level) analysis.
    Place holders in the fsf template file are assumed to be:
    - GLMXX: set to specific GLM analysis label; 
    e.g.   GLMTarFace
    - XXX_NTPS: set to number of subjects; e.g.:   22
    - copeXX: to number of 2nd level cope; e.g.:   cope1
    - feat_filesXX: full string per subject where 2nd feat copes are located 
    with additional string; e.g.:   set feat_files(1) ... .feat"
    	note this must also include the correct cope & analysis!
    - evgXX: full string per subject; for # Higher-level EV value for EV 1 and 
    input 3 set e.g.:   set fmri(evg3.1) 1 
    - groupmemXX: full string per subject; for 3rd subjects 
    e.g.:    set fmri(groupmem.3) 1
    - analysis_label: The analysis output/input label; must correspond to 2nd 
    level input analysis dir labels.
    fsf_file: fsf file name
    subjects: list of subject IDs to be used in the 3rd level analysis
    analysis_label: string of analysis label. Will be the 3rd level feat 
    analysis output label and must be the same name as of the 2nd level 
    analysis label; i.e. name of the analysis of the 2nd level feat dir 
    (e.g. GLMTarFace)
    cope: cope of interest; must correspond to the cope#.feat dir of the 2nd 
    level analyses
    space: target analysis space (key-value pair; e.g. space-T1w)
    """
    print(' . Creating fsf files for analysis: ' + analysis_label)
    # replace analysis label
    replace_text(fsf_file, 'analysis-3rdGLMXX', analysis_label)
    # replace number of volumes (i.e. participants in higher level analyses)
    replace_text(fsf_file, 'XXX_NTPS', str(len(subjects)))
    # replace cope number corresponding to cope#.feat dir
    replace_text(fsf_file, 'copeXX', cope)
    # replace analysis space place holder with target space
    replace_text(fsf_file, 'space-SXX', space)
    # construct & replace feat_filesXX string
    feat_files_string = construct_feat_files_string(subjects, analysis_label, cope, space)
    replace_text(fsf_file, 'feat_filesXX', feat_files_string) 
    # construct & replace evgXX string
    evg_string = construct_evg_string(subjects)
    replace_text(fsf_file, 'evgXX', evg_string) 
    # construct & replace groupmemXX string
    groupmem_string = construct_groupmem_string(subjects)
    replace_text(fsf_file, 'groupmemXX', groupmem_string) 

def construct_feat_files_string(subjects, analysis_label, cope, space):
    """
    Constructs feat files string for replacing placeholder in 3rd level fsf 
    template (currently specific to GLM analysis)
    subjects: list of subject IDs to be used in the 3rd level analysis
    analysis_label: string of analysis label. Will be the 3rd level feat 
    analysis output label and must be the same name as of the 2nd level 
    analysis label; i.e. name of the analysis of the 2nd level feat dir 
    (e.g. GLMTarFace)
    cope: cope of interest; must correspond to the cope#.feat dir of the 2nd 
    level analyses
    space: name of the analysis space (e.g. MNI152NLin2009cAsym)
    """
    analysis_short_label = analysis_label[analysis_label.find('3rd')+len('3rd')::]
    feat_path_string_template = 'set feat_files(%(sub_idx)s) "%(fsl_output_dir)s/%(sub)s/ses-V1/%(sub)s_ses-V1_task-Dur_analysis-2nd%(analysis_short_label)s_%(space)s.gfeat/%(cope)s.feat"\n\n'
    feat_files_string = ''
    for sub_idx in range(len(subjects)):
        feat_files_string = feat_files_string + feat_path_string_template%{'sub_idx':str(sub_idx+1), 'fsl_output_dir':fsl_output_dir, 'sub':subjects[sub_idx], 'analysis_short_label':analysis_short_label, 'space':space, 'cope':cope}
    return feat_files_string

def construct_evg_string(subjects):
    """
    Constructs evg string for replacing placeholder in 3rd level fsf 
    template (currently specific to GLM analysis)
    subjects: list of subject IDs to be used in the 3rd level analysis
    """
    evg_string = ''
    for sub_idx in range(len(subjects)):
        evg_string = evg_string + 'set fmri(evg%(sub_idx)s.1) 1\n\n'%{'sub_idx':str(sub_idx+1)}
    return evg_string

def construct_groupmem_string(subjects):
    """
    Constructs groupmem string for replacing placeholder in 3rd level fsf 
    template (currently specific to GLM analysis)
    subjects: list of subject IDs to be used in the 3rd level analysis
    """
    groupmem_string = ''
    for sub_idx in range(len(subjects)):
        groupmem_string = groupmem_string + 'set fmri(groupmem.%(sub_idx)s) 1\n\n'%{'sub_idx':str(sub_idx+1)}
    return groupmem_string


# %% run
if __name__ == '__main__':
    """
    Get analyses from analysis_definitions, then run fsf file creation for all 
    analyses.
    bids_dir: path to bids folder
    analysis_definitions: dict per analysis to be performed
    """
    # get subject list
    subjects = get_subject_list(bids_dir,subject_list_type)
    
    remove_subjects = ['sub-SD122','sub-SD196']
    for r in remove_subjects:
        subjects = subjects[subjects != r]

    print('Removed subjects:',remove_subjects)
    print('Total subjects:',len(subjects))

    # create empty df for error log
    error_df = pd.DataFrame(columns=['sub','analysis','analyses_suffix','error_run_no','error_msg'])
    
    # loop over analyses in analysis definition dict
    for key in analysis_definitions:
        print('')
        print('PROCESSING ANALYSIS: -> ' + key + ' <- ')
        
        # get analysis specification 
        analysis_dict = analysis_definitions[key]
        analyses, analyses_suffix, fsf_templates_dir, run_missing_contrast_check = get_analysis_labels(analysis_dict)
        
        # get required walltime and memory from analysis dict
        if submit_jobs:
            walltime = analysis_dict['walltime']
            memory = analysis_dict['memory']
        else:
            walltime = 0
            memory = 0
        
        # determine fsf output dir
        fsf_output_subdir = analyses_suffix[analyses_suffix.find('analysis-')+len('analysis-'):analyses_suffix.find('analysis-')+len('analysis-')+3]
        fsf_output_dir = bids_dir + os.sep + 'derivatives' + os.sep + 'fslFeat' + os.sep + 'fsf_files' + os.sep + fsf_output_subdir + '_level'
        # create fsf output directory
        if not os.path.isdir(fsf_output_dir):
            os.makedirs(fsf_output_dir)
        
        # check whether group analysis (3rd level) of individual subject analyses (1st or 2nd level) are to be performed
        if run_analysis == 3:
            # run fsf file creation for group analyses for current analysis
            copes = analysis_dict['copes']
            analysis = analyses[0]
            run_fsf_creation_and_submit_feat_job_group_analyses(bids_dir, copes, analysis, analyses_suffix, fsf_templates_dir, fsf_output_dir, run_missing_contrast_check, error_df, walltime, memory)
        else:
            # run fsf file creation for all subjects and runs of current analysis
            run_fsf_creation_and_submit_feat_job(bids_dir, subjects, analyses, analyses_suffix, fsf_templates_dir, fsf_output_dir, run_missing_contrast_check, error_df, walltime, memory)
        
        print('')
        print('waiting before continuting with next analyses')
        time.sleep(time_wait_after_analysis)
        
    # write log file for submission listing potential errors
    write_error_log(bids_dir, error_df)

