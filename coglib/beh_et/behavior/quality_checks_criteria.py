import data_reader

"""
This sets the criteria for the quality checks on subject data of experiment 1. 
This is based on the OSF pre-registration: https://osf.io/3t7ax/

@author: RonyHirsch
"""

# modules
FMRI = 'fMRI'
MEG = 'MEG'
ECOG = 'ECoG'
LAB = 'Lab'
# module per lab : taken from https://twcf-arc.slab.com/posts/institutional-abbreviations-rsi4obcd
METHOD_HPC = {'SA': MEG, 'SB': MEG, 'SC': FMRI, 'SD': FMRI, 'SE': ECOG, 'SF': ECOG, 'SG': ECOG, 'SX': ECOG, 'SZ': MEG}  # This is used when saving to the HPC derivatives folder

# Behavioral screening exclusion criteria
"""
For Experiment 1, subjects were excluded if their hit rate was lower than 80% or FAs higher than 20% for M-EEG and fMRI, 
and for iEEG, a more relaxed criteria of 70% Hits and 30% FAs was used. 
Source: https://osf.io/gm3vd
"""
HIT_RATE_MIN = {FMRI: 0.8, MEG: 0.8, ECOG: 0.7}  # minimum hit rate
FA_RATE_MAX = {FMRI: 0.2, MEG: 0.2, ECOG: 0.3}  # maximum false alarm rate


HITS = "hits"
STIM_PRESENT = "stim_present"
HIT_RATE = "hit_rate"
FAS = "fas"
STIM_ABSENT = "stim_absent"
FA_RATE = "fa_rate_all"
TASK_RELEVANT = "task_relevant"
FA_RATE_TR = "fa_rate_task_relevant"
TASK_IRRELEVANT = "task_irrelevant"
FA_RATE_TI = "fa_rate_task_irrelevant"
HITS_OK = "hits_valid?"
FAS_OK = "fas_valid?"
# Valid
VALID_2ND = "Is_BEH_Valid?"
VALID = "Is_Valid?"


def check_validity(row):
    """
    Once we have the VALID column, we need to make sure all analyses consider only subjects who BOTH passed this
    behavioral QC, AND ALSO the 3rd level QC - where neural data was checked.
    If a subject did not pass the 3rd level quality checks, they will be excluded based on this column.
    *NOTE* this is a manual step (based on the list), because the 3rd level checks were a manual step.
    """
    if row[VALID_2ND] == True:
        return True
    return False


def check_hits(row):
    hit_rate = row[HITS] / row[STIM_PRESENT]
    if hit_rate < HIT_RATE_MIN[row[data_reader.MODALITY]]:  # if hit rate is lower than threshold
        return False  # subject is to be excluded
    return True


def check_fas(row):
    fa_rate = row[FA_RATE]  #
    if fa_rate > FA_RATE_MAX[row[data_reader.MODALITY]]:  # if false alarm rate is higher than threshold
        return False  # subject is to be excluded
    return True


def check_data_table(data):
    """
    :param data: dataframe where each subject = row and columns provide information about subjects' responses.
    :return:
    """
    data[HITS_OK] = None
    data[FAS_OK] = None
    data[VALID_2ND] = None
    data[VALID] = None
    for ind, row in data.iterrows():
        data.at[ind, HITS_OK] = check_hits(row)
        data.at[ind, FAS_OK] = check_fas(row)
        data.at[ind, VALID_2ND] = data.at[ind, HITS_OK] & data.at[ind, FAS_OK]
    """
    Once the data is updated, check whether this subject passed both the 2nd (behavioral, here) and 3rd (neural, 
    see list at the top) quality checks.
    """
    for ind, row in data.iterrows():
        data.at[ind, VALID] = check_validity(row)
    return data
