# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:56:07 2021

@author: Oscar Ferrante oscfer88@gmail.com
"""


import argparse

import P01_maxwell_filtering
import P02_find_bad_eeg
import P03_artifact_annotation
import P04_extract_events
import P05_run_ica
import P06_apply_ica
import P07_make_epochs


# =============================================================================
# PARSER SETTINGS
# =============================================================================

parser=argparse.ArgumentParser()
parser.add_argument('--sub',type=str,default='CA101',help='subject_id')
parser.add_argument('--visit',type=str,default='V1',help='visit_id')
parser.add_argument('--record',type=str,default='run',help='recording_type (run or rest')
parser.add_argument('--step',type=str,default='1',help='preprocess step')

opt=parser.parse_args()


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================

subject_id = opt.sub
visit_id = opt.visit
record = opt.record

# Find out whwether the participant has EEG data
if visit_id.upper() == 'V1':
    if subject_id.upper() in ['CA101', 'CA102', 'CA103', 'CA104', 'CB036']:
        has_eeg = False
    else:
        has_eeg = True
elif visit_id.upper() == 'V2':
    if subject_id.upper() in ['CA104', 'CA106', 'CA125', 'CB036']:
        has_eeg = False
    else:
        has_eeg = True


# =============================================================================
# DEFINE PREPROCESSING STEPS
# =============================================================================

def pre_step1():
    print("\n\n\n#######################\nP01_maxwell_filtering\n#######################\n")
    P01_maxwell_filtering.run_maxwell_filter(subject_id,
                                             visit_id,
                                             record)
    if has_eeg:
        print("\n\n\n#######################\nP02_find_bad_eeg\n#######################\n")
        P02_find_bad_eeg.find_bad_eeg(subject_id,
                                      visit_id,
                                      record,
                                      has_eeg)
    print("\n\n\n#######################\nP03_artifact_annotation\n#######################\n")
    P03_artifact_annotation.artifact_annotation(subject_id,
                                                visit_id,
                                                record,
                                                has_eeg,
                                                # threshold_muscle,
                                                )
    if record == "run":
        print("\n\n\n#######################\nP04_extract_events\n#######################\n")
        P04_extract_events.run_events(subject_id,
                                      visit_id)
        print("\n\n\n#######################\nP05_run_ica\n#######################\n")
        P05_run_ica.run_ica(subject_id,
                            visit_id,
                            has_eeg)

def pre_step2():
    print("\n\n\n#######################\nP06_apply_ica\n#######################\n")
    P06_apply_ica.apply_ica(subject_id,
                            visit_id,
                            record,
                            has_eeg)

    print("\n\n\n#######################\nP07_make_epochs\n#######################\n")
    if record == "rest":
        P07_make_epochs.run_epochs(subject_id,
                                   visit_id,
                                   "rest",
                                   has_eeg)
    elif visit_id == 'V1':
        P07_make_epochs.run_epochs(subject_id,
                                   visit_id,
                                   'dur',
                                   has_eeg)
    elif visit_id == 'V2':
        P07_make_epochs.run_epochs(subject_id,
                                   visit_id,
                                   'vg',
                                   has_eeg)
        P07_make_epochs.run_epochs(subject_id,
                                   visit_id,
                                   'replay',
                                   has_eeg)


# =============================================================================
# RUN
# =============================================================================
if opt.step == '1':
    pre_step1()
elif opt.step == '2':
    pre_step2()
elif opt.step == '0':
    pre_step1()
    pre_step2()
