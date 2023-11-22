# -*- coding: utf-8 -*-
"""
@author: Urszula Górska gorska@wisc.edu
"""

import argparse

import QC_processing

# =============================================================================
# PARSER SETTINGS
# =============================================================================

parser=argparse.ArgumentParser()
parser.add_argument('--sub', type=str, default='CA101', help='subject_id')
parser.add_argument('--visit', type=str, default='V1', help='visit_id')

opt=parser.parse_args()

# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================

subject_id = opt.sub
visit_id = opt.visit

# Find out whether the participant has EEG data
if visit_id.upper() == 'V1':
    if subject_id.upper() in ['CA101', 'CA102', 'CA103', 'CA104']:
        has_eeg = False
    else:
        has_eeg = True
elif visit_id.upper() == 'V2':
    if subject_id.upper() in ['CA104', 'CA106']:
        has_eeg = False
    else:
        has_eeg = True


# # =============================================================================
# # DEFINE PREPROCESSING STEPS
# # =============================================================================

# def pre_step1():
#     P01_maxwell_filtering.run_maxwell_filter(subject_id,
#                                              visit_id)
#     if has_eeg:
#         P02_find_bad_eeg.find_bad_eeg(subject_id,
#                                       visit_id,
#                                       has_eeg)
#     P03_artifact_annotation.artifact_annotation(subject_id,
#                                                 visit_id,
#                                                 has_eeg,
#                                                 # threshold_muscle,
#                                                 )
#     P04_extract_events.run_events(subject_id,
#                                   visit_id)
#     P05_run_ica.run_ica(subject_id,
#                         visit_id,
#                         has_eeg)

# def pre_step2(
#               # meg_ica_eog=opt.mICA_eog, meg_ica_ecg=opt.mICA_ecg,
#               # eeg_ica_eog=opt.eICA_eog, eeg_ica_ecg=opt.eICA_ecg,
#               ):
#     P06_apply_ica.apply_ica(subject_id,
#                             visit_id,
#                             has_eeg)

#     P07_make_epochs.run_epochs(subject_id,
#                             visit_id,
#                             has_eeg)


# =============================================================================
# RUN
# =============================================================================
# if opt.step == '1':
#     pre_step1()
# elif opt.step == '2':
#     pre_step2()
QC_processing.run_qc_processing(subject_id, visit_id, has_eeg)
