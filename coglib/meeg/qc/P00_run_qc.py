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
parser.add_argument('--sub', type=str, default='SA101', help='subject_id')
parser.add_argument('--visit', type=str, default='V1', help='visit_id')

opt=parser.parse_args()

# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================

subject_id = opt.sub
visit_id = opt.visit

# Find out whether the participant has EEG data
if visit_id.upper() == 'V1':
    if subject_id.upper() in ['SA101', 'SA102', 'SA103', 'SA104']:
        has_eeg = False
    else:
        has_eeg = True
elif visit_id.upper() == 'V2':
    if subject_id.upper() in ['SA104', 'SA106']:
        has_eeg = False
    else:
        has_eeg = True


QC_processing.run_qc_processing(subject_id, visit_id, has_eeg)
