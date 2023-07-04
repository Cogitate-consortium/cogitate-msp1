"""
Modified by Urszula Gorska (gorska@wisc.edu)
===================================
01. Maxwell filter using MNE-python
===================================

Includes Maxwell filter function used by MEG Team preprocessing.
The data are Maxwell filtered using tSSS/SSS.
It is critical to mark bad channels before Maxwell filtering.

"""  

import os.path as op
import os

import mne
from mne.preprocessing import find_bad_channels_maxwell


def run_maxwell_filter(raw, destination, crosstalk_file, fine_cal_file):
    # Detect bad channels
    raw.info['bads'] = []
    raw_check = raw.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check,
        cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        return_scores=True,
        verbose=True)
    raw.info['bads'].extend(auto_noisy_chs + auto_flat_chs)

    # Fix Elekta magnetometer coil types
    raw.fix_mag_coil_types()

    # Perform tSSS/SSS and Maxwell filtering
    raw_sss = mne.preprocessing.maxwell_filter(
        raw,
        cross_talk=crosstalk_file,
        calibration=fine_cal_file,
        st_duration=None,
        origin='auto',
        destination=destination,
        coord_frame='head', # 'meg' only for empy room, comment it if using HPI
        verbose=True)

    return raw_sss, {
        'noisy': auto_noisy_chs,
        'flat': auto_flat_chs
    }