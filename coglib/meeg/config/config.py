# -*- coding: utf-8 -*-
"""
===========
Config file
===========

Configurate the parameters of the study.

"""

import os

# =============================================================================
# BIDS SETTINGS
# =============================================================================
# if os.getlogin() in ['oscfe', 'ferranto', 'FerrantO']:  #TODO: doesn't work on the HPC
#     bids_root = r'Z:\_bids_'
# else:
bids_root = r'Z:\_bids_'


# =============================================================================
# MAXWELL FILTERING SETTINGS
# =============================================================================

# Set filtering method
method='sss'
if method == 'tsss':
    st_duration = 10
else:
    st_duration = None


# =============================================================================
# FILTERING AND DOWNSAMPLING SETTINGS
# =============================================================================

# Filter and resampling params
l_freq = 1
h_freq = 40
sfreq = 100


# =============================================================================
# EPOCHING SETTINGS
# =============================================================================

# Set timewindow
tmin = -1
tmax = 2.5

# Epoch rejection criteria
reject_meg_eeg = dict(grad=4000e-13,    # T / m (gradiometers)
                      mag=6e-12        # T (magnetometers)
                      #eeg=200e-6       # V (EEG channels)
                      )
reject_meg = dict(grad=4000e-13,    # T / m (gradiometers)
                  mag=6e-12         # T (magnetometers)
                  )


# =============================================================================
# ICA SETTINGS
# =============================================================================

ica_method = 'fastica'
n_components = 0.99
max_iter = 800
random_state = 1688


# =============================================================================
#  FACTOR AND CONDITIONS OF INTEREST
# =============================================================================

# factor = ['Category']
# conditions = ['face', 'object', 'letter', 'false']

# factor = ['Duration']
# conditions = ['500ms', '1000ms', '1500ms']

# factor = ['Task_relevance']
# conditions = ['Relevant_target','Relevant_non-target','Irrelevant']

# factor = ['Duration', 'Task_relevance']
# conditions = [['500ms', '1000ms', '1500ms'],
#               ['Relevant target','Relevant non-target','Irrelevant']]

factor = ['Category', 'Task_relevance']
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant target','Relevant non-target','Irrelevant']]

    
# =============================================================================
# TIME-FREQUENCY REPRESENTATION SETTINGS
# =============================================================================

baseline_w = [-0.5, -0.25]     #only for plotting
freq_band = 'both' #can be 'low', 'high' or 'both'


# =============================================================================
# SOURCE MODELING
# =============================================================================

# Forward model
spacing='oct6'  #from forward_model

# Inverse model
#   Beamforming
beam_method = 'dics'  #'lcmv' or 'dics'

active_win = (0.75, 1.25)
baseline_win = (-.5, 0)


# =============================================================================
# PLOTTING
# =============================================================================

# Subset of posterior sensors
post_sens = {'grad': ['MEG1932', 'MEG1933', 'MEG2122', 'MEG2123',
                     'MEG2332', 'MEG2333', 'MEG1922', 'MEG1923',
                     'MEG2112', 'MEG2113', 'MEG2342', 'MEG2343'],
             'mag': ['MEG1931', 'MEG2121',
                     'MEG2331', 'MEG1921',
                     'MEG2111', 'MEG2341'],
             'eeg': ['EEG056', 'EEG030',
                     'EEG057', 'EEG018',
                     'EEG032', 'EEG019']}
