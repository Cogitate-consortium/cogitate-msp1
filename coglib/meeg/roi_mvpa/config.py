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
bids_root = r'/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids'


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

plot_param = {
  "font": "times new roman",
  "font_size": 22,
  "figure_size_mm": [183, 108],
  "fig_res_dpi": 300,
  "colors": {
    "iit": [
      0.00392156862745098,
      0.45098039215686275,
      0.6980392156862745],
    "gnw": [
      0.00784313725490196,
      0.6196078431372549,
      0.45098039215686275
    ],
    "IIT": [
        0.00392156862745098,
        0.45098039215686275,
        0.6980392156862745
    ],
    "GNW": [
        0.00784313725490196,
        0.6196078431372549,
        0.45098039215686275
    ],    
    "MT": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "FP": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
        ],
    "IITPFC_f": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
    ],
    "Relevant to Irrelevant": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
        ],
    "Irrelevant to Relevant": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "Relevant non-target": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "Irrelevant": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
    ],
    "task relevant": [
      0.8352941176470589,
      0.3686274509803922,
      0.0
    ],
    "Irrelevant": [
      0.5450980392156862,
      0.16862745098039217,
      0.8862745098039215
    ],
    # "face": [
    #   0.00784313725490196,
    #   0.24313725490196078,
    #   1.0
    # ],
    # "object": [
    #   0.10196078431372549,
    #   0.788235294117647,
    #   0.2196078431372549
    # ],
    # "letter": [
    #   0.9098039215686274,
    #   0.0,
    #   0.043137254901960784
    # ],
    # "false": [
    #   0.9450980392156862,
    #   0.2980392156862745,
    #   0.7568627450980392
    # ],
    "500ms": [
      1.0,
      0.48627450980392156,
      0.0
    ],
    "1000ms": [
      0.6235294117647059,
      0.2823529411764706,
      0.0
    ],
    "1500ms": [
      1.0,
      0.7686274509803922,
      0.0
    ],
    "face": [
      0/255,
      53/255,
      68/255
    ],
    "object": [
      173/255,
      80/255,
      29/255
    ],
    "letter": [
      57/255,
      115/255,
      132/255
    ],
    "false": [
      97/255,
      15/255,
      0/255
    ],
    "cmap": "RdYlBu_r"
  }
}
