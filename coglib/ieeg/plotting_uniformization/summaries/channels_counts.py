import pandas as pd
from pathlib import Path
import mne
import numpy as np


bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"

# List the subjects:
subjects_tsv = pd.read_csv(Path(bids_root, "participants.tsv"), sep="\t")

# List all the subjects:
subjects_list = subjects_tsv.loc[subjects_tsv["visual_responsiveness"] == True, "participant_id"].to_list()

all_channels = []
for subject in subjects_list:
    all_channels.append(pd.read_csv(Path(bids_root,
                                         subject, "ses-V1/ieeg", "{}_ses-V1_task-Dur_channels.tsv".format(subject)),
                        sep="\t"))
# Concatenate:
all_channels = pd.concat(all_channels, ignore_index=True)

# Separate ecog and seeg:
ecog_channels = all_channels.loc[all_channels["type"] == "ECOG"]
# Further separate ecog channels by grids and strips:
strips = []
grids = []
for ch in ecog_channels["name"].to_list():
    if ch[0] == "G":
        grids.append(ch)
    else:
        strips.append(ch)
seeg_channels = all_channels.loc[all_channels["type"] == "SEEG"]
print("Overall:")
print("# SEEG channels={}".format(seeg_channels.shape[0]))
print("# Grids channels={}".format(len(grids)))
print("# Strips channels={}".format(len(strips)))
print("# Subjects={}".format(len(subjects_list)))


n_grids = []
n_strips = []
n_seeg = []

for subject in subjects_list:
    epoch_dir = Path(bids_root, "derivatives", "preprocessing", subject,
                     "ses-V1/ieeg/epoching/high_gamma/desbadcharej_notfil_lapref/"
                     "{}_ses-V1_task-Dur_desc-epoching_ieeg-epo.fif".format(subject))
    epochs = mne.read_epochs(epoch_dir, preload=True, verbose=False)
    try:
        ecog = epochs.copy().pick_types(ecog=True, exclude='bads')
        grids = []
        strips = []
        for ch in ecog.ch_names:
            if ch[0] == "G":
                grids.append(ch)
            else:
                strips.append(ch)
    except ValueError:
        strips = []
        grids = []
    seeg = epochs.copy().pick_types(seeg=True, exclude='bads')
    channel_types = epochs.get_channel_types()
    n_grids.append(len(grids))
    n_strips.append(len(strips))
    n_seeg.append(len(seeg.ch_names))

print("Analyzed:")
print("# SEEG channels={}".format(np.sum(n_seeg)))
print("# Grids channels={}".format(np.sum(n_grids)))
print("# Strips channels={}".format(np.sum(n_strips)))

