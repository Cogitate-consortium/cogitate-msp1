import mne
import numpy as np
from mne_bids import (write_raw_bids, BIDSPath)
import os
import tempfile


CHANNELS_FREQUENCY_RANGES = {
    "SEEG1": [0, 10],
    "SEEG2": [20, 30],
    "SEEG3": [40, 50],
    "SEEG4": [60, 70],
    "SEEG5": [80, 90],
    "SEEG6": [100, 110],
    "ECOG1": [0, 10],
    "ECOG2": [20, 30],
    "ECOG3": [40, 50],
    "ECOG4": [60, 70],
    "ECOG5": [80, 90],
    "ECOG6": [100, 110],
    "EEG1": [0, 10],
    "EEG2": [20, 30],
    "EEG3": [40, 50],
    "EEG4": [60, 70],
    "EEG5": [80, 90],
    "EEG6": [100, 110],
    "EKG1": [0, 10],
    "EKG2": [20, 30],
}
SIGNAL_DURATION_SEC = 20
ch_types = ["seeg"] * 6 + ["ecog"] * 6 + ["eeg"] * 6 + ["ecg"] * 2
sfreq = 1000
line_freq = 60
SUBJECT_ID = "SAMPLE1"
SESSION = "V1"
TASK = "TEST"
DATA_TYPE = "ieeg"
BIDS_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bids")
AMPLITUDE = 0.0001
IEEG_CHANNELS = ["ECOG1", "ECOG2", "ECOG3", "ECOG4", "ECOG5", "ECOG6",
                 "SEEG1", "SEEG2", "SEEG3", "SEEG4", "SEEG5", "SEEG6"]


def save_to_bids(fname):
    """

    :return:
    """
    # Loading the file without loading the data:
    raw = mne.io.read_raw(fname, preload=False)
    # Separating the ieeg and ecog data:
    for data_type in ["ieeg", "eeg"]:
        if data_type == "ieeg":
            # Making a fake montage
            ch_pos = np.random.rand(len(IEEG_CHANNELS), 3) * 10
            ch_pos = ch_pos / 1000.
            montage = mne.channels.make_dig_montage(ch_pos=dict(zip(IEEG_CHANNELS,
                                                                    ch_pos)),
                                                    coord_frame='mni_tal')
            raw.set_montage(montage, on_missing='warn')
            # Generating the bids path:
            bids_path = BIDSPath(subject=SUBJECT_ID, session=SESSION,
                                 task=TASK, datatype=data_type, root=BIDS_ROOT)
            # Saving the file to bids dir:
            write_raw_bids(raw.copy().pick_types(seeg=True, ecog=True, ecg=True), bids_path,
                           overwrite=True, format='auto')
        elif data_type == "eeg":
            # Generating the bids path:
            bids_path = BIDSPath(subject=SUBJECT_ID, session=SESSION,
                                 task=TASK, datatype="eeg", root=BIDS_ROOT)
            # Saving the file to bids dir:
            write_raw_bids(raw.copy().pick_types(eeg=True), bids_path, overwrite=True, format='auto')

    return None


def simulate_bids_data():
    """
    This function generates raw data and saves them into the BIDS format. This data are generated to enable easy
    spotting of issues in the preprocessing scripts
    :return:
    """

    # First, creating info about the signal:
    info = mne.create_info(list(CHANNELS_FREQUENCY_RANGES.keys()), ch_types=ch_types, sfreq=sfreq)
    info['description'] = 'Test data'
    print(info)

    data = []
    # Every 4th of the duration, we increment by 1 the mean amplitude to avoid being centered on 0 only:
    time = np.arange(0, SIGNAL_DURATION_SEC, 1/sfreq)
    for ch in CHANNELS_FREQUENCY_RANGES.keys():
        # Generate the freq band:
        freq_band1 = np.linspace(CHANNELS_FREQUENCY_RANGES[ch][0], CHANNELS_FREQUENCY_RANGES[ch][1], time.size)
        data.append((AMPLITUDE * np.sin(2 * np.pi * freq_band1 * time)))

    # Converting the data to a numpy array:
    data = np.array(data)
    # Create the simluate raw:
    simulated_raw = mne.io.RawArray(data, info)
    # Plotting the simulated raws:
    simulated_raw.plot()

    # Create annotations to make sure the preprocessing works fine:
    # Creating sine waves matching each defined frequencies:
    my_annotations = mne.Annotations(onset=np.array(list(range(0, SIGNAL_DURATION_SEC, int(SIGNAL_DURATION_SEC/4)))),
                                     duration=np.array([int(SIGNAL_DURATION_SEC/4)] * 4),
                                     description=np.array(
                                         [f'Evt{evt:03}' for evt in list(range(1, int(SIGNAL_DURATION_SEC/4)))]))
    simulated_raw.set_annotations(my_annotations)
    # simulated_raw.plot()
    # Saving the file to temp dir:
    temp_dir = tempfile.mkdtemp()
    # Generate a file name
    fname = temp_dir + os.sep + "_raw.fif"
    # Saving the data in that temp dir:
    simulated_raw.save(fname)
    # Saving the file to BIDS:
    save_to_bids(fname)

    print("S")


if __name__ == "__main__":
    simulate_bids_data()
