import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd

import mne
import mne_bids
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import annotate_muscle_zscore
from autoreject import get_rejection_threshold
from numpy import arange

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from .config.config import (bids_root, tmin, tmax)

from matplotlib.backends.backend_pdf import PdfPages

from qc.maxwell_filtering import run_maxwell_filter
from qc.extract_events import run_events

from qc.viz_psd import viz_psd


def run_qc_epochs(subject_id, visit_id, has_eeg):
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")

    # Set path to qc derivatives and create the related folders
    qc_output_path = op.join(bids_root, "derivatives", "qc", visit_id)
    if not op.exists(qc_output_path):
        os.makedirs(qc_output_path)

    print("Processing subject: %s" % subject_id)

    #raw_list = list()
    #events_list = list()
    #metadata_list = list()

    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    elif visit_id == "V2":
        bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")


    #with PdfPages(op.join(qc_output_path, subject_id + '_' + visit_id + '_MEG_V1_epochs.pdf')) as pdf:

        #FirstPage = plt.figure(figsize=(8,1), dpi=108)
        #FirstPage.clf()
        #plt.axis('off')
        #plt.text(0.5, 0.5, subject_id, transform=FirstPage.transFigure, size=16, ha="center")
        #pdf.savefig(FirstPage)
        #plt.close()


    #raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    #del raw_list

     # Concatenate metadata tables
    #metadata = pd.concat(metadata_list)
    # metadata.to_csv(op.join(out_path, file_name[0:14] + 'ALL-meta.csv'), index=False)

    # Select sensor types
    #picks = mne.pick_types(raw.info,
    #                     meg = True,
    #                    eeg = has_eeg,
    #                    stim = True,
    #                    eog = has_eeg,
    #                    ecg = has_eeg,
    #                    )

    # Set trial-onset event_ids
    if visit_id == 'V1':
        events_id = {}
        types = ['face','object','letter','false']
        for j,t in enumerate(types):
            for i in range(1,21):
                events_id[t+str(i)] = i + j * 20
#    elif visit_id == 'V2':
#        events_id = {}
#        events_id['blank'] = 50
#        types = ['face','object']
#        for j,t in enumerate(types):
#            for i in range(1,11):
#                events_id[t+str(i)] = i + j * 20

    elif visit_id == 'V2':
        if bids_task == "vg":
            events_id = {}
            events_id['blank'] = 50
            types = ['face','object']
            for j,t in enumerate(types):
                for i in range(1,11):
                    events_id[t+str(i)] = i + j * 20
        elif bids_task == "replay":
            events_id = {}
            events_id['blankFT'] = 150
            events_id['blankOT'] = 250
            typesF = ['faceT','faceNT']
            for j,t in enumerate(typesF):
                for i in range(1,11):
                    events_id[t+str(i)] = i + 100 + j * 100
            typesO = ['objectNT','objectT']
            for j,t in enumerate(typesO):
                for i in range(1,11):
                    events_id[t+str(i)] = i + 120 + j * 100

    # Epoch raw data
    #epochs = mne.Epochs(raw,
    #                    events,
    #                    events_id,
    #                    tmin, tmax,
    #                    baseline=None,
    #                    proj=True,
    #                    picks=picks,
    #                    detrend=1,
    #                    reject=None,
    #                    reject_by_annotation=True,
    #                    verbose=True)

    # Add metadata
    #epochs.metadata = metadata

    # Read epoched data from preprocessed
    bids_path_epo = mne_bids.BIDSPath(
            root=prep_deriv_root,
            subject=subject_id,
            datatype='meg',
            task=bids_task,
            session=visit_id,
            suffix='epo',
            extension='.fif',
            check=False)

    epochs = mne.read_epochs(bids_path_epo.fpath, preload=False)

    if visit_id == 'V1':
        print("VERY_IMPORTANT :)")
        print("FACES task relevant")
        epochs_rel_F = epochs['Task_relevance == "Relevant non-target" and Category == "face"']
        print(epochs_rel_F)
        print("FACES task irrelevant")
        epochs_irr_F = epochs['Task_relevance == "Irrelevant" and Category == "face"']
        print(epochs_irr_F)

        print("OBJECTS task relevant")
        epochs_rel_O = epochs['Task_relevance == "Relevant non-target" and Category == "object"']
        print(epochs_rel_O)
        print("OBJECTS task irrelevant")
        epochs_irr_O = epochs['Task_relevance == "Irrelevant" and Category == "object"']
        print(epochs_irr_O)

        print("LETTERS task relevant")
        epochs_rel_L = epochs['Task_relevance == "Relevant non-target" and Category == "letter"']
        print(epochs_rel_L)
        print("LETTERS task irrelevant")
        epochs_irr_L = epochs['Task_relevance == "Irrelevant" and Category == "letter"']
        print(epochs_irr_L)

        print("FALSE FONTS task relevant")
        epochs_rel_S = epochs['Task_relevance == "Relevant non-target" and Category == "false"']
        print(epochs_rel_S)
        print("FALSE FONTS task irrelevant")
        epochs_irr_S = epochs['Task_relevance == "Irrelevant" and Category == "false"']
        print(epochs_irr_S)

    elif visit_id == 'V2':
        print("FACES probe")
        epochs_rel_F = epochs['Trial_type == "Probe" and Stimuli_type == "Face"']
        print(epochs_rel_F)
        print("FACES filler")
        epochs_irr_F = epochs['Trial_type == "Filler" and Stimuli_type == "Face"']
        print(epochs_irr_F)

        print("OBJECTS probe")
        epochs_rel_O = epochs['Trial_type == "Probe" and Stimuli_type == "Object"']
        print(epochs_rel_O)
        print("OBJECTS filler")
        epochs_irr_O = epochs['Trial_type == "Filler" and Stimuli_type == "Object"']
        print(epochs_irr_O)

        print("BLANKS probe")
        epochs_rel_O = epochs['Trial_type == "Probe" and Stimuli_type == "Blank"']
        print(epochs_rel_O)
        print("BLANKS filler")
        epochs_irr_O = epochs['Trial_type == "Filler" and Stimuli_type == "Blank"']
        print(epochs_irr_O)

        #types0 = ['Filler', 'Probe']
        #type1 = ['Face', 'Object', 'Blank']
        #location = ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']
        #response = ['Seen', 'Unseen']

    #print(epochs[['face1', 'face2','face3','face4','face5','face6','face7','face8','face9','face10']])
    #print(epochs[['object1', 'object2','object3','object4','object5','object6','object7','object8','object9','object10']])
    #print(epochs[['letter1', 'letter2','letter3','letter4','letter5','letter6','letter7','letter8','letter9','letter10']])
    #print(epochs[['false1', 'false2','false3','false4','false5','false6','false7','false8','false9','false10']])

    # Get rejection thresholds - MEG only
    #reject = get_rejection_threshold(epochs, ch_types=['mag', 'grad'], #'eeg'],
    #                                        decim=2)

    # Drop bad epochs based on peak-to-peak magnitude
    #epochs.drop_bad(reject=reject)

    #print("VERY_IMPORTANT EPOCHS faces after drop")
    #epochs_rel_F = epochs['Task_relevance == "Relevant non-target" and Category == "face"']
    #print(epochs_rel_F)
    #print(epochs[['face1', 'face2','face3','face4','face5','face6','face7','face8','face9','face10']])
    #print(epochs[['object1', 'object2','object3','object4','object5','object6','object7','object8','object9','object10']])
    #print(epochs[['letter1', 'letter2','letter3','letter4','letter5','letter6','letter7','letter8','letter9','letter10']])
    #print(epochs[['false1', 'false2','false3','false4','false5','false6','false7','false8','false9','false10']])

    #print("VERY_IMPORTANT DROP LOG")
    #print(epochs.drop_log)

    # Plot percentage of rejected epochs per channel
    #fig1 = epochs.plot_drop_log()
    #pdf.savefig(fig1)
    plt.close()


if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., CA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    run_qc_epochs(subject_id, visit_id)