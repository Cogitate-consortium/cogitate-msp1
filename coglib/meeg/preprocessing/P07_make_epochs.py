"""
====================
07. Make epochs
====================

Open questions:
    - separate MEG and EEG in two different FIF files?
    - Exp.2: separate VG and replay in two different files?

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os.path as op
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from fpdf import FPDF
import mne
import mne_bids
from autoreject import get_rejection_threshold

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import (bids_root, tmin, tmax)


def run_epochs(subject_id, visit_id, task, has_eeg=False):
    
    # Prepare PDF report
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    
    # Set path to preprocessing derivatives
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    prep_figure_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    prep_report_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "reports")
    prep_code_root = op.join(prep_deriv_root,
                             f"sub-{subject_id}",f"ses-{visit_id}","meg",
                             "codes")
    
    # Create empty lists
    raw_list = list()
    events_list = list()
    metadata_list = list()
    
    print("Processing subject: %s" % subject_id)
    
    # Loop over runs
    data_path = os.path.join(bids_root,f"sub-{subject_id}",f"ses-{visit_id}","meg")
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".json") and task in fname:  
            if "run" in fname or "rest" in fname:
                # Set run
                if "run" in fname:
                    run = f"{int(fname[-10]):02}"
                elif "rest" in fname:
                    run = None
                print("  Run: %s" % run)
                
                # Read filtered data
                bids_path_filt = mne_bids.BIDSPath(
                    root=prep_deriv_root, 
                    subject=subject_id,  
                    datatype='meg',  
                    task=task,
                    run=run,
                    session=visit_id, 
                    suffix='filt',
                    extension='.fif',
                    check=False)
                
                raw_tmp = mne_bids.read_raw_bids(bids_path_filt)
                
                # Read events
                if "run" in fname:
                    bids_path_eve = mne_bids.BIDSPath(
                        root=prep_deriv_root, 
                        subject=subject_id,  
                        datatype='meg',  
                        task=task,
                        run=run,
                        session=visit_id, 
                        suffix='eve',
                        extension='.fif',
                        check=False)
                    
                    events_tmp = mne.read_events(bids_path_eve.fpath)
                    
                    # Read metadata
                    bids_path_meta = mne_bids.BIDSPath(
                        root=prep_deriv_root, 
                        subject=subject_id,  
                        datatype='meg',  
                        task=task,
                        run=run,
                        session=visit_id, 
                        suffix='meta',
                        extension='.csv',
                        check=False)
                    
                    metadata_tmp = pd.read_csv(op.join(bids_path_meta.fpath))
                    
                    metadata_list.append(metadata_tmp)
                    
                elif "rest" == task:
                    events_tmp = mne.make_fixed_length_events(
                        raw_tmp, duration=5)
               
                # Append read data to list
                raw_list.append(raw_tmp)
                events_list.append(events_tmp)
                

    # Concatenate raw instances as if they were continuous
    raw, events = mne.concatenate_raws(raw_list,
                                       events_list=events_list)
    del raw_list
    
    # Concatenate metadata tables and save it
    if task != "rest":
        metadata = pd.concat(metadata_list)
        
        bids_path_meta.update(
                run=None,
                check=False)
        
        metadata.to_csv(bids_path_meta.fpath,
                        index=False)
    
    # Set trial-onset event_ids
    if task == "rest":
        events_id = {"rest": 1}
    elif visit_id == 'V1':
        events_id = {}
        types = ['face','object','letter','false']
        for j,t in enumerate(types):
            for i in range(1,21):
                events_id[t+str(i)] = i + j * 20
    elif visit_id == 'V2':
        if task == "vg":
            events_id = {}
            events_id['blank'] = 50
            types = ['face','object']
            for j,t in enumerate(types):
                for i in range(1,11):
                    events_id[t+str(i)] = i + j * 20
        elif task == "replay":
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
    
    # Select sensor types
    picks = mne.pick_types(raw.info,
                           meg = True,
                           eeg = has_eeg,
                           stim = True,
                           eog = has_eeg,
                           ecg = has_eeg)
    
    # Epoch raw data
    epochs = mne.Epochs(raw,
                        events, 
                        events_id,
                        tmin, tmax,
                        baseline=None,
                        proj=True,
                        picks=picks,
                        detrend=1,
                        reject=None,
                        reject_by_annotation=True, #reject muscle artifacts
                        verbose=True)
    del raw
    
    # Add metadata
    if task != "rest":
        epochs.metadata = metadata
    
    # Get rejection thresholds
    reject = get_rejection_threshold(epochs, 
                                     ch_types=['mag', 'grad'], #'eeg'], #eeg not used for epoch rejection
                                     decim=2)
    
    # Drop bad epochs based on peak-to-peak magnitude
    nr_epo_prerej = len(epochs.events)
    epochs.drop_bad(reject=reject)
    nr_epo_postrej = len(epochs.events)
    
    # Plot percentage of rejected epochs per channel
    fig1 = epochs.plot_drop_log()
    fname_fig1 = op.join(prep_figure_root,
                        f'07_{task}rAll_epoch_drop.png')
    fig1.savefig(fname_fig1)
    plt.close()
    
    # Add figure to report
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, fname[:16])
    pdf.ln(120)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Percentage of rejected epochs', 'B', ln=1)
    pdf.image(fname_fig1, 0, 45, pdf.epw*.8)
    pdf.ln(20)
    pdf.cell(0, 10, "Number of epochs:")
    pdf.ln(20)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, f'Before rejection: {nr_epo_prerej}', 'B', ln=1)
    pdf.cell(0, 10, f'After rejection: {nr_epo_postrej}', 'B', ln=1)

    # Plot evoked by epochs
    fig2 = epochs.plot(picks='meg',
                      title='meg',
                      n_epochs=10)
    fname_fig2 = op.join(prep_figure_root,
                        f'07_{task}rAll_epoch_evk.png')
    fig2.savefig(fname_fig2)
    plt.close(fig2)
    
    # Add figures to report
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, fname[:16])
    pdf.ln(20)
    pdf.cell(0, 10, 'Epoched data', 'B', ln=1)
    pdf.image(fname_fig2, 0, 45, pdf.epw)
    
    # Count the number of epochs defined by different events
    num = {}
    for key in events_id:
        num[key] = len(epochs[key])
    df = pd.DataFrame(num,
                      index = ["Total"])
    df.to_csv(op.join(prep_report_root,
                      f'P07_make_epochs-count_{task}_event.csv'),  
              index=False)
    print(df)
    
    # Save epoched data
    bids_path_epo = bids_path_filt.copy().update(
            root=prep_deriv_root,
            run=None,
            suffix="epo",
            check=False)

    epochs.save(bids_path_epo, overwrite=True)
    
    # Save code
    shutil.copy(__file__, prep_code_root)
    
    # Save report
    if task == "rest":
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + f'-{task}-report_rest.pdf'))
    else:
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + f'_{task}-report.pdf'))


def input_bool(message):
    value = input(message)
    if value == "True":
        return True
    if value == "False":
        return False
    

if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    task = input("Type the task (dur, vg or replay)\n>>> ")
    has_eeg = input_bool("Has this recording EEG data? (True or False)\n>>> ")
    run_epochs(subject_id, visit_id, task, has_eeg)
    