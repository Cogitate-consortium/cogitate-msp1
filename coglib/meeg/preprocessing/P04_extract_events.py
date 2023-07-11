"""
===================
04. Extract events
===================

Extract events from the stimulus channel

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os.path as op
import os
import numpy as np
import pandas as pd
from fpdf import FPDF
import shutil

import mne
import matplotlib.pyplot as plt
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


def run_events(subject_id, visit_id):
    
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
    
    print("Processing subject: %s" % subject_id)
    
    # Lopp over runs
    data_path = os.path.join(bids_root,f"sub-{subject_id}",f"ses-{visit_id}","meg")
    
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".json") and "run" in fname:
            
            # Set run
            run = int(fname[-10])
            print("  Run: %s" % run)

            # Set task
            if 'dur' in fname:
                bids_task = 'dur'
            elif 'vg' in fname:
                bids_task = 'vg'
            elif 'replay' in fname:
                bids_task = 'replay'
            else:
                raise ValueError("Error: could not find the task for %s" % fname)

            # Set BIDS path
            bids_path_annot = mne_bids.BIDSPath(
                root=prep_deriv_root, 
                subject=subject_id,  
                datatype='meg',  
                task=bids_task,
                run=f"{run:02}",
                session=visit_id, 
                suffix='annot',
                extension='.fif',
                check=False)
            
            # Read raw data
            raw = mne_bids.read_raw_bids(bids_path_annot)
        
            ###############
            # Read events #
            ###############
        
            # Find response events
            response = mne.find_events(raw,
                                     stim_channel='STI101',
                                     consecutive = False,
                                     mask = 65280,
                                     mask_type = 'not_and'
                                    )
            response = response[response[:,2] == 255]
            
            # Find all other events
            events = mne.find_events(raw,
                                     stim_channel='STI101',
                                     consecutive = True,
                                     min_duration=0.001001,
                                     mask = 65280,
                                     mask_type = 'not_and'
                                    )
            events = events[events[:,2] != 255]
            
            # Concatenate all events
            events = np.concatenate([response,events],axis = 0)
            events = events[events[:,0].argsort(),:]
            
            # Show events
            fig = mne.viz.plot_events(events)
            fname_fig = op.join(prep_figure_root,
                                "04_%sr%s_events.png" % (bids_task,run))
            fig.savefig(fname_fig)
            plt.close(fig)
            
            # Add figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:-8])
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Events', 'B', ln=1)
            pdf.image(fname_fig, 0, 45, pdf.epw)
            
            # Save event array
            bids_path_eve = bids_path_annot.copy().update(
                suffix="eve",
                check=False)
            if not op.exists(bids_path_eve):
                bids_path_eve.fpath.parent.mkdir(exist_ok=True, parents=True)
                            
            mne.write_events(bids_path_eve.fpath, events)
            
            #################
            # Read metadata #
            #################
            
            # # Generate metadata table
            if visit_id == 'V1':
                eve = events.copy()
                events = eve[eve[:, 2] < 81].copy()
                metadata = {}
                metadata = pd.DataFrame(metadata, index=np.arange(len(events)),
                                        columns=['Stim_trigger', 'Category',
                                                 'Orientation', 'Duration',
                                                 'Task_relevance', 'Trial_ID',
                                                 'Response', 'Response_time(s)'])
                Category = ['face', 'object', 'letter', 'false']
                Orientation = ['Center', 'Left', 'Right']
                Duration = ['500ms', '1000ms', '1500ms']
                Relevance = ['Relevant target', 'Relevant non-target', 'Irrelevant']
                k = 0
                for i in range(eve.shape[0]):
                    if eve[i, 2] < 81:
                    ##find the end of each trial (trigger 97)
                        t = [t for t, j in enumerate(eve[i:i + 9, 2]) if j == 97][0]
                        metadata.loc[k]['Stim_trigger'] = eve[i,2]
                        metadata.loc[k]['Category'] = Category[int((eve[i,2]-1)//20)]
                        metadata.loc[k]['Orientation'] = Orientation[[j-100 for j in eve[i:i+t,2]
                                                                      if j in [101,102,103]][0]-1]
                        metadata.loc[k]['Duration'] = Duration[[j-150 for j in eve[i:i+t,2]
                                                                if j in [151,152,153]][0]-1]
                        metadata.loc[k]['Task_relevance'] = Relevance[[j-200 for j in eve[i:i+t,2]
                                                                       if j in [201,202,203]][0]-1]
                        metadata.loc[k]['Trial_ID'] = [j for j in eve[i:i+t,2]
                                                       if (j>110) and (j<149)][0]
                        metadata.loc[k]['Response'] = True if any(eve[i:i+t,2] == 255) else False
                        if metadata.loc[k]['Response'] == True:
                            r = [r for r,j in enumerate(eve[i:i+t,2]) if j == 255][0]
                            metadata.loc[k]['Response_time(s)'] = (eve[i+r,0] - eve[i,0])
                        # miniblock = [j for j in eve[i:i+t,2] if (j>160) and (j<201)]
                        # metadata.loc[k]['Miniblock_ID'] = miniblock[0] if miniblock != [] else np.nan
                        k += 1

            elif visit_id == 'V2':
                if bids_task == "vg":
                    eve = events.copy()
                    metadata = {}
                    metadata = pd.DataFrame(metadata, index=np.arange(np.sum(events[:, 2] < 51)),
                                            columns=['Trial_type', 
                                                     'Stim_trigger',
                                                     'Stimuli_type',
                                                     'Location', 
                                                     'Response',
                                                     'Response_time'])
                    types0 = ['Filler', 'Probe']
                    type1 = ['Face', 'Object', 'Blank']
                    location = ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']
                    response = ['Seen', 'Unseen']
                    k = 0
                    for i in range(eve.shape[0]):
                        if eve[i, 2] < 51:
                            metadata.loc[k]['Stim_trigger'] = eve[i, 2]
                            t = int(eve[i + 1, 2] % 10)
                            metadata.loc[k]['Trial_type'] = types0[t]
                            if eve[i, 2] == 50:
                                metadata.loc[k]['Stimuli_type'] = type1[2]
                            else:
                                metadata.loc[k]['Stimuli_type'] = type1[eve[i, 2] // 20]
                                metadata.loc[k]['Location'] = location[eve[i + 1, 2] // 10 - 6]
                            if t == 1:
                                metadata.loc[k]['Response'] = response[int(eve[i + 4, 2] - 98)]
                                metadata.loc[k]['Response_time(s)'] = (eve[i + 4, 0] - eve[i + 3, 0]) #/ sfreq
                            k += 1
                elif bids_task == "replay":
                    eve = events.copy()
                    metadata = {}
                    metadata = pd.DataFrame(metadata, 
                                            index=np.arange(np.size(
                                                [i for i in events[:, 2] if i in list(range(101,151)) + list(range(201,251))])),
                                            columns=['Stim_trigger', 
                                                     'Stimuli_type',
                                                     'Trial_type', 
                                                     'Location', 
                                                     'Response',
                                                     'Response_time'])
                    types0 = ['Non-Target', 'Target']
                    type1 = ['Face', 'Object', 'Black']
                    # type1 = ['Face Target', 'Object Non-Target', 'Blank during Face Target',
                    #          'Object Target', 'Face Non-Target', 'Blank during Object Target']
                    location = ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']
                    response = ['Seen', 'Unseen']
                    k = 0
                    for i in range(eve.shape[0]):
                        if eve[i, 2] in list(range(101,151)) + list(range(201,251)):
                            metadata.loc[k]['Stim_trigger'] = eve[i, 2]
                            # t = int(eve[i + 1, 2] % 10)
                            if eve[i, 2] in range(101,111):
                                metadata.loc[k]['Stimuli_type'] = type1[0]
                                metadata.loc[k]['Trial_type'] = types0[1]
                            elif eve[i, 2] in range(121,131):
                                metadata.loc[k]['Stimuli_type'] = type1[1]
                                metadata.loc[k]['Trial_type'] = types0[0]
                            elif eve[i, 2] == 150:
                                metadata.loc[k]['Stimuli_type'] = type1[2]
                            elif eve[i, 2] in range(221,231):
                                metadata.loc[k]['Stimuli_type'] = type1[1]
                                metadata.loc[k]['Trial_type'] = types0[1]
                            elif eve[i, 2] in range(201,211):
                                metadata.loc[k]['Stimuli_type'] = type1[0]
                                metadata.loc[k]['Trial_type'] = types0[0]
                            elif eve[i, 2] == 250:
                                metadata.loc[k]['Stimuli_type'] = type1[2]
                            metadata.loc[k]['Location'] = location[eve[i + 1, 2] // 10 - 6]
                            if metadata.loc[k]['Trial_type'] == 'Target':
                                if 198 in eve[i:i + 4, 2]:
                                    print(eve[i:i + 4, 2])
                                    metadata.loc[k]['Response'] = response[0]
                                    metadata.loc[k]['Response_time'] = (eve[i + 4, 0] - eve[i + 3, 0]) #/ sfreq
                                else:
                                    metadata.loc[k]['Response'] = response[1]
                            k += 1
            
            # Save metadata table as csv
            bids_path_meta = bids_path_annot.copy().update(
                suffix="meta",
                extension='.csv',
                check=False)
            if not op.exists(bids_path_meta):
                bids_path_meta.fpath.parent.mkdir(exist_ok=True, parents=True)
            
            metadata.to_csv(bids_path_meta.fpath,
                            index=False)
            
    # Save code
    shutil.copy(__file__, prep_code_root)
    
    # Save report
    pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report.pdf'))


if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    run_events(subject_id, visit_id)
    