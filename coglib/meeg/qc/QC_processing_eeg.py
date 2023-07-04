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

from config import (bids_root, tmin, tmax)

from matplotlib.backends.backend_pdf import PdfPages

from qc.maxwell_filtering import run_maxwell_filter
from qc.extract_events import run_events

from qc.viz_psd import viz_psd

run_part_1 = True
run_part_2 = True
run_part_3 = False

def run_qc_processing(subject_id, visit_id, has_eeg):
    # Set path to qc derivatives and create the related folders
    prep_deriv_root = op.join(bids_root, "derivatives", "qc", visit_id)
    if not op.exists(prep_deriv_root):
        os.makedirs(prep_deriv_root)
    
    print("Processing subject: %s" % subject_id)

    raw_list = list()
    events_list = list()
    metadata_list = list()

    with PdfPages(op.join(prep_deriv_root, subject_id + '_' + visit_id + '_MEG_QC.pdf')) as pdf:
        #region first page

        FirstPage = plt.figure(figsize=(8,1), dpi=108)
        FirstPage.clf()
        plt.axis('off')
        plt.text(0.5, 0.5, subject_id, transform=FirstPage.transFigure, size=16, ha="center")
        pdf.savefig(FirstPage)
        plt.close()
        
        #endregion first page

        bids_eo_path = mne_bids.BIDSPath(root=bids_root, 
                                        datatype='meg',
                                        subject=subject_id,
                                        session=visit_id,
                                        task='rest',
                                        extension='.fif'
                                        )
        # Read raw data
        raw_eo = mne_bids.read_raw_bids(bids_eo_path)

        fig_eo = plt.figure(figsize=(9,6), dpi=108)
        ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
        ax1.set_title('Resting EO spectra')
        ax2 = plt.subplot2grid((2,2), (1,0), colspan=2)

        raw_eo.plot_psd(fmax=100, ax=[ax1, ax2], picks = ['meg'])
        pdf.savefig(fig_eo)
        plt.close()

        data_path = os.path.join(bids_root, f"sub-{subject_id}", f"ses-{visit_id}", "meg")
        run = 0
        for fname in os.listdir(data_path):
            if fname.endswith(".fif") and "run" in fname:      
                run = run + 1
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
                bids_path = mne_bids.BIDSPath(
                    root=bids_root, 
                    subject=subject_id,  
                    datatype='meg',  
                    task=bids_task,
                    run=f"{run:02}",
                    session=visit_id, 
                    extension='.fif')
                
                # Read raw data
                raw = mne_bids.read_raw_bids(bids_path)

                #region PART 2a - MEG data filtering using Maxwell filters, method - defined in config
            
                if run_part_2:
                    # Find initial head position
                    if run == 1:
                        destination = raw.info['dev_head_t']

                    # # Set BIDS path
                    # bids_path_sss = mne_bids.BIDSPath(
                    #     root=op.join(bids_root, "derivatives", "preprocessing"), 
                    #     subject=subject_id,  
                    #     datatype='meg',  
                    #     task=bids_task,
                    #     run=f"{run:02}",
                    #     session=visit_id, 
                    #     suffix="sss", 
                    #     extension='.fif',
                    #     check=False)
                    
                    # # Read raw data
                    # raw_sss = mne_bids.read_raw_bids(bids_path_sss).load_data()

                    raw_sss, bad_chan = run_maxwell_filter(raw, destination, bids_path.meg_crosstalk_fpath, bids_path.meg_calibration_fpath)

                    # fig = plt.figure(figsize=(9,6), dpi=108)
                    # ax1 = plt.subplot2grid((3,4), (0,0), colspan=2)
                    # ax1.set_title('EEG spectra before filtering')  #TODO why it is not working?
                    # ax2 = plt.subplot2grid((3,4), (1,0), colspan=2)
                    # ax3 = plt.subplot2grid((3,4), (0,2), colspan=2)
                    # ax3.set_title('MEG spectra after filtering')
                    # ax4 = plt.subplot2grid((3,4), (1,2), colspan=2)

                    # # raw.plot_psd(picks=['meg'], fmin=1, fmax=100, ax=[axes[0][0], axes[1][0]])
                    # raw.plot_psd(picks=['meg'], fmin=1, fmax=100, ax=[ax1, ax2], show=False)
                    
                    # # raw_sss.plot_psd(picks=['meg'], fmin=1, fmax=100, ax=[axes[0][1], axes[1][1]])
                    # raw_sss.plot_psd(picks=['meg'], fmin=1, fmax=100, ax=[ax3, ax4], show=False)

                    # plt.axis('on')
                    # ax5 = plt.subplot2grid((3,4), (2,0), colspan=2)
                    # plt.axis('off')
                    # ax5.text(0, 0.7, 'noisy: '  + ', '.join(bad_chan['noisy']))
                    # ax5.text(0, 0.4, 'flat: '  + ', '.join(bad_chan['flat']))

                    # pdf.savefig(fig)
                    # plt.close()

                    ###########################
                    # Check EEG data quality #
                    ###########################

                    if has_eeg:
                        print("has_eeg: viz_psd")

                        fig = viz_psd(raw_sss)
                        pdf.savefig(fig)
                        plt.close()

                        line_freqs = arange(raw.info['line_freq'], raw.info["sfreq"] / 2, raw.info['line_freq'])

                        prep_params = {
                            "ref_chs": "eeg",
                            "reref_chs": "eeg",
                            "line_freqs": line_freqs,
                            "max_iterations": 4}

                        montage = raw.get_montage()
                        prep = PrepPipeline(raw_sss, prep_params, montage, ransac=True)        
                        prep.fit()
                        raw_car = prep.raw
                        raw_car.interpolate_bads(reset_bads=True)

                        fig = viz_psd(raw_car)
                        pdf.savefig(fig)
                        plt.close()
                        print("end - has_eeg: viz_psd")

                #endregion PART 2a - MEG data filtering using Maxwell filters, method - defined in config

                #region annotations

                ###########################
                # Detect ocular artifacts #
                ###########################

                if has_eeg:
                    # Resetting the EOG channel
                    eog_ch = raw_sss.copy().pick_types(meg=False, eeg=False, eog=True)
                    if len(eog_ch.ch_names) < 2:
                        raw_sss.set_channel_types({'BIO002':'eog'})
                        raw_sss.rename_channels({'BIO002': 'EOG002'})
                    
                    # Find EOG events
                    eog_events = mne.preprocessing.find_eog_events(raw_sss)
                    onsets = (eog_events[:, 0] - raw_sss.first_samp) / raw_sss.info['sfreq'] - 0.25
                    durations = [0.5] * len(eog_events)
                    descriptions = ['Blink'] * len(eog_events)
                    
                    # Annotate events
                    annot_blink = mne.Annotations(
                        onsets, 
                        durations,
                        descriptions)

                ###########################
                # Detect muscle artifacts #
                ###########################
                threshold_muscle = 7

                # Notch filter
                raw_muscle = raw_sss.copy().notch_filter([50, 100])
                
                # Choose one channel type, if there are axial gradiometers and magnetometers,
                # select magnetometers as they are more sensitive to muscle activity.
                annot_muscle, scores_muscle = annotate_muscle_zscore(
                    raw_muscle, 
                    ch_type="mag", 
                    threshold=threshold_muscle, 
                    min_length_good=0.3,
                    filter_freq=[110, 140])
                
                #################
                # Detect breaks #
                #################
                
                # Get events
                # events, event_id = mne.events_from_annotations(raw_sss)
                
                # Detect breaks based on events
                # annot_break = mne.preprocessing.annotate_break(
                #     raw=raw_sss,
                #     events=events,
                #     min_break_duration=15.0)
                
                ###########################
                
                # Contatenate blink and muscle artifact annotations
                if has_eeg:
                    annot_artifact = annot_blink + annot_muscle
                else:
                    annot_artifact = annot_muscle
                annot_artifact = mne.Annotations(onset = annot_artifact.onset + raw_sss._first_time,
                                                    duration = annot_artifact.duration,
                                                    description = annot_artifact.description,
                                                    orig_time = raw_sss.info['meas_date'])
                
                # Add artifact annotations in raw_sss
                # raw_sss.set_annotations(raw_sss.annotations + annot_artifact + annot_break)
                raw_sss.set_annotations(raw_sss.annotations + annot_artifact)
                
                #endregion annotations

                events, metadata = run_events(raw_sss, visit_id)
                # Show events
                fig = mne.viz.plot_events(events)
                pdf.savefig(fig)
                plt.close()

                raw_list.append(raw_sss)
                events_list.append(events)
                metadata_list.append(metadata)

        if run_part_3:
            raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
            del raw_list
                
            # Concatenate metadata tables
            metadata = pd.concat(metadata_list)
            # metadata.to_csv(op.join(out_path, file_name[0:14] + 'ALL-meta.csv'), index=False)
            
            # Select sensor types
            picks = mne.pick_types(raw.info,
                                meg = True,
                                eeg = has_eeg,
                                stim = True,
                                eog = has_eeg,
                                ecg = has_eeg,
                                )
            
            # Set trial-onset event_ids
            if visit_id == 'V1':
                events_id = {}
                types = ['face','object','letter','false']
                for j,t in enumerate(types):
                    for i in range(1,21):
                        events_id[t+str(i)] = i + j * 20
            elif visit_id == 'V2':
                events_id = {}
                events_id['blank'] = 50
                types = ['face','object']
                for j,t in enumerate(types):
                    for i in range(1,11):
                        events_id[t+str(i)] = i + j * 20

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
                                reject_by_annotation=True,
                                verbose=True)
                
            del raw
            
            # Add metadata
            epochs.metadata = metadata
            
            # Get rejection thresholds
            reject = get_rejection_threshold(epochs, 
                                            ch_types=['mag', 'grad'], #'eeg'], #TODO: eeg not use for epoch rejection
                                            decim=2)
            
            # Drop bad epochs based on peak-to-peak magnitude
            epochs.drop_bad(reject=reject)

            # Plot percentage of rejected epochs per channel
            fig1 = epochs.plot_drop_log()
            pdf.savefig(fig1)
            plt.close()


if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    run_qc_processing(subject_id, visit_id)