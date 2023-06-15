"""
===================================
02. Find bad EEG sensors
===================================

EEG bad sensors are detected using a revisited version of 
the PREP pipeline https://doi.org/10.3389/fninf.2015.00016

@author: Oscar Ferrante oscfer88@gmail.com

"""  # noqa: E501

import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import shutil

from fpdf import FPDF
from mne.time_frequency import psd_multitaper
import mne_bids
from pyprep.prep_pipeline import PrepPipeline

from config import  bids_root


def find_bad_eeg(subject_id, visit_id, record="run", has_eeg=False):
    
    # Check whether there are EEG data for this participant and stop if not    
    if not has_eeg:
        raise ValueError("Error: there is no EEG recording for this participant (%s)" % subject_id)
    
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
    
    # Loop over runs
    data_path = os.path.join(bids_root,f"sub-{subject_id}",f"ses-{visit_id}","meg")
    
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".json") and record in fname:
            
            # Set run
            if "run" in fname:
                run = f"{int(fname[-10]):02}"
            elif "rest" in fname:
                run = None
            print("  Run: %s" % run)
            
            # Set task
            if 'dur' in fname:
                bids_task = 'dur'
            elif 'vg' in fname:
                bids_task = 'vg'
            elif 'replay' in fname:
                bids_task = 'replay'
            elif "rest" in fname:
                bids_task = "rest"
            else:
                raise ValueError("Error: could not find the task for %s" % fname)
            
            # Set BIDS path
            bids_path_sss = mne_bids.BIDSPath(
                root=prep_deriv_root, 
                subject=subject_id,  
                datatype='meg',  
                task=bids_task,
                run=run,
                session=visit_id, 
                suffix="sss", 
                extension='.fif',
                check=False)
            
            # Read raw data
            raw = mne_bids.read_raw_bids(bids_path_sss).load_data()
            
            # Read EEG electrode layout
            if run in ["01", None] and bids_task in ['rest', 'dur', 'replay']:
                montage = raw.get_montage()
                
                # Plot montage (EEG layout)
                fig = montage.plot(kind='topomap', show_names=False)
                fname_fig = op.join(prep_figure_root,
                                '02_rAll_eeg_montage.png')
                fig.savefig(fname_fig)
                plt.close()
                
                # Add montage figure to the report
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(0, 10, fname[:-8])
                pdf.ln(20)
                pdf.set_font('helvetica', 'B', 12)
                pdf.cell(0, 10, 'EEG Montage', 'B', ln=1)
                pdf.image(fname_fig, 0, 45, pdf.epw*.8)
            
            # Set line freq and its harmonics
            line_freqs = np.arange(raw.info['line_freq'], raw.info["sfreq"] / 2, raw.info['line_freq'])
            
            # Set prep params
            prep_params = {
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs": line_freqs,
                "max_iterations": 4}
            
            # Run Prep pipeline
            prep = PrepPipeline(raw, 
                                prep_params, 
                                montage,
                                ransac=True)
            prep.fit()
            
            # Print results
            print("Bad channels: {}".format(prep.interpolated_channels))
            print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))
            
            # Extract raw
            raw_car = prep.raw
            
            # Interpolate bad channels left by the prep method
            raw_car.interpolate_bads(reset_bads=True)
            
            # Mark bad channels in the raw bids folder
            bids_path = mne_bids.BIDSPath(
                root=bids_root, 
                subject=subject_id,  
                datatype='meg',  
                task=bids_task,
                run=run,
                session=visit_id, 
                extension='.fif')
            
            mne_bids.mark_channels(ch_names=(prep.interpolated_channels+prep.still_noisy_channels), 
                                       bids_path=bids_path, 
                                       status='bad',
                                       verbose=False)
            
            # Save filtered data
            bids_path_car = bids_path_sss.copy().update(
                suffix="car",
                check=False)
            
            raw_car.save(bids_path_car, overwrite=True)
            
            # Plot EEG data
            fig = raw.copy().pick('eeg').plot(bad_color=(1., 0., 0.),
                                              scalings = dict(eeg=10e-5),
                                              duration=5,
                                              start=100)
            fname_fig = op.join(prep_figure_root,
                                '02_%sr%s_bad_egg_0raw.png' % (bids_task,run))
            fig.savefig(fname_fig)
            plt.close()
            
            # Plot EEG power spectrum
            fig1 = viz_psd(raw)
            fname_fig1 = op.join(prep_figure_root,
                                '02_%sr%s_bad_egg_0pow.png' % (bids_task,run))
            fig1.savefig(fname_fig1)
            plt.close()
            
            # Add figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:-8])
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Power Spectrum of Raw EEG Data', 'B', ln=1)
            pdf.image(fname_fig1, 0, 45, pdf.epw*.8)
            
            # Plot re-referenced EEG data
            fig = raw_car.copy().pick('eeg').plot(bad_color=(1., 0., 0.),
                                              scalings = dict(eeg=10e-5),
                                              duration=5,
                                              start=100)
            fname_fig = op.join(prep_figure_root,
                                '02_%sr%s_bad_egg_3refer.png' % (bids_task,run))
            fig.savefig(fname_fig)
            plt.close()
            
            # Plot re-referenced EEG power spectrum
            fig1 = viz_psd(raw_car)
            fname_fig1 = op.join(prep_figure_root,
                                '02_%sr%s_bad_egg_Ipow.png' % (bids_task,run))
            fig1.savefig(fname_fig1)
            plt.close()
            
            # Add figures to report
            pdf.ln(120)
            pdf.cell(0, 10, 'Power Spectrum of Interpolated/Re-referenced EEG Data', 'B', ln=1)
            pdf.image(fname_fig1, 0, 175, pdf.epw*.8)
            
            # Add note about bad channels
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, "Bad channels:")
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Before prep: %s' % prep.interpolated_channels, 'B', ln=1)
            pdf.cell(0, 10, 'After prep: %s' % prep.still_noisy_channels, 'B', ln=1)
            pdf.cell(0, 10, 'After intepolation: %s' % raw_car.info['bads'], 'B', ln=1)
            
    # Save code
    shutil.copy(__file__, prep_code_root)
    
    # Save report
    if record == "rest":
       pdf.output(op.join(prep_report_root,
                          os.path.basename(__file__) + '-report_rest.pdf'))
    else:
       pdf.output(op.join(prep_report_root,
                          os.path.basename(__file__) + '-report.pdf'))


def viz_psd(raw):
    # Compute averaged power
    psds, freqs = psd_multitaper(raw,fmin = 1,fmax = 40, picks=['eeg'])
    psds = np.sum(psds,axis = 1)
    psds = 10. * np.log10(psds)
    # Show power spectral density plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    raw.plot_psd(picks = ["eeg"], 
                  fmin = 1,fmax = 40,
                  ax=ax[0])
    # Normalize (z-score) channel-specific average power values 
    psd = {}
    psd_zscore = zscore(psds)
    for i in range(len(psd_zscore)):
        psd["EEG%03d"%(i+1)] = psd_zscore[i]
    # Plot chennels ordered by power
    ax[1].bar(sorted(psd, key=psd.get,reverse = True),sorted(psd.values(),reverse = True),width = 0.5)
    labels = sorted(psd, key=psd.get,reverse = True)
    ax[1].set_xticklabels(labels, rotation=90)
    ax[1].annotate("Average power: %.2e dB"%(np.average(psds)),(27,np.max(psd_zscore)*0.9),fontsize = 'x-large')
    return fig
    
def input_bool(message):
    value = input(message)
    if value == "True":
        return True
    if value == "False":
        return False
    

if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    has_eeg = input_bool("Has this recording EEG data? (True or False)\n>>> ")
    find_bad_eeg(subject_id, visit_id, has_eeg)
    
