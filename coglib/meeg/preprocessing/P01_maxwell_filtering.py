"""
===================================
01. Maxwell filter using MNE-python
===================================

The data are Maxwell filtered using tSSS/SSS.

It is critical to mark bad channels before Maxwell filtering.

@author: Oscar Ferrante oscfer88@gmail.com

"""  # noqa: E501

import os.path as op
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

from fpdf import FPDF
import mne
from mne.preprocessing import find_bad_channels_maxwell
import mne_bids

from config import bids_root


def run_maxwell_filter(subject_id, visit_id, record="run"):
    
    # Prepare PDF report
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    
    # Set path to preprocessing derivatives and create the related folders
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    if not op.exists(prep_deriv_root):
        os.makedirs(prep_deriv_root)
    prep_figure_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(prep_figure_root):
        os.makedirs(prep_figure_root)
    prep_report_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "reports")
    if not op.exists(prep_report_root):
        os.makedirs(prep_report_root)
    prep_code_root = op.join(prep_deriv_root,
                             f"sub-{subject_id}",f"ses-{visit_id}","meg",
                             "codes")
    if not op.exists(prep_code_root):
        os.makedirs(prep_code_root)
    
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
            
            # Set split
            if len([f for f in os.listdir(data_path) if op.splitext(fname)[0][:-3] in f and f.endswith(".fif")]) > 1:
                split = 1
            else:
                split = None
            
            # Set BIDS path
            bids_path = mne_bids.BIDSPath(
                root=bids_root, 
                subject=subject_id,  
                datatype='meg',  
                task=bids_task,
                run=run,
                session=visit_id, 
                split=split,
                extension='.fif')
            
            # Read raw data
            raw = mne_bids.read_raw_bids(bids_path)
            
            # Find initial head position
            if run in ["01", None]:
                destination = raw.info['dev_head_t']
            
            # Detect bad channels
            raw.info['bads'] = []
            raw_check = raw.copy()
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, 
                cross_talk=bids_path.meg_crosstalk_fpath, 
                calibration=bids_path.meg_calibration_fpath,
                return_scores=True,
                verbose=True)
            raw.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
            
            # Mark bad channels in BIDS events
            mne_bids.mark_channels(ch_names=raw.info['bads'], 
                                       bids_path=bids_path, 
                                       status='bad',
                                       verbose=False)
            
            # Visualize the scoring used to classify channels as noisy or flat
            ch_type = 'grad'
            fig = viz_badch_scores(auto_scores, ch_type)
            fname_fig = op.join(prep_figure_root,
                                "01_%sr%s_badchannels_%sscore.png" % (bids_task,run,ch_type))
            fig.savefig(fname_fig)
            plt.close(fig)
            ch_type = 'mag'
            fig = viz_badch_scores(auto_scores, ch_type)
            fname_fig = op.join(prep_figure_root,
                                "01_%sr%s_badchannels_%sscore.png" % (bids_task,run,ch_type))
            fig.savefig(fname_fig)
            plt.close(fig)
            
            # Fix Elekta magnetometer coil types
            raw.fix_mag_coil_types()
            
            # Set coordinate frame
            if subject_id == 'empty':
                coord_frame = 'meg'
            else:
                coord_frame = 'head'
            
            # Perform tSSS/SSS and Maxwell filtering
            raw_sss = mne.preprocessing.maxwell_filter(
                raw,
                cross_talk=bids_path.meg_crosstalk_fpath,
                calibration=bids_path.meg_calibration_fpath,
                st_duration=None,
                origin='auto',
                destination=destination,  #align head location to first run
                coord_frame=coord_frame, 
                verbose=True)
            
            # Show original and filtered signals
            fig = raw.copy().pick(['meg']).plot(duration=5,
                                                start=100,
                                                butterfly=True)        
            fname_fig = op.join(prep_figure_root,
                                '01_%sr%s_plotraw.png' % (bids_task,run))
            fig.savefig(fname_fig)
            plt.close(fig)
            fig = raw_sss.copy().pick(['meg']).plot(duration=5,
                                                    start=100,
                                                    butterfly=True)
            fname_fig = op.join(prep_figure_root,
                                '01_%sr%s_plotrawsss.png' % (bids_task,run))
            fig.savefig(fname_fig)
            plt.close(fig)
            
            # Show original and filtered power
            fig1 = raw.plot_psd(picks = ['meg'],fmin = 1,fmax = 100)
            fname_fig1 = op.join(prep_figure_root,
                                '01_%sr%s_plot_psd_raw100.png' % (bids_task,run))
            fig1.savefig(fname_fig1)
            plt.close(fig1)
            fig2 = raw_sss.plot_psd(picks = ['meg'],fmin = 1,fmax = 100)
            fname_fig2 = op.join(prep_figure_root,
                                '01_%sr%s_plot_psd_raw100sss.png' % (bids_task,run))
            fig2.savefig(fname_fig2)
            plt.close(fig2)
            
            # Add figures to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:-8])
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Power Spectrum of Raw MEG Data', 'B', ln=1)
            pdf.image(fname_fig1, 0, 45, pdf.epw)
            pdf.ln(120)
            pdf.cell(0, 10, 'Power Spectrum of Filtered MEG Data', 'B', ln=1)
            pdf.image(fname_fig2, 0, 175, pdf.epw)
            
            # Save filtered data
            bids_path_sss = bids_path.copy().update(
                root=prep_deriv_root,
                split=None,
                suffix="sss",
                check=False)
            if not op.exists(bids_path_sss):
                bids_path_sss.fpath.parent.mkdir(exist_ok=True, parents=True)
    
            raw_sss.save(bids_path_sss, overwrite=True)
            
            # Add note about reconstructed sensors to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, "Reconstructed sensors:")
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'bad MEG sensors: %s' % raw.info['bads'], 'B', ln=1)
    
    # Save code
    shutil.copy(__file__, prep_code_root)
        
    # Save report
    if record == "rest":
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report_rest.pdf'))
    else:
        pdf.output(op.join(prep_report_root,
                           os.path.basename(__file__) + '-report.pdf'))


def viz_badch_scores(auto_scores, ch_type):
    fig, ax = plt.subplots(1, 4, figsize=(12, 8))
    fig.suptitle(f'Automated noisy/flat channel detection: {ch_type}',
                  fontsize=16, fontweight='bold')
    
    #### Noisy channels ####
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  #the windows that were evaluated
    
    # Label each segment by its start and stop time (3 digits / 1 ms precision)
    bin_labels = [f'{start:3.3f} - {stop:3.3f}' 
                  for start, stop in bins]
    
    # Store  data in DataFrame
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))
    
    # First, plot the raw scores
    sns.heatmap(data=data_to_plot, 
                cmap='Reds', 
                cbar=False,
                # cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[0].set_title('Noisy: All Scores', fontweight='bold')

    # Second, highlight segments that exceeded the 'noisy' limit
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),
                cmap='Reds', 
                cbar=True, 
                # cbar_kws=dict(label='Score'), 
                ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[1].set_title('Noisy: Scores > Limit', fontweight='bold')
    
    #### Flat channels ####
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_flat'][ch_subset]
    limits = auto_scores['limits_flat'][ch_subset]
    bins = auto_scores['bins']  #the windows that were evaluated
    
    # Label each segment by its start and stop time (3 digits / 1 ms precision)
    bin_labels = [f'{start:3.3f} - {stop:3.3f}' 
                  for start, stop in bins]
    
    # Store  data in DataFrame
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))
    
    # First, plot the raw scores
    sns.heatmap(data=data_to_plot, 
                cmap='Reds', 
                cbar=False,
                # cbar_kws=dict(label='Score'),
                ax=ax[2])
    [ax[2].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[2].set_title('Flat: All Scores', fontweight='bold')

    # Second, highlight segments that exceeded the 'noisy' limit
    sns.heatmap(data=data_to_plot,
                vmax=np.nanmax(limits),
                cmap='Reds', 
                cbar=True,
                # cbar_kws=dict(label='Score'), 
                ax=ax[3])
    [ax[3].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
        for x in range(1, len(bins))]
    ax[3].set_title('Flat: Scores > Limit', fontweight='bold')

    # Fit figure title to not overlap with the subplots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    run_maxwell_filter(subject_id, visit_id)
