"""
===========
05. Run ICA
===========

Run indipendent component analysis.

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os.path as op
import os
import matplotlib.pyplot as plt
import shutil

from fpdf import FPDF
import mne
from mne.preprocessing import ICA
import mne_bids

from config import bids_root


def run_ica(subject_id, visit_id, has_eeg=False):
    
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
            raw = mne_bids.read_raw_bids(bids_path_annot).load_data()
            raw.info['bads'] = []
        
            # Band-pass filter raw between 1 and 40 Hz
            raw.filter(1, 40)
            
            # Downsample raw to 200 Hz
            raw.resample(200)
            
            # Concatenate raw copies
            if run == 1:
                raw_all = mne.io.concatenate_raws([raw])
            else:
                raw_all = mne.io.concatenate_raws([raw_all, raw])
            
            del raw
        
    ###################
    # ICA on MEG data #
    ###################
    
    # Prepare PDF report
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    
    # Define ICA settings
    ica = ICA(method='fastica',
              random_state=1688,
              n_components=0.99,
              verbose=True)
    
    # Run ICA on filtered raw data    
    ica.fit(raw_all,
            picks='meg',
            reject_by_annotation=True,
            verbose=True)
    
    # Plot timecourse and topography of the ICs
    # before, get the total number of ICs and divide them into n sets of 20
    n_comp_list = range(ica.n_components_)
    plot_comp_list = [n_comp_list[i:i + 20] for i in range(0, len(n_comp_list), 20)]

    for i in range(len(plot_comp_list)):
        
        # Plot timecourse
        fig = ica.plot_sources(raw_all,
                               picks=plot_comp_list[i],
                               start=100,
                               show_scrollbars=False,
                               title='ICA_MEG')
        fname_fig = op.join(prep_figure_root, 
                          "05_rAll_ica_meg_src%d.png" % i)
        fig.savefig(fname_fig)
        plt.close(fig)
    
        # Add timecourse figure to report
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, fname[:-8] + ' - MEG')
        pdf.ln(20)
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, 'Timecourse of MEG ICs', 'B', ln=1)
        pdf.image(fname_fig, 0, 45, pdf.epw)
    
        # Plot topography
        fig = ica.plot_components(title='ICA_MEG',
                                  picks=plot_comp_list[i])
        fname_fig = op.join(prep_figure_root, 
                            '05_rAll_ica_meg_cmp%d.png' % i)
        fig.savefig(fname_fig)
        plt.close(fig)
        
        # Add topography figure to report
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, fname[:-8] + ' - MEG')
        pdf.ln(20)
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, 'Topography of MEG ICs', 'B', ln=1)
        pdf.image(fname_fig, 0, 45, pdf.epw)
        
    # Save ICA file
    bids_path_ica = bids_path_annot.copy().update(
        task=None,
        run=None,
        suffix="meg_ica",
        check=False)
    if not op.exists(bids_path_ica):
        bids_path_ica.fpath.parent.mkdir(exist_ok=True, parents=True)

    ica.save(bids_path_ica)
    
    # Save report
    pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + 'MEG-report.pdf'))

    ###################
    # ICA on EEG data #
    ###################
    
    if has_eeg:
        # Prepare PDF report
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        
        # Define ICA settings
        ica = ICA(method='fastica',
              random_state=1688,
              n_components=0.99,
              verbose=True)
        
        # Run ICA on filtered raw data
        ica.fit(raw_all,
                picks='eeg',
                verbose=True)
        
        # Plot timecourse and topography of the ICs
        # Get the total number of ICs and divide them into sets of 20 ICs
        n_comp_list = range(ica.n_components_)
        plot_comp_list = [n_comp_list[i:i + 20] for i in range(0, len(n_comp_list), 20)]
    
        for i in range(len(plot_comp_list)):
            # Plot timecourse
            fig = ica.plot_sources(raw_all,
                                   picks=plot_comp_list[i],
                                   start=100,
                                   show_scrollbars=False,
                                   title='ICA_EEG')
    
            fname_fig = op.join(prep_figure_root, 
                              "05_rAll_ica_eeg_src%d.png" % i)
            fig.savefig(fname_fig)
            plt.close(fig)
        
            # Add timecourse figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:16] + ' - EEG')
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Timecourse of EEG ICs', 'B', ln=1)
            pdf.image(fname_fig, 0, 45, pdf.epw)
        
            # Plot topography
            fig = ica.plot_components(title='ICA_EEG',
                                      picks=plot_comp_list[i])
            fname_fig = op.join(prep_figure_root, 
                                '05_rAll_ica_eeg_cmp%d.png' % i)
            fig.savefig(fname_fig)
            plt.close(fig)
            
            # Add topography figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:16] + ' - EEG')
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Topography of EEG ICs', 'B', ln=1)
            pdf.image(fname_fig, 0, 45, pdf.epw)
        
        # Save ICA file
        bids_path_ica = bids_path_annot.copy().update(
            task=None,
            run=None,
            suffix="eeg_ica",
            check=False)
        if not op.exists(bids_path_ica):
            bids_path_ica.fpath.parent.mkdir(exist_ok=True, parents=True)
    
        ica.save(bids_path_ica)
        
        # Save report
        pdf.output(op.join(prep_report_root,
                           os.path.basename(__file__) + 'EEG-report.pdf'))
    # Save code
    shutil.copy(__file__, prep_code_root)

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
    run_ica(subject_id, visit_id, has_eeg)
    