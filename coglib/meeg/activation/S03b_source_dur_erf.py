"""
================
S07. Source localization of ERF duration activity

based on S04 Source Localization of frequency-band-specific duration activity  @author: Oscar Ferrante oscfer88@gmail.com
================


@author: Ling Liu  ling.liu@pku.edu.cn

"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import json
import pandas as pd

import mne
import mne_bids
from mne.minimum_norm import apply_inverse

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root, sfreq


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA124',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--method',
                    type=str,
                    default='dspm',
                    help='method used for the inverse solution ("lcmv", "dics", "dspm")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
inv_method = opt.method  #this variable is used only to set the output filename

debug = False


factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant non-target','Irrelevant'],
              ['500ms', '1000ms', '1500ms']]


def run_source_dur(subject_id, visit_id):
    # Set directory paths
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur_ERF")
    if not op.exists(source_deriv_root):
        os.makedirs(source_deriv_root)
    source_figure_root =  op.join(source_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(source_figure_root):
        os.makedirs(source_figure_root)
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
    
    epochs_all = mne.read_epochs(bids_path_epo.fpath,
                             preload=False)
    
    # Pick trials
    epochs_all = epochs_all['Task_relevance in ["Relevant non-target", "Irrelevant"]']
    if debug:
        epochs_all = epochs_all[0:100]
        
    # Select sensor type
    epochs_all.load_data().pick('meg')
    
    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs = epochs_all.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs.filter(0, 30, n_jobs=-1)
    
    epochs_rs.crop(tmin=-0.5, tmax=2,include_tmax=True, verbose=None)
    
    # Run baseline correction
    b_tmin = -.5
    b_tmax = 0.
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)
    
    # Read labels from FS parc
    if subject_id in ['SA102','SA104','SA110', 'SA111','SA152']:
        labels_atlas = mne.read_labels_from_annot(
            "fsaverage", 
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)
    else:
        labels_atlas = mne.read_labels_from_annot(
            "sub-"+subject_id, 
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)
    
    # labels_atlas_names = [l.name for l in labels_atlas]
    
    # Read GNW and IIT ROI list
    f = open(op.join(rois_deriv_root,
                      'iit_gnw_rois.json'))
    gnw_iit_rois = json.load(f)
    
    # Create labels for selected ROIs
    labels = {}
    if subject_id in ['SA102','SA104','SA110', 'SA111','SA152']:
        for lab in gnw_iit_rois['surf_labels']['iit_1']:
            lab = lab.replace('&','_and_')  # Fix the label name to match the template one
            print(lab)
            labels["iit_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])
        
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            lab = lab.replace('&','_and_')  # Fix the label name to match the template one
            print(lab)
            labels["gnw_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])
    else:
        for lab in gnw_iit_rois['surf_labels']['iit_1']:
            print(lab)
            labels["iit_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])
        
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            print(lab)
            labels["gnw_"+lab] = np.sum([l for l in labels_atlas if lab in l.name])
    
    # Merge all labels in a single one separatelly for GNW and IIT 
    labels['gnw_all'] = np.sum([l for l_name, l in labels.items() if 'gnw' in l_name])
    labels['iit_all'] = np.sum([l for l_name, l in labels.items() if 'iit' in l_name])
    
    # Compute rank
    rank = mne.compute_rank(epochs_rs, 
                            tol=1e-6, 
                            tol_kind='relative')
    
    # Read forward model
    bids_path_fwd = bids_path_epo.copy().update(
            root=fwd_deriv_root,
            task=None,
            suffix="surface_fwd",
            extension='.fif',
            check=False)
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)
    
    # Compute covariance matrices
    base_cov = mne.compute_covariance(epochs_rs, 
                                   tmin=b_tmin, 
                                   tmax=b_tmax, 
                                   method='empirical', 
                                   rank=rank)
    
    active_cov = mne.compute_covariance(epochs_rs, 
                                 tmin=0,
                                 tmax=None,
                                 method='empirical', 
                                 rank=rank)
    common_cov = base_cov + active_cov
    
    # Make inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs_rs.info,
        fwd, 
        common_cov,
        loose=.2,
        depth=.8,
        fixed=False,
        rank=rank,
        use_cps=True)
    
    # Find all combinations between variables' levels
    if len(factor) == 1:
        cond_combs = list(itertools.product(conditions[0]))
    if len(factor) == 2:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1]))
    if len(factor) == 3:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1],
                                            conditions[2]))
    

    
    # Loop over conditions of interest
    for cond_comb in cond_combs:
        print("\nAnalyzing %s: %s" % (factor, cond_comb))
        
        # Select epochs
        if len(factor) == 1:
            epochs = epochs_rs['%s == "%s"' % (
                factor[0], cond_comb[0])]
            fname = cond_comb[0]
        if len(factor) == 2:
            epochs = epochs_rs['%s == "%s" and %s == "%s"' % (
                factor[0], cond_comb[0],
                factor[1], cond_comb[1])]
            fname = cond_comb[0] + "_" + cond_comb[1]
        if len(factor) == 3:
            epochs = epochs_rs['%s == "%s" and %s == "%s" and %s == "%s"' % (
                factor[0], cond_comb[0], 
                factor[1], cond_comb[1], 
                factor[2], cond_comb[2])]
            fname = cond_comb[0] + "_" + cond_comb[1] + "_" + cond_comb[2]
        
        
        # Get evoked response by computing the epoch average
        evoked = epochs.average()
        # Compute inverse solution for each epoch
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        stc = apply_inverse(evoked, inverse_operator, 1. / lambda2, 'dSPM', pick_ori="normal")
        
        # # Save stc
        # bids_path_source = bids_path_epo.copy().update(
        #         root=source_deriv_root,
        #         suffix=f"desc-{fname},ERF_stc",
        #         extension='.stc',
        #         check=False)
            
        # stc.save(bids_path_source)
        
        
        
        # Loop over labels        
        for label_name, label in labels.items():
            print(f"label: {label_name}")
                        
            # extract time course in label with pca_flip mode
            src = inverse_operator['src']
            # Extract time course data
            times = epochs.times
            tcs = stc.extract_label_time_course(label,src,mode='mean')
            data = tcs[0]
            
            # Convert to root mean square
            data = np.sqrt((np.array(data)**2))
            
            # Create and save a tsv table with the label time course data
            df = pd.DataFrame({
                'times': times,
                'data': data})
            
            bids_path_source = bids_path_epo.copy().update(
                root=source_deriv_root,
                suffix=f"desc-{fname},ERF,{label_name}_datatable",
                extension='.tsv',
                check=False)
            df.to_csv(bids_path_source.fpath, sep="\t")
            
            # Show results
            # Apply low-pass filter
            dataf = apply_fir_filter(data)
            
            # Set time limits
            tmin = (np.abs(times - -.5)).argmin()
            tmax = (np.abs(times - 2.)).argmin()
            
            # Plot
            plt.plot(times[tmin:tmax], dataf[tmin:tmax])
            plt.xlabel('Time (ms)')
            plt.ylabel('rms')
            plt.title(f'ERF in {label_name}:\n{fname}')
            
            # Save figure
            fname_fig = op.join(source_figure_root, 
                                f'source_dur_{fname}_ERF_{label_name}.png')
            plt.savefig(fname_fig)
            plt.close('all')
        
        del stc, evoked


def apply_fir_filter(time_series, fs=1000, fc=30, num_taps=101):
    # Compute the center of the filter taps
    center = (num_taps - 1) // 2
    
    # Define the filter coefficients for a low-pass filter with the given cutoff frequency
    filter_coeffs = np.sinc(2 * fc / fs * np.arange(center - num_taps + 1, center + 1))

    # Reverse the filter coefficients since convolution uses a flipped version
    filter_coeffs = np.flip(filter_coeffs)
    
    # Apply the filter to the time series using convolution
    filtered_series = np.convolve(time_series, filter_coeffs, mode='same')
    
    return filtered_series



if __name__ == '__main__':
    run_source_dur(subject_id, visit_id)