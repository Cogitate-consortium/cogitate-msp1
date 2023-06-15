# -*- coding: utf-8 -*-
"""
===================================
Co00. GED spatial filter
===================================

Create category-specific spatial filters through generalized eigendecomposition

@author: Oscar Ferrante oscfer88@gmail.com
"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import stats as stats
import argparse
import pickle

import mne
import mne_bids

from config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--labels',
                    nargs='+',
                    default=['fusifor'],
                    help='name of the label to which contrain the spatial filter (e.g., "fusiform"')
parser.add_argument('--parc',
                    type=str,
                    default='aparc',
                    help='name of the parcellation atlas to use for contraining the spatial filter (e.g., "aparc", "aparc.a2009s")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
visit_id = 'V1'
label_list = opt.labels
label_name = ''.join(label_list)
parc = opt.parc

act_win_tmin = 0.
act_win_tmax = .5

debug = False


# Set participant list
phase = 3

if debug:
    sub_list = ["SA124", "SA124"]
else:
    # Read the .txt file
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")


def ged_ga():
    # Set derivatives paths
    if act_win_tmin == 0. and act_win_tmax == .5:
        ged_deriv_root = op.join(bids_root, "derivatives", "ged")
    else:
        ged_deriv_root = op.join(bids_root, "derivatives", "ged", f"_{act_win_tmin}-{act_win_tmax}")

    ged_figure_root =  op.join(ged_deriv_root,
                                f"sub-groupphase{phase}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(ged_figure_root):
        os.makedirs(ged_figure_root)
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    # Loop over subjects
    ged_facFilt_facCond_ts = []
    ged_objFilt_objCond_ts = []
    ged_objFilt_facCond_ts = []
    ged_facFilt_objCond_ts = []
    for sub in sub_list:
        print("\nLoading data for subject:", sub, "\nand label(s):", label_name)
            
        # Read GED filters' time courses
        bids_path_ged = mne_bids.BIDSPath(
            root=ged_deriv_root, 
            subject=sub,  
            datatype='meg',  
            task=bids_task,
            session=visit_id, 
            suffix=f'desc-{label_name},facFilt_facCond_compts',
            extension='.npy',
            check=False)
        ged_facFilt_facCond_ts.append(np.load(bids_path_ged.fpath))
    
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},objFilt_objCond_compts')
        ged_objFilt_objCond_ts.append(np.load(bids_path_ged.fpath))
        
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},objFilt_facCond_compts')
        ged_objFilt_facCond_ts.append(np.load(bids_path_ged.fpath))
        
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},facFilt_objCond_compts')
        ged_facFilt_objCond_ts.append(np.load(bids_path_ged.fpath))
    
    # Average trials within participants (ev=evoked)
    ged_facFilt_facCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_facFilt_facCond_ts]
    ged_objFilt_objCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_objFilt_objCond_ts]
    ged_objFilt_facCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_objFilt_facCond_ts]
    ged_facFilt_objCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_facFilt_objCond_ts]

    # Low-pass filter the data
    print("\nLow-pass filtering the data...")
    order = 6
    fs = 1000.0  # sample rate (Hz)
    cutoff = 30.0
    b, a = ss.butter(order, 
                        cutoff, 
                        fs=fs, 
                        btype='low', 
                        analog=False)
    
    ged_facFilt_facCond_ts_lp = [ss.lfilter(b, a, ged) for ged in ged_facFilt_facCond_ts_ev]
    ged_objFilt_objCond_ts_lp = [ss.lfilter(b, a, ged) for ged in ged_objFilt_objCond_ts_ev]
    ged_objFilt_facCond_ts_lp = [ss.lfilter(b, a, ged) for ged in ged_objFilt_facCond_ts_ev]
    ged_facFilt_objCond_ts_lp = [ss.lfilter(b, a, ged) for ged in ged_facFilt_objCond_ts_ev]
    
    # Compute root mean square
    print("\nComputing RMS...")
    ged_facFilt_facCond_ts_rms = [np.sqrt((np.array(ged)**2)) for ged in ged_facFilt_facCond_ts_lp]
    ged_objFilt_objCond_ts_rms = [np.sqrt((np.array(ged)**2)) for ged in ged_objFilt_objCond_ts_lp]
    ged_objFilt_facCond_ts_rms = [np.sqrt((np.array(ged)**2)) for ged in ged_objFilt_facCond_ts_lp]
    ged_facFilt_objCond_ts_rms = [np.sqrt((np.array(ged)**2)) for ged in ged_facFilt_objCond_ts_lp]
    
    # Baseline correction
    print("\nCorrecting for the baseline...")
    baseline_win = [-.5, 0]
    times = np.arange(-1, 2.501, .001)
    
    imin = (np.abs(times - baseline_win[0])).argmin()
    imax = (np.abs(times - baseline_win[1])).argmin()
    
    ged_facFilt_facCond_ts_bc = []
    for ged in ged_facFilt_facCond_ts_rms:
        mean_ts = np.mean(ged[..., imin:imax], axis=-1, keepdims=True)
        ged -= mean_ts
        ged /= mean_ts
        ged_facFilt_facCond_ts_bc.append(ged)
    
    ged_objFilt_objCond_ts_bc = []
    for ged in ged_objFilt_objCond_ts_rms:
        mean_ts = np.mean(ged[..., imin:imax], axis=-1, keepdims=True)
        ged -= mean_ts
        ged /= mean_ts
        ged_objFilt_objCond_ts_bc.append(ged)
    
    ged_objFilt_facCond_ts_bc = []
    for ged in ged_objFilt_facCond_ts_rms:
        mean_ts = np.mean(ged[..., imin:imax], axis=-1, keepdims=True)
        ged -= mean_ts
        ged /= mean_ts
        ged_objFilt_facCond_ts_bc.append(ged)
    
    ged_facFilt_objCond_ts_bc = []
    for ged in ged_facFilt_objCond_ts_rms:
        mean_ts = np.mean(ged[..., imin:imax], axis=-1, keepdims=True)
        ged -= mean_ts
        ged /= mean_ts
        ged_facFilt_objCond_ts_bc.append(ged)

    # Average over participants
    print("\nComputing grandaverage...")
    ged_facFilt_facCond_ts_ga = np.mean(ged_facFilt_facCond_ts_bc, axis=0)
    ged_objFilt_objCond_ts_ga = np.mean(ged_objFilt_objCond_ts_bc, axis=0)
    ged_objFilt_facCond_ts_ga = np.mean(ged_objFilt_facCond_ts_bc, axis=0)
    ged_facFilt_objCond_ts_ga = np.mean(ged_facFilt_objCond_ts_bc, axis=0)
    
    # Save averaged data
    bids_path_ged = mne_bids.BIDSPath(
        root=ged_deriv_root, 
        subject=f"groupphase{phase}",
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix=f'desc-{label_name}_compts',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, 
            np.concatenate(
                [ged_facFilt_facCond_ts_ga,
                 ged_objFilt_objCond_ts_ga,
                 ged_objFilt_facCond_ts_ga,
                 ged_facFilt_objCond_ts_ga]))
    
    # Set limits used to crop edges in figures
    print("\nRemoving edges...")
    t_win = [-.5, 2.]
    
    # Plot filter time course
    print("\nPlotting...")
    fig, axs = plt.subplots(2)
    axs[0].plot(times, ged_facFilt_facCond_ts_ga, 
             label='face', color='b', linestyle='-')
    axs[0].plot(times, ged_facFilt_objCond_ts_ga, 
             label='object', color='r', linestyle='-')
    axs[1].plot(times, ged_objFilt_facCond_ts_ga, 
             label='face', color='b', linestyle='-')
    axs[1].plot(times, ged_objFilt_objCond_ts_ga, 
             label='object', color='r', linestyle='-')
    
    axs[0].set_title('Grandaverage evoked-activity in face-selective filter')
    axs[1].set_title('Grandaverage evoked-activity in object-selective filter')
    for ax in axs:
        ax.legend()
        ax.legend()
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('RMS amplitude (a.u.)')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlim(t_win)
    plt.tight_layout()
    
    # Save figure
    print("\nSaving figures...")
    fname_fig = op.join(ged_figure_root,
                        f"ged_filter_ts_{label_name}.png")
    fig.savefig(fname_fig)
    plt.close(fig)
    
    # Compute statistics
    pval = 0.05  # arbitrary
    n_observations = len(sub_list)
    df = n_observations - 1  # degrees of freedom for the test
    threshold = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    T_obs_facFilt, clusters_facFilt, cluster_p_values_facFilt, H0_facFilt = \
        mne.stats.permutation_cluster_1samp_test(
            np.array(ged_facFilt_facCond_ts_rms) - np.array(ged_facFilt_objCond_ts_rms),
            threshold=threshold, 
            out_type='mask')
    
    T_obs_objFilt, clusters_objFilt, cluster_p_values_objFilt, H0_objFilt = \
        mne.stats.permutation_cluster_1samp_test(
            np.array(ged_objFilt_objCond_ts_rms) - np.array(ged_objFilt_facCond_ts_rms),
            threshold=threshold, 
            out_type='mask')
    
    # Save significant clusters
    bids_path_mask = bids_path_ged.copy().update(
        subject=f"groupphase{phase}",
        suffix=f"desc-{label_name},facFilt_clusters",
        extension=".pkl",
        check=False)
    with open(bids_path_mask.fpath, 'wb') as file:
        pickle.dump(clusters_facFilt, file)
    
    bids_path_mask = bids_path_mask.update(
        suffix=f"desc-{label_name},objFilt_clusters",)
    with open(bids_path_mask.fpath, 'wb') as file:
        pickle.dump(clusters_objFilt, file)
        
    # Plot difference between conditions with significant clusters
    fig, axs = plt.subplots(2)
    axs[0].plot(times, 
                ged_facFilt_facCond_ts_ga - ged_facFilt_objCond_ts_ga, 
                color='k', linestyle='-')
    axs[1].plot(times, 
                ged_objFilt_objCond_ts_ga - ged_objFilt_facCond_ts_ga, 
                color='k', linestyle='-')
        
    for i_c, c in enumerate(clusters_facFilt):
        c = c[0]
        if cluster_p_values_facFilt[i_c] < 0.05:
            axs[0].axvspan(times[c.start], times[c.stop - 1],
                            color='r', alpha=0.3)
        # else:
        #     axs[0].axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
        #                 alpha=0.3)
    for i_c, c in enumerate(clusters_objFilt):
        c = c[0]
        if cluster_p_values_objFilt[i_c] < 0.05:
            axs[1].axvspan(times[c.start], times[c.stop - 1],
                            color='r', alpha=0.3)
        # else:
        #     axs[1].axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
        #                 alpha=0.3)
    
    axs[0].set_title('Face>Object evoked-activity in face-selective filter')
    axs[1].set_title('Object>Face evoked-activity in object-selective filter')
    for ax in axs:
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('RMS amplitude (a.u.)')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlim(t_win)
    plt.tight_layout()
    
    # Save figure
    print("\nSaving figures...")
    fname_fig = op.join(ged_figure_root,
                        f"ged_filter_ts_{label_name}_diff.png")
    fig.savefig(fname_fig)
    plt.close(fig)
    print("\nCompleted!")
   
   
# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    ged_ga()