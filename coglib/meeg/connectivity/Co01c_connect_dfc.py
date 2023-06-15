# -*- coding: utf-8 -*-
"""
===================================
Co01. Connectivity DFC
===================================

Compute guassion-copula mutal information in source space using a MNE inverse solution

@author: Oscar Ferrante oscfer88@gmail.com
"""

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import argparse
import json
import statsmodels.api as sm
import xarray as xr
from scipy import stats

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse_epochs, 
                              # write_inverse_operator
                              )
import mne_bids

from frites.conn import conn_dfc, define_windows

from config import bids_root


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
                    default='dfc',
                    help='method used to measure connectivity (e.g. "coh")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
# parser.add_argument('--out_con',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward',
#                     help='Path to the connectivity (derivative) directory')
opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
con_method = opt.method
ged_label_list = ['fusifor']

task_rel = ["Irrelevant"]

surrogate = False
remove_evoked = True

debug = False

# Define vars for output folder name
if task_rel == ["Relevant non-target", "Irrelevant"]:
    t = ""
elif task_rel == ["Irrelevant"]:
    t = "_irr"
elif task_rel == ["Relevant non-target"]:
    t = "_rel"
if remove_evoked:
    e = "_no-evoked"
else:
    e = ""
if surrogate:
    s = "_surrogate"
else:
    s = ""


def connectivity_dfc(subject_id, visit_id):
    # Set path to preprocessing derivatives and create the related folders
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    ged_deriv_root = op.join(bids_root, "derivatives", "ged")
    
    con_deriv_root = op.join(bids_root, "derivatives", "connectivity"+t, "_dfc", e, s)
    if not op.exists(con_deriv_root):
        os.makedirs(con_deriv_root)
    con_figure_root =  op.join(con_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures",
                                con_method)
    if not op.exists(con_figure_root):
        os.makedirs(con_figure_root)
    
    print("Processing subject: %s" % subject_id)
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
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
    
    epochs = mne.read_epochs(bids_path_epo.fpath,
                             preload=False)
    
    # Pick trials
    epochs = epochs[f'Task_relevance in {task_rel} and Duration != "500ms"']
    if debug:
        epochs = epochs[0:100]
    
    # Select sensor
    epochs.load_data().pick('meg')  #try with EEG as well
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.0
    baseline = (b_tmin, b_tmax)
    epochs.apply_baseline(baseline=baseline)
    
    # Crop epochs time window to save memory
    e_tmin= -.5
    e_tmax= 2.
    epochs.crop(e_tmin, e_tmax)
    
    
    ## LABELS
    
    # Read labels from FS parc
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels_Datlas = mne.read_labels_from_annot(  #Destrieux's atlas
            "fsaverage", 
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)
        labels_Watlas = mne.read_labels_from_annot(  #Wang's atlas
            "fsaverage", 
            parc='wang2015_mplbl',
            subjects_dir=fs_deriv_root)
    else:
        labels_Datlas = mne.read_labels_from_annot(
            "sub-"+subject_id, 
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)
        labels_Watlas = mne.read_labels_from_annot(
            "sub-"+subject_id, 
            parc='wang2015_mplbl',
            subjects_dir=fs_deriv_root)
    
    # labels_Datlas_names = [l.name for l in labels_Datlas]
    # labels_Watlas_names = [l.name for l in labels_Watlas]
    
    # Read GNW and IIT ROI list
    f = open(op.join(rois_deriv_root,
                      'iit_gnw_rois.json'))
    gnw_iit_rois = json.load(f)
    
    # Create labels for selected ROIs
    labels = {}
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        for lab in gnw_iit_rois['surf_labels']['iit_wang']:
            lab = lab.replace('&','_and_')  # Fix the label name to match the template one
            print(lab)
            labels["iit_"+lab+"_lh"] = [l for l in labels_Watlas if l.name == lab+"-lh"]
            labels["iit_"+lab+"_rh"] = [l for l in labels_Watlas if l.name == lab+"-rh"]
        
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            lab = lab.replace('&','_and_')  # Fix the label name to match the template one
            print(lab)
            labels["gnw_"+lab+"_lh"] = [l for l in labels_Datlas if l.name == lab+"-lh"]
            labels["gnw_"+lab+"_rh"] = [l for l in labels_Datlas if l.name == lab+"-rh"]
    else:
        for lab in gnw_iit_rois['surf_labels']['iit_wang']:
            print(lab)
            labels["iit_"+lab+"_lh"] = [l for l in labels_Watlas if l.name == lab+"-lh"]
            labels["iit_"+lab+"_rh"] = [l for l in labels_Watlas if l.name == lab+"-rh"]
        
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            print(lab)
            labels["gnw_"+lab+"_lh"] = [l for l in labels_Datlas if l.name == lab+"-lh"][0]
            labels["gnw_"+lab+"_rh"] = [l for l in labels_Datlas if l.name == lab+"-rh"][0]
    
    # # Save labels
    # bids_path_con = bids_path_epo.copy().update(
    #     root=con_deriv_root,
    #     suffix="labels",
    #     extension='.pkl',
    #     check=False)
    
    # with open(bids_path_con.fpath, 'wb') as outp:
    #     pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)
    
    # Get V1/V2 labels and sum
    iit_v1v2_label = np.sum([labels["iit_V1d_lh"],
                             labels["iit_V1d_rh"],
                             labels["iit_V1v_lh"],
                             labels["iit_V1v_rh"],
                             labels["iit_V2d_lh"],
                             labels["iit_V2d_rh"],
                             labels["iit_V2v_lh"],
                             labels["iit_V2v_rh"]])
    
    
    ## Category-selective GED
    
    # Set params
    ged_label_name = ''.join(ged_label_list)
    
    # Create label
    ged_labels = []
    # Loop over labels
    for regexp in ged_label_list:
            
        # Create label for the given region
        if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
            lab = mne.read_labels_from_annot(
                "fsaverage", 
                parc='aparc',
                regexp=regexp,
                hemi='both',
                subjects_dir=fs_deriv_root)
        else:
            lab = mne.read_labels_from_annot(
                "sub-"+subject_id, 
                parc='aparc',
                regexp=regexp,
                hemi='both',
                subjects_dir=fs_deriv_root)
            
        # # Save label
        # bids_path_ged = mne_bids.BIDSPath(
        #     root=ged_deriv_root, 
        #     subject=subject_id,  
        #     datatype='meg',  
        #     task=None,
        #     session=visit_id, 
        #     suffix=f"desc-{regexp}_label-lh",
        #     extension='.label',
        #     check=False)
        # lab[0].save(bids_path_ged.fpath)
        
        # bids_path_ged = bids_path_ged.copy().update(
        #     suffix=f"desc-{regexp}_label-rh",)
        # lab[1].save(bids_path_ged.fpath)
        
        # Append to GED labels
        ged_labels.append(lab)
        
    # Combine GED labels
    ged_labels = np.sum(ged_labels)
    
    # Read GED filter
    bids_path_ged = bids_path_epo.copy().update(
            root=ged_deriv_root,
            suffix=f'desc-{ged_label_name},face_evecs',
            extension='.npy',
            check=False)
    ged_face_evecs = np.load(bids_path_ged.fpath)
    
    bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{ged_label_name},object_evecs')
    ged_object_evecs = np.load(bids_path_ged.fpath)
    
    
    ## GNW prefrontal GED
    
    # Merge all labels in prefrontal GNW ROI
    ged_gnw_label = np.sum([l for l_name, l in labels.items() if 'gnw' in l_name])
    
    # Read GNW prefrontal GED filter
    bids_path_ged = bids_path_epo.copy().update(
            root=ged_deriv_root,
            suffix='desc-gnw_evecs',
            extension='.npy',
            check=False)
    ged_gnw_evecs = np.load(bids_path_ged.fpath)
    
    
    ## SOURCE MODELLING
    
    # Compute rank
    rank = mne.compute_rank(epochs, 
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
    base_cov = mne.compute_covariance(epochs, 
                                   tmin=b_tmin, 
                                   tmax=b_tmax, 
                                   method='empirical', 
                                   rank=rank)
    
    active_cov = mne.compute_covariance(epochs, 
                                 tmin=0,
                                 tmax=None,
                                 method='empirical', 
                                 rank=rank)
    common_cov = base_cov + active_cov
    
    # Make inverse operator
    inverse_operator = make_inverse_operator(
        epochs.info,
        fwd, 
        common_cov,
        loose=.2,
        depth=.8,
        fixed=False,
        rank=rank,
        use_cps=True)
    
    # # Save inverse operator
    # bids_path_inv = bids_path_con.copy().update(
    #         suffix="inv_c%s" % condition,
    #         extension='.fif',
    #         check=False)
    # write_inverse_operator(bids_path_inv.fpath,
    #                         inverse_operator)
    
    
    ## CONNECTIVITY

    # Loop over conditions
    for condition in range(1,3):
    
        # Pick condition
        if condition == 1:
            epochs_cond = epochs['Category == "object"'].copy()
            cond_name = "object"
        elif condition == 2:
            epochs_cond = epochs['Category == "face"'].copy()
            cond_name = "face"
        else:
            raise ValueError("Condition %s does not exists" % condition)
        print("\n Running condition " + cond_name + "\n")
        
        # Compute inverse solution for each epoch
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        method = "dSPM"
        
        stcs = apply_inverse_epochs(epochs_cond, 
                                    inverse_operator, 
                                    lambda2, 
                                    method,
                                    pick_ori="normal", 
                                    return_generator=False)
        del epochs_cond
        
        # Average source estimates within each label to reduce signal cancellations
        src = inverse_operator['src']
        iit_label_ts = mne.extract_label_time_course(
            stcs, iit_v1v2_label, src, 
            mode='pca_flip', #was mean_flip
            return_generator=False)
        
        # Apply GED filter to source-level epochs
        ged_face_ts = []
        ged_object_ts = []
        ged_gnw_ts = []
        for i in range(len(stcs)):            
            # Get data
            data = stcs[i].in_label(ged_labels).data
            data_gnw = stcs[i].in_label(ged_gnw_label).data
            # Apply GED filter
            ged_face_ts.append(ged_face_evecs[:,0].T @ data)
            ged_object_ts.append(ged_object_evecs[:,0].T @ data)
            ged_gnw_ts.append(ged_gnw_evecs[:,0].T @ data_gnw)
        
        del stcs
        
        # Concatenate GNW & IIT labels and GED spatial filters
        all_ts = []
        for i in range(len(ged_face_ts)):
            all_ts.append(np.vstack([ged_gnw_ts[i], iit_label_ts[i], ged_face_ts[i],  ged_object_ts[i]]))
        ged_filter_labels = ['pfc','v1v2','face filter','object filter']
        
        # Create surrogate data by shuffling trial labels
        if surrogate:
            # Convert list to array
            all_ts_array = np.array(all_ts)
            
            # Loop over nodes
            for n in range(len(all_ts_array[0])):
                
                # Get trial number indices
                ind = np.arange(len(all_ts_array))
                
                # Shuffle trial indeces
                np.random.shuffle(ind)
                # plt.plot(ind)
                
                # Shuffle trials in the node data
                all_ts_array[:,n,:] = all_ts_array[ind,n,:]
                
            # Convert array back to list
            all_ts = [all_ts_array[i,:,:] for i in range(len(all_ts_array))]
        
        # Remove evoked using regression
        if remove_evoked:
            all_evoked = np.mean(all_ts, axis=0)
            for node in range(len(all_ts[0])):
                node_evoked = all_evoked[node,:]
                for trial in range(len(all_ts)):
                    all_ts[trial][node,:] = sm.OLS(np.array(all_ts)[trial,node,:], node_evoked).fit().resid
        
        
        # Compute Dynamic Functional Connectivity using the Gaussian-Copula Mutual Information (GCMI)
        
        # Insert data in an epochs object
        info = mne.create_info(ged_filter_labels, epochs.info['sfreq'], ch_types='grad')
        ep = mne.EpochsArray(all_ts, info) #tmin set to 0 for convinience (real tmin = -500)        
        
        # Create indices of label-to-label couples for which to compute connectivity
        n_labels = 2
        indices = (np.concatenate([range(0,n_labels),range(0,n_labels)]),
                   np.array([n_labels]*len(range(0,n_labels)) + [n_labels+1]*len(range(0,n_labels))))
        
        # Set params
        times = epochs.times
        trials = np.arange(len(all_ts))
        
        # Define the sliding windows
        window_len = 0.1 #100ms
        step = 0.02  #20ms
        sl_win = define_windows(times, 
                                slwin_len=window_len,
                                slwin_step=step)[0]
        
        # Compute tfr   
        tfr = mne.time_frequency.tfr_multitaper(
            ep,
            freqs=np.concatenate(
                (np.arange(2,30,1),
                 np.arange(30,101,2))),
            n_cycles=np.concatenate(
                (np.tile(4,len(np.arange(2,30,1))),
                 np.arange(30,101,2)/4)),
            use_fft=True,
            return_itc=False,
            average=False,
            time_bandwidth=2.,
            verbose=True)
        
        # Create empty array
        conndat = np.empty(
            (len(indices[0]),len(tfr.freqs),len(sl_win)))
        
        # Run DFC analysis
        for f, freq in enumerate( tfr.freqs ):
            for i_, ind_ in enumerate(zip( indices[0], indices[1])):
                # Convert data to xarray
                x = np.squeeze(tfr.data[:, [ind_[0], ind_[1]], f, :])
                rr = ['r0', 'r1']
                x = xr.DataArray(
                    x, 
                    dims=('trials', 'space', 'times'),
                    coords=(trials, rr, times))
        
                # Compute DFC on sliding windows
                dfc = conn_dfc(
                    x, 
                    times='times',
                    roi='space',
                    win_sample=sl_win)
        
                conndat[i_, f, :] = dfc.mean('trials').squeeze().data
        
        # Save results
        print('\nSaving...')
        bids_path_con = bids_path_epo.copy().update(
            root=con_deriv_root,
            suffix=f"desc-{con_method}_{cond_name}_con",
            extension='.npy',
            check=False)
                
        np.save(bids_path_con.fpath, conndat)
        
        # Save times and freqs info
        bids_path_con = bids_path_epo.copy().update(
            root=con_deriv_root,
            suffix=f"desc-{con_method}_times",
            extension='.npy',
            check=False)
                
        np.save(bids_path_con.fpath, dfc['times'].values)
        
        bids_path_con = bids_path_epo.copy().update(
            root=con_deriv_root,
            suffix=f"desc-{con_method}_freqs",
            extension='.npy',
            check=False)
                
        np.save(bids_path_con.fpath, tfr.freqs)
        
               
        # Plot
        analysis_time = [round(x,3) for x in dfc['times'].values]
        freqs = [int(x) for x in tfr.freqs]
        extent = list([analysis_time[0],analysis_time[-1],1,len(freqs)])
        
        indices_comb = [[i,j] for i,j in zip(indices[0], indices[1])]
        
        for i in indices_comb:
            # Get data and do z-scoring by frequencies
            data = stats.zscore(conndat[indices_comb.index(i),:,:], axis=1)
            
            # Plot
            fig, ax = plt.subplots(figsize=[8,6])
            im = ax.imshow(data,
                      cmap='RdYlBu_r',
                      extent=extent,
                      origin="lower", 
                      aspect='auto')
                    
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
            
            ax.set_yticklabels(freqs[0::5])
            ax.axhline(freqs.index(30), color='w', lw=4)

            plt.xlabel("Time (ms)", fontsize=14)
            plt.ylabel("Frequency (Hz)", fontsize=14)
            plt.title(f"Conn {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}: {cond_name}", fontsize=14, fontweight="bold")
            
            # Save figure
            fname_fig = op.join(con_figure_root,
                                f"conn_{con_method}_{cond_name}_{ged_filter_labels[i[0]]}-x-{ged_filter_labels[i[1]]}.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)


if __name__ == '__main__':
    # subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    # visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    connectivity_dfc(subject_id, visit_id)
