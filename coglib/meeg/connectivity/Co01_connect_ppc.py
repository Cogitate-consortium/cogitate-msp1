# -*- coding: utf-8 -*-
"""
===================================
Co01. Connectivity
===================================

Compute coherence in source space using a MNE inverse solution

@author: Oscar Ferrante oscfer88@gmail.com
"""

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import json
import statsmodels.api as sm

import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse_epochs, 
                              # write_inverse_operator
                              )
from mne_connectivity import spectral_connectivity_epochs  #spectral_connectivity
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA124',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--task_rel', 
                    type=str,
                    default='irr', 
                    help='specify the task condition ("irr", "rel" or "comb")')
parser.add_argument('--remove_evoked', 
                    type=str, 
                    default='false', 
                    help='Remove evoked response? (true or false)')
opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
con_method = 'ppc'
durs = ["1000ms", "1500ms"]
task_rel = opt.task_rel
remove_evoked = opt.remove_evoked.lower() == 'true'

surrogate = False
use_long_ged = False
surrogate = False

debug = False

# Define vars for output folder name
if task_rel == "comb":
    tasks = ["Relevant non-target", "Irrelevant"]
    t = ""
elif task_rel == "irr":
    tasks = ["Irrelevant"]
    t = "_irr"
elif task_rel == "rel":
    tasks = ["Relevant non-target"]
    t = "_rel"
else:
    raise ValueError(f"Error: tasks={tasks} not valid")
    
if len(durs) == 3:
    d = "_all-durs"
else:
    d = ""
    
if remove_evoked:
    e = "_no-evoked"
else:
    e = ""
    
if surrogate:
    s = "_surrogate"
else:
    s = ""
    
if use_long_ged:
    g = "_0.0-2.0"
    ged_label_list = ['fusiform']
else:
    g = ""
    ged_label_list = ['fusifor']


def connectivity(subject_id, visit_id):
    # Set path to preprocessing derivatives and create the related folders
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    ged_deriv_root = op.join(bids_root, "derivatives", "ged")
    
    con_deriv_root = op.join(bids_root, "derivatives", "connectivity"+t, d, g, e, s)
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
    epochs = epochs[f'Task_relevance in {tasks} and Duration in {durs}']
    if debug:
        epochs = epochs[0:100]
    
    # Select sensor
    epochs.load_data().pick('meg')  #try with EEG as well
    
    # Get sampling frequency
    sfreq = epochs.info['sfreq']
    
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
                parc='aparc',  #aparc  aparc.a2009s
                regexp=regexp, #'inferiortemporal'
                hemi='both',
                subjects_dir=fs_deriv_root)
        else:
            lab = mne.read_labels_from_annot(
                "sub-"+subject_id, 
                parc='aparc',  #aparc  aparc.a2009s
                regexp=regexp, #'inferiortemporal'
                hemi='both',
                subjects_dir=fs_deriv_root)
        
        # Append to GED labels
        ged_labels.append(lab)
        
    # Combine GED labels
    ged_labels = np.sum(ged_labels)
    
    # Read GED filter
    bids_path_ged = bids_path_epo.copy().update(
            root=op.join(ged_deriv_root,g),
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
    if task_rel == "comb":
        n_cond = 4
    else:
        n_cond = 2
    
    for condition in range(1,n_cond+1):
    
        # Pick condition
        if condition == 1:
            epochs_cond = epochs['Category == "object"'].copy()
            cond_name = "object"
        elif condition == 2:
            epochs_cond = epochs['Category == "face"'].copy()
            cond_name = "face"
        elif condition == 3:
            epochs_cond = epochs['Task_relevance == "Relevant non-target"'].copy()
            cond_name = "relev"
        elif condition == 4:
            epochs_cond = epochs['Task_relevance == "Irrelevant"'].copy()
            cond_name = "irrel"
        else:
            raise ValueError("Condition %s does not exists" % condition)
        print("\n\n\n### Running condition " + cond_name + " ###\n\n")
        
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
            mode='pca_flip', 
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
        
        # # Save GED time series
        # bids_path_ged = bids_path_ged.copy().update(
        #         root=op.join(ged_deriv_root,g),
        #         suffix=f'desc-{ged_label_name},face_ts',
        #         extension='.npy',
        #         check=False)
        # np.save(bids_path_ged.fpath, ged_face_ts)
        
        # bids_path_ged = bids_path_ged.copy().update(
        #         suffix=f'desc-{ged_label_name},object_ts')
        # np.save(bids_path_ged.fpath, ged_object_ts)
        
        # bids_path_ged = bids_path_ged.copy().update(
        #         suffix='desc-gnw_ts')
        # np.save(bids_path_ged.fpath, ged_gnw_ts)
        
        # Concatenate GNW & IIT labels and GED spatial filters
        all_ts = []
        for i in range(len(ged_face_ts)):
            all_ts.append(np.vstack([ged_gnw_ts[i], iit_label_ts[i], ged_face_ts[i],  ged_object_ts[i]]))
        ged_filter_labels = ['pfc','v1v2','face filter','object filter']
        
        # Create indices of label-to-label couples for which to compute connectivity
        n_labels = 2
        indices = (np.concatenate([range(0,n_labels),range(0,n_labels)]),
                   np.array([n_labels]*len(range(0,n_labels)) + [n_labels+1]*len(range(0,n_labels))))
        
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
        
        # Run connectivity separatelly for low and high frequencies
        for freq_range in ['low', 'high']:
            print('\nComputing connectivity in', freq_range, 'frequency range...')
            
            # Set connectivity params
            mode = 'cwt_morlet'
            if freq_range == 'low':
                fmin = 2.
                fmax = 30.
                fstep = 1.
                cwt_freqs = np.arange(fmin, fmax, fstep)
                cwt_n_cycles = 4
            elif freq_range == 'high':
                fmin = 30.
                fmax = 101.
                fstep = 2.
                cwt_freqs = np.arange(fmin, fmax, fstep)
                cwt_n_cycles = cwt_freqs / 4.
            
            # Run connectivity
            con = spectral_connectivity_epochs(
                all_ts,
                method=con_method, 
                indices=indices, 
                mode=mode, 
                sfreq=sfreq, 
                cwt_freqs=cwt_freqs, 
                cwt_n_cycles=cwt_n_cycles
                )
            
            # Save connectivity results
            bids_path_con = bids_path_epo.copy().update(
                root=con_deriv_root,
                suffix=f"desc-gnw-pfc-ged,{con_method},{freq_range},{cond_name}_con",
                extension='.nc',
                check=False)
            
            con.save(bids_path_con.fpath)
            
            # Plot results (con_data = time x label1 x label2 x freq)
            times = ['%.0f' %t for t in (np.array(con.times) - .5) * 1000]
            freqs = ['%.0f' %f for f in con.freqs]
            indices_comb = [[i,j] for i,j in zip(indices[0], indices[1])]
            
            for i in indices_comb:
                fig, ax = plt.subplots()
                sns.heatmap(con.get_data()[indices_comb.index(i),:,:],
                            xticklabels=250, yticklabels=5,
                            # vmin=0, vmax=.4,
                            cmap='RdYlBu_r',
                            ax=ax)
                ax.set_xticklabels(times[0::250],
                                    fontsize=8)
                ax.invert_yaxis()
                ax.set_yticklabels(freqs[0::5], rotation='horizontal',
                                    fontsize=8)
                
                # ax.set_xticklabels(np.rint((tmins[1:]-(twin/2))*1000).astype(int))
                plt.xlabel("time (ms)", fontsize=14)
                # ax.invert_yaxis()
                # ax.set_yticklabels(freqs, rotation='horizontal')
                plt.ylabel("Frequency (Hz)", fontsize=14)
                plt.title(f"Connect b/w {ged_filter_labels[i[0]]} & {ged_filter_labels[i[1]]}", fontsize=14, fontweight="bold")
                
                # Save figure
                fname_fig = op.join(con_figure_root,
                                    f"conn-gnw-pfc-ged_{con_method}_{freq_range}_{cond_name}_{ged_filter_labels[i[0]]}-x-{ged_filter_labels[i[1]]}.png")
                fig.savefig(fname_fig)
                plt.close(fig)

    
if __name__ == '__main__':
    # subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    # visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    connectivity(subject_id, visit_id)
