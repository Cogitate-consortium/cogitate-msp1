# -*- coding: utf-8 -*-
"""
===================================
Co02.  Grand-average connectivity
===================================

Compute the grand average for the connectivity analysis

@author: Oscar Ferrante oscfer88@gmail.com
"""

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import argparse
from scipy import stats as stats
import pickle

import mne
import mne_bids

from config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--method',
                    type=str,
                    default='dfc',
                    help='method used to measure connectivity (e.g. "coh")')
opt=parser.parse_args()


# Set params
visit_id = "V1"
con_method = opt.method

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


def connectivity_dfc_ga(sub_list, visit_id):
    # Set path to preprocessing derivatives and create the related folders
    con_deriv_root = op.join(bids_root, "derivatives", "connectivity"+t, "_dfc", e, s)
    if not op.exists(con_deriv_root):
        raise ValueError("Error: connectivity derivatives folder does not exist")
    con_figure_root =  op.join(con_deriv_root,
                                f"sub-groupphase{phase}",f"ses-{visit_id}","meg",
                                "figures",
                                con_method)
    if not op.exists(con_figure_root):
        os.makedirs(con_figure_root)
        
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    print('\nCompute connectivity grandaverage with method:', con_method)
    
    # Load times and freq
    bids_path_times = mne_bids.BIDSPath(
        root=con_deriv_root, 
        subject=sub_list[0],  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix=f"desc-{con_method}_times",
        extension='.npy',
        check=False)
    times = np.load(bids_path_times.fpath)
    
    bids_path_freqs = bids_path_times.copy().update(
        root=con_deriv_root,
        subject=sub_list[0],  
        suffix=f"desc-{con_method}_freqs")
    freqs = np.load(bids_path_freqs.fpath)
        
    # Create indices of label-to-label couples for which to compute connectivity
    n_labels = 2
    roi_labels = ['pfc','v1v2','face filter','object filter']
    indices = (np.concatenate([range(0,n_labels),range(0,n_labels)]),
               np.array([n_labels]*len(range(0,n_labels)) + [n_labels+1]*len(range(0,n_labels))))
    indices_comb = [[i,j] for i,j in zip(indices[0], indices[1])]
    
    con_condlist = {}
    # Loop over conditions
    for cond_name in ["object", "face"]:
        print("\n Running condition " + cond_name + "\n")
        
        # Load indivisual results
        con_all = []
        for sub in sub_list:
            print("subject id:", sub)
            
            # Load connectivity data
            bids_path_con = mne_bids.BIDSPath(
                root=con_deriv_root, 
                subject=sub,  
                datatype='meg',  
                task=bids_task,
                session=visit_id, 
                suffix=f"desc-{con_method}_{cond_name}_con",
                extension='.npy',
                check=False)
            
            con_all.append(np.load(bids_path_con.fpath))
            
        # Averaged over participants
        con_all = np.array(con_all)
        con_all_ga = np.mean(con_all, axis=0)
        
        # Save grandaverage
        bids_path_con = bids_path_con.update(
            subject=f"groupphase{phase}")
        np.save(bids_path_con.fpath, con_all_ga)
        
        # Append to list
        con_condlist[cond_name] = con_all
        
        # # Plot single condition
        analysis_time = [round(x,3) for x in times]
        freqs = [int(x) for x in freqs]
        extent = list([analysis_time[0],analysis_time[-1],1,len(freqs)])
        
        vmin = -5
        vmax = 5
        
        for i in indices_comb:
            # Get data and do z-scoring by frequencies
            data = stats.zscore(con_all_ga[indices_comb.index(i),:,:], axis=1)
            
            # Plot
            fig, ax = plt.subplots(figsize=[8,6])
            im = ax.imshow(data,
                      cmap='RdYlBu_r',
                      extent=extent,
                      origin="lower", 
                      aspect='auto',
                      vmin=vmin, vmax=vmax)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
            
            ax.set_yticklabels(freqs[0::5])
            ax.axhline(freqs.index(30), color='w', lw=4)
            
            plt.xlabel("Time (ms)", fontsize=14)
            plt.ylabel("Frequency (Hz)", fontsize=14)
            plt.title(f"{con_method} on {roi_labels[i[0]]} - {roi_labels[i[1]]}: {cond_name}", fontsize=14, fontweight="bold")
            
            # Save figure
            fname_fig = op.join(con_figure_root,
                                f"conn_{con_method}_{roi_labels[i[0]]}-x-{roi_labels[i[1]]}_{cond_name}.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)
    
    
    # Permutation analysis
        
    # Set test params
    pval = 0.05  # arbitrary
    n_observations = len(sub_list)
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    # Loop over indices
    t_obs_all = []
    clusters_all = []
    cluster_p_values_all = []
    p_values_all = []
    for i in indices_comb:
        print(f'\nTesting clursters for {roi_labels[i[0]]} - {roi_labels[i[1]]}')
        
        # Get data (subjects) × time × space
        Xfac = con_condlist['face'][:,indices_comb.index(i),:,:]
        Xobj = con_condlist['object'][:,indices_comb.index(i),:,:]
        
        # Run permutation analysis
        t_obs, clusters, cluster_p_values, H0 = \
            mne.stats.permutation_cluster_1samp_test(
                Xfac - Xobj, 
                threshold=thresh,
                out_type='mask')

        # Append results to list        
        t_obs_all.append(t_obs)
        clusters_all.append(clusters)
        cluster_p_values_all.append(cluster_p_values)
        p_values_all.append(cluster_p_values)
        
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_all = []
    for clusters, cluster_p_values in zip(clusters_all, cluster_p_values_all):
        good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        good_clusters_all.append(good_clusters)
    
    # Save significant clusters
    anal_contr = ['face', 'object']
    bids_path_con = bids_path_con.copy().update(
        subject=f"groupphase{phase}",
        suffix=f"desc-{con_method}_{anal_contr}_clusters",
        extension=".pkl",
        check=False)
    
    with open(bids_path_con.fpath, 'wb') as file:
        pickle.dump(good_clusters_all, file)
    
    
    # Plotting
    
    # Compute difference between face and object trials
    con_dif_data = np.mean(con_condlist['face'] - con_condlist['object'],
                           axis=0)
    vmin = -.2
    vmax = .2
    
    for i in indices_comb:
        # Get data
        data = con_dif_data[indices_comb.index(i),:,:]
        sig_mask = np.any(good_clusters_all[indices_comb.index(i)], axis=0)
        masked_data = np.ma.masked_where(sig_mask == 0, data)
        
        # Plot all data
        fig, ax = plt.subplots(figsize=[8,6])
        ax.imshow(data,
                  cmap='RdYlBu_r',
                  extent=extent,
                  origin="lower", 
                  alpha=.4, 
                  aspect='auto',
                  vmin=vmin, vmax=vmax)
        
        # Plot masked data
        im = ax.imshow(masked_data, 
                       cmap='RdYlBu_r', 
                       origin='lower',
                       extent=extent,
                       aspect='auto', 
                       vmin=vmin, vmax=vmax)
        
        # Draw contour
        if np.any(sig_mask == 1):
            ax.contour(sig_mask == 0, sig_mask == 0, 
                       colors="k", 
                       origin="lower",
                       extent=extent)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_yticklabels(freqs[0::5])
        ax.axhline(freqs.index(30), color='w', lw=4)
        
        plt.xlabel("time (ms)", fontsize=14)
        plt.ylabel("Frequency (Hz)", fontsize=14)
        plt.title(f"{con_method} on {roi_labels[i[0]]} - {roi_labels[i[1]]}: face vs object", fontsize=14, fontweight="bold")
        
        # Save figure
        fname_fig = op.join(con_figure_root,
                                f"conn-{con_method}_FvsO_{roi_labels[i[0]]}-x-{roi_labels[i[1]]}_FvsO.png")
        fig.savefig(fname_fig, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    # subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    # visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    connectivity_dfc_ga(sub_list, visit_id)
