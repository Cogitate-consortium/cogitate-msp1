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
import seaborn as sns
from scipy import stats as stats
import pickle

import mne
from mne_connectivity import read_connectivity, SpectroTemporalConnectivity
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--method',
                    type=str,
                    default='ppc',
                    help='method used to measure connectivity (e.g. "coh")')
opt=parser.parse_args()


# Set params
visit_id = "V1"
con_method = opt.method

task_rel = ["Irrelevant"]
remove_evoked = False

all_durs = False
use_long_ged = False
surrogate = False

debug = False

# Define vars for output folder name
if task_rel == ["Relevant non-target", "Irrelevant"]:
    t = ""
elif task_rel == ["Irrelevant"]:
    t = "_irr"
elif task_rel == ["Relevant non-target"]:
    t = "_rel"
if all_durs:
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
else:
    g = ""


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


def connectivity_ga(sub_list, visit_id):
    # Set path to preprocessing derivatives and create the related folders
    con_deriv_root = op.join(bids_root, "derivatives", "connectivity"+t, d, g, e, s)
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
    
    # Create indices of connections for which connectivity was computed
    n_labels = 2
    indices = (np.concatenate([range(0,n_labels),range(0,n_labels)]),
               np.array([n_labels]*len(range(0,n_labels)) + [n_labels+1]*len(range(0,n_labels))))
    
    # Loop over frequencies
    for freq_range in ['low', 'high']:
        print(f'\nFreq range: {freq_range}')
        
        # Loop over analysis (i.e., contrasts)
        con_dif = {}
        if task_rel == ["Relevant non-target", "Irrelevant"]:
            cond_contr = [['face', 'object'],['relev', 'irrel']]
        else:
            cond_contr = [['face', 'object']]
        for anal_contr in cond_contr:
            print(f"\nAnalysis: {anal_contr[0]} vs {anal_contr[1]}")
            
            # Loop over conditions
            con_condlist = {}
            for cond_name in anal_contr:
                print(f"\nCondition: {cond_name}")
                
                # Load connectivity results
                con_all = []
                for sub in sub_list:
                    print(f"subject id: {sub}")
                    
                    # Set path
                    bids_path_con = mne_bids.BIDSPath(
                        root=con_deriv_root, 
                        subject=sub,  
                        datatype='meg',  
                        task=bids_task,
                        session=visit_id, 
                        suffix=f"desc-gnw-pfc-ged,{con_method},{freq_range},{cond_name}_con",
                        extension='.nc',
                        check=False)
                    
                    # Load data
                    con_all.append(
                        read_connectivity(bids_path_con.fpath))
                
                # Get data
                con_all_data = []
                for con in con_all:
                    con_all_data.append(con.get_data())
                    times = ['%.0f' %t for t in (np.array(con.times) - .5) * 1000]
                    freqs = ['%.0f' %f for f in con.freqs]
                
                con_all_data = np.asarray(con_all_data)  #convert to array
                
                # Append individual con data to full data list
                con_condlist[cond_name] = np.asarray(con_all_data)
                
                # Average data across participants and put them in a connectivity object
                con_ga = SpectroTemporalConnectivity(
                    data = np.mean(con_all_data, axis=0), 
                    freqs = con.freqs, 
                    times = con.times, 
                    n_nodes = con.n_nodes,
                    indices = indices)
                
                # Save grandaverage data
                bids_path_con = bids_path_con.copy().update(
                    subject=f"groupphase{phase}",
                    check=False)
                
                con_ga.save(bids_path_con.fpath)
                
                
                ## Plotting
                
                # Set plotting params
                ged_filter_labels = ['pfc','v1v2','face filter','object filter']
                indices_comb = [[i,j] for i,j in zip(indices[0], indices[1])]
                vmin = 0.
                vmax = .15
                
                # Plot individual ROI results
                for i in indices_comb:
                    print(f'\nPlotting {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}...')
                    fig, ax = plt.subplots(figsize=[8,6])
                    sns.heatmap(con_ga.get_data()[indices_comb.index(i),:,:],
                                xticklabels=250, yticklabels=5,
                                vmin=vmin, 
                                vmax=vmax,
                                cmap='RdYlBu_r',
                                ax=ax)
                    ax.set_xticklabels(times[0::250],
                                        fontsize=8)
                    ax.invert_yaxis()
                    ax.set_yticklabels(freqs[0::5], rotation='horizontal',
                                        fontsize=8)
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=8)
                    
                    plt.xlabel("time (ms)", fontsize=14)
                    plt.ylabel("Frequency (Hz)", fontsize=14)
                    plt.title(f"{con_method} on {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}: {cond_name}", fontsize=14, fontweight="bold")
                    
                    # Save figure
                    fname_fig = op.join(con_figure_root,
                                        f"conn-{con_method}_{freq_range}_{cond_name}_{ged_filter_labels[i[0]]}-x-{ged_filter_labels[i[1]]}.png")
                    fig.savefig(fname_fig, dpi=300)
                    plt.close(fig)
            
            
            ## Face vs. object / relev. vs. irrel.: Permutation analysis
                
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
                print(f'\nTesting clursters for {ged_filter_labels[i[0]]} - {ged_filter_labels[i[1]]}')
                
                # Get data (subjects) × time × space
                Xfac = con_condlist[f'{anal_contr[0]}'][:,indices_comb.index(i),:,:]
                Xobj = con_condlist[f'{anal_contr[1]}'][:,indices_comb.index(i),:,:]
                
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
                
                # Compute BF on the regions of interest
                Xfac_gnw_data = Xfac[:, np.isin(freqs, np.arange(60, 91)), :][:, :, np.isin(times, np.arange(300, 501))]
                Xfac_gnw_data = np.mean(Xfac_gnw_data, axis=(1,2))
                
                Xobj_gnw_data = Xobj[:, np.isin(freqs, np.arange(60, 91)), :][:, :, np.isin(times, np.arange(300, 501))]
                Xobj_gnw_data = np.mean(Xobj_gnw_data, axis=(1,2))
                
                Xfac_iit_data = Xfac[:, np.isin(freqs, np.arange(60, 91)), :][:, :, np.isin(times, np.arange(100, 1001))]
                Xfac_iit_data = np.mean(Xfac_iit_data, axis=(1,2))
                
                Xobj_iit_data = Xobj[:, np.isin(freqs, np.arange(60, 91)), :][:, :, np.isin(times, np.arange(100, 1001))]
                Xobj_iit_data = np.mean(Xobj_iit_data, axis=(1,2))
                
                # Compute BF
                bf_gnw = bayes_ttest(Xfac_gnw_data, y=Xobj_gnw_data, paired=True)
                print(f"GNW Bayesian T-test: {bf_gnw}")
                bf_iit = bayes_ttest(Xfac_iit_data, y=Xobj_iit_data, paired=True)
                print(f"IIT Bayesian T-test: {bf_iit}")
                
            # Select the clusters that are statistically significant at p < 0.05
            good_clusters_all = []
            for clusters, cluster_p_values in zip(clusters_all, cluster_p_values_all):
                good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
                good_clusters = [clusters[idx] for idx in good_clusters_idx]
                good_clusters_all.append(good_clusters)
            
            # Save significant clusters
            bids_path_con = bids_path_con.copy().update(
                subject=f"groupphase{phase}",
                suffix=f"desc-gnw-pfc-ged,{con_method},{freq_range},{anal_contr}_clusters",
                extension=".pkl",
                check=False)
            
            with open(bids_path_con.fpath, 'wb') as file:
                pickle.dump(good_clusters_all, file)
            
            
            ## Face vs. object / relev. vs. irrel.: Plotting
            
            # Compute difference between face and object trials
            con_dif[f"{anal_contr}"] = con_condlist[f'{anal_contr[0]}'] - con_condlist[f'{anal_contr[1]}']
            
            con_dif_data = np.mean(con_dif[f"{anal_contr}"], axis=0)
            
            vmin = -.075
            vmax = .075
            
            # Plot
            for i in indices_comb:
                print(f'\nPlotting {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}...')
                # Get data
                data = con_dif_data[indices_comb.index(i),:,:]
                # extent = [0,len(times),0,len(freqs)]
                extent = list(map(int, [times[0],times[-1],freqs[0],freqs[-1]]))
                sig_mask = np.any(good_clusters_all[indices_comb.index(i)], axis=0)
                masked_data = np.ma.masked_where(sig_mask == 0, data)
                
                # Open figure
                fig, ax = plt.subplots(figsize=[8,6])
                
                # Plot all data
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
                    ax.contour(sig_mask, 
                               levels=[0, 1], 
                               colors="k", 
                               origin="lower",
                               extent=extent)
                
                ax.set_yticklabels(freqs[0::5])
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=8)
                
                plt.xlabel("time (ms)", fontsize=14)
                plt.ylabel("Frequency (Hz)", fontsize=14)
                plt.title(f"{con_method} on {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}: {anal_contr[0]} vs {anal_contr[1]}", fontsize=14, fontweight="bold")
                
                # Save figure
                fname_fig = op.join(con_figure_root,
                                        f"conn-{con_method}_{freq_range}_{anal_contr[0][0]}vs{anal_contr[1][0]}_{ged_filter_labels[i[0]]}-x-{ged_filter_labels[i[1]]}.png")
                fig.savefig(fname_fig, dpi=300)
                plt.close(fig)
            
            
            # # Compute the difference of the difference (face trials vs object trials / face filter vs. object filter)
            # con_dif_c1flt_data = con_condlist[f'{anal_contr[0]}'][:,:len(con_dif_data)//2,:,:] \
            #                               - con_condlist[f'{anal_contr[1]}'][:,:len(con_dif_data)//2,:,:]
            # con_dif_c2flt_data = con_condlist[f'{anal_contr[0]}'][:,len(con_dif_data)//2:,:,:] \
            #                               - con_condlist[f'{anal_contr[1]}'][:,len(con_dif_data)//2:,:,:]
            # con_dif_dif_data = np.mean(con_dif_c1flt_data - con_dif_c2flt_data,
            #                            axis=0)
            
            # vmin = -.075
            # vmax = .075
            
            # # Plot
            # for i in range(len(con_dif_dif_data)):
            #     fig, ax = plt.subplots(figsize=[8,6])
            #     sns.heatmap(con_dif_dif_data[i,:,:],
            #                 xticklabels=250, yticklabels=5,
            #                 vmin=vmin, 
            #                 vmax=vmax,
            #                 cmap='RdYlBu_r',
            #                 ax=ax)
            #     ax.set_xticklabels(times[0::250],
            #                         fontsize=8)
            #     ax.invert_yaxis()
            #     ax.set_yticklabels(freqs[0::5], rotation='horizontal',
            #                         fontsize=8)
            #     cbar = ax.collections[0].colorbar
            #     cbar.ax.tick_params(labelsize=8)
                
            #     plt.xlabel("time (ms)", fontsize=14)
            #     plt.ylabel("Frequency (Hz)", fontsize=14)
            #     plt.title(f"{con_method} {anal_contr[0]}-vs-{anal_contr[1]} on {ged_filter_labels[i]}: {anal_contr[0]} vs {anal_contr[1]} filter", fontsize=14, fontweight="bold")
                
            #     # Save figure
            #     fname_fig = op.join(con_figure_root,
            #                         f"conn-{con_method}_{freq_range}_{anal_contr[0][0]}vs{anal_contr[1][0]}DiffDiff_{ged_filter_labels[i]}.png")
            #     fig.savefig(fname_fig, dpi=300)
            #     plt.close(fig)
        
        
        ## Stimulus vs. task: Permutation analysis
        
        if task_rel == ["Relevant non-target", "Irrelevant"]:
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
                print(f'\nTesting clursters for {ged_filter_labels[i[0]]} - {ged_filter_labels[i[1]]}')
                
                # Get data (subjects) × time × space
                Xsti = con_dif["['face', 'object']"][:,indices_comb.index(i),:,:]
                Xtas = con_dif["['relev', 'irrel']"][:,indices_comb.index(i),:,:]
                
                # Run permutation analysis
                t_obs, clusters, cluster_p_values, H0 = \
                    mne.stats.permutation_cluster_1samp_test(
                        Xsti - Xtas, 
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
            bids_path_con = bids_path_con.copy().update(
                subject=f"groupphase{phase}",
                suffix=f"desc-gnw-pfc-ged,{con_method},{freq_range},stim_vs_relev_clusters",
                extension=".pkl",
                check=False)
            
            with open(bids_path_con.fpath, 'wb') as file:
                pickle.dump(good_clusters_all, file)
            
            
            ## Plotting
                
            # Compute difference between stimulus and task effects
            con_dif_dif_data = np.mean(
                con_dif["['face', 'object']"] - con_dif["['relev', 'irrel']"],
                axis=0)
            
            vmin = -.075
            vmax = .075
            
            # Plot
            for i in indices_comb:
                print(f'\nPlotting {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}...')
                # Get data
                data = con_dif_dif_data[indices_comb.index(i),:,:]
                # extent = [0,len(times),0,len(freqs)]
                extent = list(map(int, [times[0],times[-1],freqs[0],freqs[-1]]))
                sig_mask = np.any(good_clusters_all[indices_comb.index(i)], axis=0)
                masked_data = np.ma.masked_where(sig_mask == 0, data)
                
                # Open figure
                fig, ax = plt.subplots(figsize=[8,6])
                
                # Plot all data
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
                    ax.contour(sig_mask, 
                               levels=[0, 1], 
                               colors="k", 
                               origin="lower",
                               extent=extent)
                
                ax.set_yticklabels(freqs[0::5])
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=8)
                
                plt.xlabel("time (ms)", fontsize=14)
                plt.ylabel("Frequency (Hz)", fontsize=14)
                plt.title(f"{con_method} on {ged_filter_labels[i[0]]}-{ged_filter_labels[i[1]]}: stimuus vs task", fontsize=14, fontweight="bold")
                
                # Save figure
                fname_fig = op.join(con_figure_root,
                                        f"conn-{con_method}_{freq_range}_svst_{ged_filter_labels[i[0]]}-x-{ged_filter_labels[i[1]]}.png")
                fig.savefig(fname_fig, dpi=300)
                plt.close(fig)

if __name__ == '__main__':
    # subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    # visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    connectivity_ga(sub_list, visit_id)
