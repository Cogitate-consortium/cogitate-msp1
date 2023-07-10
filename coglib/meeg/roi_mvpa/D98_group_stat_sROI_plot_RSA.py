"""
====================
D08. Group analysis for RSA
====================

@author: Ling Liu  ling.liu@pku.edu.cn

"""

import os.path as op
import os
import argparse

import pickle
import mne

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='ticks')


from scipy import stats as stats


import matplotlib as mpl

from matplotlib.patches import Rectangle

from rsa_helper_functions_meg import subsample_matrices,compute_correlation_theories

import ptitprince as pt

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root,plot_param
from sublist import sub_list


parser = argparse.ArgumentParser()
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--cT', type=str, nargs='*', default=['500ms', '1000ms', '1500ms'],
                    help='condition in Time duration')

parser.add_argument('--cC', type=str, nargs='*', default=['FO'],
                    help='selected decoding category, FO for face and object, LF for letter and false')
parser.add_argument('--cD',type=str,nargs='*', default=['Irrelevant', 'Relevant non-target'],
                    help='selected decoding Task, Relevant non Target or Irrelevant condition')
parser.add_argument('--space',
                    type=str,
                    default='surface',
                    help='source space ("surface" or "volume")')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--methods',
                    type=str,
                    default='roi_mvpa',
                    help='decoding methods name, for the data folder')
parser.add_argument('--analysis',
                    type=str,
                    default='RSA_ID',
                    help='the name for anlaysis, e.g. Tall for 3 durations combined analysis')


opt = parser.parse_args()

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path
methods_name=opt.methods
analysis_name=opt.analysis


opt = parser.parse_args()
con_C = opt.cC
con_D = opt.cD
con_T = opt.cT

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path

#1) Select Category
if con_C[0] == 'FO':
    conditions_C = ['face', 'object']
    print(conditions_C)
elif con_C[0] == 'LF':
    conditions_C = ['letter', 'false']
    print(conditions_C)
elif con_C[0] == 'F':
    conditions_C = ['face']
    print(conditions_C)
elif con_C[0] == 'O':
    conditions_C = ['object']
    print(conditions_C)
elif con_C[0] == 'L':
    conditions_C = ['letter']
    print(conditions_C)
elif con_C[0] == 'FA':
    conditions_C = ['false']
    print(conditions_C)
    
#1) Select time duration
if con_T[0] == 'T_all':
    con_T = ['500ms', '1000ms','1500ms']
    print(con_T)
elif con_T[0] == 'ML':# middle and long
    con_T = ['1000ms','1500ms']
    print(con_T)

# get the parameters dictionary
def mm2inch(val):
    return val / 25.4

param = plot_param
colors=param['colors']
fig_size = param["figure_size_mm"]
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('font', size=param["font_size"])  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"])  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"])  # legend fontsize
plt.rc('figure', titlesize=param["font_size"])  # fontsize of the fi
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)



# set the path for decoding analysis
def set_path_plot(bids_root, visit_id, analysis_name,con_name):

    ### I  Set the group Data Path
    # Set path to decoding derivatives
    decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa')

    data_path=op.join(decoding_path, analysis_name)

    # Set path to group analysis derivatives
    group_deriv_root = op.join(data_path, "group")
    if not op.exists(group_deriv_root):
        os.makedirs(group_deriv_root)
        
    
    # Set path to the ROI MVPA output(1) stat_data, 2) figures, 3) codes)
   
    # 1) output_stat_data
    stat_data_root = op.join(group_deriv_root,"stat_data",con_name)
    if not op.exists(stat_data_root):
        os.makedirs(stat_data_root)

    # 2) output_figure
    stat_figure_root = op.join(group_deriv_root,"stat_figures",con_name)
    if not op.exists(stat_figure_root):
        os.makedirs(stat_figure_root)

    return group_deriv_root,stat_data_root,stat_figure_root    


def rsa2gat(dat,roi_name,cond_name,decoding_name,analysis):
    if analysis=='RSA_ID':
        time_points=201
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
    if analysis=='RSA_Ori':
        time_points=251
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
    elif analysis=='RSA_Cat':
        time_points=251
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name][decoding_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
        
    return roi_rsa_g   

def rsa_plot(roi_rsa_data,C1_stat,time_points,fname_fig):
     
    #fig, ax = plt.subplots(1)
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[0])])
    
    roi_rsa_mean=np.mean(roi_rsa_data,0)
    RDM_avg_plot = np.nan * np.ones_like(roi_rsa_mean)
    for c, p_val in zip(C1_stat['cluster'], C1_stat['cluster_p']):
        if p_val <= 0.05:
            RDM_avg_plot[c] = roi_rsa_mean[c]
            
    cmap = mpl.cm.RdYlBu_r
    im=ax.imshow(roi_rsa_mean, interpolation='lanczos', origin='lower', cmap=cmap, alpha=0.9,aspect='equal',
                       extent=time_points[[0, -1, 0, -1]],vmin=-0.3, vmax=0.3)
    ax.contour(RDM_avg_plot > 0, RDM_avg_plot > 0, colors="grey", linewidths=2, origin="lower",extent=time_points[[0, -1, 0, -1]])
    im = ax.imshow(RDM_avg_plot, origin='lower', cmap=cmap,aspect='equal',
                   extent=time_points[[0, -1, 0, -1]], vmin=-0.3, vmax=0.3)
    
    # Define the size and position of the squares
    square_size = 0.2
    x=[0.3,0.8,1.3,1.8]
    y=[0.3,0.8,1.3,1.8]
    squares=[]
    for ii in range(16):
        for nn in range(4):
            for mm in range(4):
                squares.append((x[nn],y[mm])) 


    # Draw the squares
    for square in squares:
        
        rect = Rectangle(square, square_size, square_size, linewidth=3,edgecolor=[0, 0, 0], facecolor='none', linestyle=":")
        ax.add_patch(rect)

    ax.axhline(0,color='k')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    #ax.set_title(f'RSA_ {roi_name}')
    ax.set(xlabel='Time (s)', ylabel='Time (s)')
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    #plt.colorbar(im, ax=ax,fraction=0.03, pad=0.05)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    #cb.ax.set_ylabel(cbar_label)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalizat
    mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    #mne.stats.permutation_cluster_1samp_test

def rsa_ID_plot(roi_rsa_data,C1_stat,time_points,fname_fig):
     
    #fig, ax = plt.subplots(1)
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[0])])
    
    roi_rsa_mean=np.mean(roi_rsa_data,0)
    RDM_avg_plot = np.nan * np.ones_like(roi_rsa_mean)
    for c, p_val in zip(C1_stat['cluster'], C1_stat['cluster_p']):
        if p_val <= 0.05:
            RDM_avg_plot[c] = roi_rsa_mean[c]
            
    cmap = mpl.cm.RdYlBu_r
    im=ax.imshow(roi_rsa_mean, interpolation='lanczos', origin='lower', cmap=cmap, alpha=0.9,aspect='equal',
                       extent=time_points[[0, -1, 0, -1]],vmin=-0.3, vmax=0.3)
    ax.contour(RDM_avg_plot > 0, RDM_avg_plot > 0, colors="grey", linewidths=2, origin="lower",extent=time_points[[0, -1, 0, -1]])
    im = ax.imshow(RDM_avg_plot, origin='lower', cmap=cmap,aspect='equal',
                   extent=time_points[[0, -1, 0, -1]], vmin=-0.3, vmax=0.3)
    
    # Define the size and position of the squares
    square_size = 0.2
    x=[0.3,0.8,1.3]
    y=[0.3,0.8,1.3]
    squares=[]
    for ii in range(9):
        for nn in range(3):
            for mm in range(3):
                squares.append((x[nn],y[mm])) 


    # Draw the squares
    for square in squares:
        
        rect = Rectangle(square, square_size, square_size, linewidth=3,edgecolor=[0, 0, 0], facecolor='none', linestyle=":")
        ax.add_patch(rect)

    ax.axhline(0,color='k')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    #ax.set_title(f'RSA_ {roi_name}')
    ax.set(xlabel='Time (s)', ylabel='Time (s)')
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    #plt.colorbar(im, ax=ax,fraction=0.03, pad=0.05)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    #cb.ax.set_ylabel(cbar_label)
    cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalizat
    mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    #mne.stats.permutation_cluster_1samp_test

def rsa_subsample_plot(roi_rsa_mean, subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict,vmin,vmax,cmap,fname_fig):
     
    fig, ax = plt.subplots(1)
    
    
    
    cmap = mpl.cm.RdYlBu_r
    im=ax.imshow(roi_rsa_mean, interpolation='lanczos', origin='lower', cmap=cmap,
                       aspect='equal',vmin=vmin, vmax=vmax)
    ax.axhline(0,color='k')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    #ax.set_title(f'RSA_ {roi_name}')
    ax.set(xlabel='Time (s)', ylabel='Time (s)')
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    plt.colorbar(im, ax=ax,fraction=0.03, pad=0.05) 
    
    # Adding the matrices demarcations in case of subsampling:
    [ax.axhline(ind + 0.5, color='k', linestyle='--')
    for ind in matrices_delimitations_ref]
    [ax.axvline(ind + 0.5, color='k', linestyle='--')
    for ind in matrices_delimitations_ref]
    # Adding axis break to mark the difference:
    d = 0.01
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # Looping through each demarcations to mark them::
    for ind in matrices_delimitations_ref:
        ind_trans = (ind + 1) / len(roi_rsa_mean)
        ax.plot((ind_trans - 0.005 - d, ind_trans
                - 0.005 + d), (-d, +d), **kwargs)
        ax.plot((ind_trans + 0.005 - d, ind_trans
                + 0.005 + d), (-d, +d), **kwargs)
        ax.plot((-d, +d), (ind_trans - 0.005 - d,
                          ind_trans - 0.005 + d), **kwargs)
        ax.plot((-d, +d), (ind_trans + 0.005 - d,
                          ind_trans + 0.005 + d), **kwargs)
    # Generate the ticks:
    ticks_pos = np.linspace(0, roi_rsa_mean.shape[0] - 1, 8)
    # Generate the tick position and labels:
    ticks_labels = [str(subsampled_time_ref[int(ind)]) for ind in ticks_pos]
    ax.set_xticks(ticks_pos)
    ax.set_yticks(ticks_pos)
    ax.set_xticklabels(ticks_labels)
    ax.set_yticklabels(ticks_labels)
    plt.tight_layout()
    
    fig.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
# def sign_test(data):
#     seed=1999
#     random_state = check_random_state(seed)
#     p=np.mean(data * random_state.choice([1, -1], len(data)))
#     return p

def theory_rdm(RSA_methods):
    if RSA_methods=='RSA_ID':
        GNW_rdm=np.zeros([63,63])
        GNW_rdm[0:21,0:21]=1
        GNW_rdm[0:21,42:63]=1
        GNW_rdm[42:63,0:21]=1
        GNW_rdm[42:63,42:63]=1    
        
        IIT_rdm=np.zeros([63,63])
        IIT_rdm[0:42,0:42]=1
        
        theory_rdm=dict()
        theory_rdm['IIT']=IIT_rdm
        theory_rdm['GNW']=GNW_rdm
    elif RSA_methods=='RSA_Cat'or'RSA_Ori':
        GNW_rdm=np.zeros([84,84])
        GNW_rdm[0:21,0:21]=1
        GNW_rdm[0:21,63:84]=1
        GNW_rdm[63:84,0:21]=1
        GNW_rdm[63:84,63:84]=1    
        
        IIT_rdm=np.zeros([84,84])
        IIT_rdm[0:63,0:63]=1
        
        theory_rdm=dict()
        theory_rdm['IIT']=IIT_rdm
        theory_rdm['GNW']=GNW_rdm
    
    return theory_rdm

def corr_theory(rsa_subsample,analysis_name,decoding_name):
    
    #1:generated theory_rdm
    theory_rdm_matrix=theory_rdm(analysis_name)
    
    #2:correlate the theories matrices with the observed matrices for each subjects
    for n in range(len(sub_list)):
        observed_matrix=rsa_subsample[n,:,:]
        if n==0:
            correlation_results, correlation_results_corrected=compute_correlation_theories([observed_matrix], theory_rdm_matrix, method="kendall")
            group_corr_corrected=correlation_results_corrected
            group_corr=correlation_results
        else:
            correlation_results, correlation_results_corrected=compute_correlation_theories([observed_matrix], theory_rdm_matrix, method="kendall")
            group_corr_corrected=group_corr_corrected.append(correlation_results_corrected,ignore_index=True)
            group_corr=group_corr.append(correlation_results,ignore_index=True)
       
    #stat
    
    p_value=dict()
    stat,p_value['IIT']=stats.wilcoxon(group_corr['IIT'])
    stat,p_value['GNW']=stats.wilcoxon(group_corr['GNW'])
    stat,p_value['diff']=stats.mannwhitneyu(group_corr_corrected['GNW'],group_corr_corrected['IIT'])
    
    fname_p_value=op.join(stat_data_root, task_info +"_" + analysis_name + roi_name + decoding_name +'_stat_value.npz')
    np.savez(fname_p_value,p_value,group_corr,group_corr_corrected)
    
    
    corr_palette=[colors['IIT'],colors['GNW']]
    #plot
    group_corr_plot=group_corr.melt(var_name='theory',value_name='corr')
    
    fig, ax = plt.subplots(1)
    ax=pt.RainCloud(x='theory',y='corr',data=group_corr_plot,palette=corr_palette,bw=.2,width_viol=.5,ax=ax,orient='v')
    plt.title(analysis_name+'_corr_'+roi_name)
    fname_corr_fig=op.join(stat_figure_root, task_info +"_" + analysis_name + roi_name+'_'+analysis_name + decoding_name +'_corr.svg')
    fig.savefig(fname_corr_fig,format="svg", transparent=True, dpi=300)

def RSA_ID_plot(roi_name):
    analysis_name='RSA_ID'
    time_point = np.array(range(-500, 1501, 10))/1000
    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=['rsa'],decoding_name='ID',analysis=analysis_name)
    C1_stat=stat_cluster_1sample_RDM(roi_rsa_g,test_win_on=0, test_win_off=201,chance_index=0)
    
    fname_fig=op.join(stat_figure_root, task_info +"_" + analysis_name +'_' + roi_name+'_.svg')
    #plot
    rsa_ID_plot(roi_rsa_g,C1_stat=C1_stat,time_points=time_point,fname_fig=fname_fig)
    
    #subsample data
    intervals_of_interest={"x": [[0.3,0.5],[0.8,1.0],[1.3,1.5]],"y":[[0.3,0.5],[0.8,1.0],[1.3,1.5]]}
    
    rsa_subsample=np.zeros([len(sub_list),63,63])
    
    for n in range(len(sub_list)):
        rsa_subsample[n,:,:], subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict=subsample_matrices(roi_rsa_g[n,:,:], -0.5, 1.5, intervals_of_interest)
    
    
    roi_rsa_mean=np.mean(rsa_subsample,0)
    fname_fig_sub=op.join(stat_figure_root, task_info +"_" + analysis_name + roi_name +'_subsample.svg')
    cmap = mpl.cm.RdYlBu_r
    vmin=-0.02
    vmax=0.1
    #plot
    rsa_subsample_plot(roi_rsa_mean, subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict,vmin,vmax,cmap,fname_fig_sub)
    
    #correlated with theory rdm
    corr_theory(rsa_subsample,analysis_name,decoding_name='ID')


def RSA_Cat_plot(roi_name,condition):
    analysis_name='RSA_Cat'

    time_point = np.array(range(-500,2001, 10))/1000
    if condition=='Irrelevant':
        conD='IR'
    elif condition=='Relevant non-target':
        conD='RE'
    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=['rsa'],decoding_name=condition,analysis=analysis_name)
    
    
    C1_stat=stat_cluster_1sample_RDM(roi_rsa_g,test_win_on=0, test_win_off=251,chance_index=0)
    
    fname_fig=op.join(stat_figure_root, task_info +"_" + analysis_name +'_' + roi_name + '_' + conD +'.svg')
    #plot
    rsa_plot(roi_rsa_g,C1_stat=C1_stat,time_points=time_point,fname_fig=fname_fig)
    
    #subsample data
    intervals_of_interest={"x": [[0.3,0.5],[0.8,1.0],[1.3,1.5],[1.8,2.0]],"y":[[0.3,0.5],[0.8,1.0],[1.3,1.5],[1.8,2.0]]}
    
    rsa_subsample=np.zeros([len(sub_list),84,84])
    
    for n in range(len(sub_list)):
        rsa_subsample[n,:,:], subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict=subsample_matrices(roi_rsa_g[n,:,:], -0.5, 2, intervals_of_interest)
    
    roi_rsa_mean=np.mean(rsa_subsample,0)
    fname_fig_sub=op.join(stat_figure_root, task_info +"_" + analysis_name + roi_name + '_' + conD + '_subsample.svg')
    cmap = mpl.cm.RdYlBu_r
    vmin=-0.02
    vmax=0.1
    #plot
    rsa_subsample_plot(roi_rsa_mean, subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict,vmin,vmax,cmap,fname_fig_sub)
    
    #correlated with theory rdm
    corr_theory(rsa_subsample,analysis_name,decoding_name=condition)

def RSA_Ori_plot(roi_name):
    analysis='RSA_Ori'
    time_point = np.array(range(-500,2001, 10))/1000
    
    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=['rsa'],decoding_name='Ori',analysis=analysis)
    
    C1_stat=stat_cluster_1sample_RDM(roi_rsa_g,test_win_on=0, test_win_off=251,chance_index=0)
    
    fname_fig=op.join(stat_figure_root, task_info +"_" + analysis_name +'_' + roi_name +'.svg')
    #plot
    rsa_plot(roi_rsa_g,C1_stat=C1_stat,time_points=time_point,fname_fig=fname_fig)
    
    #subsample data
    intervals_of_interest={"x": [[0.3,0.5],[0.8,1.0],[1.3,1.5],[1.8,2.0]],"y":[[0.3,0.5],[0.8,1.0],[1.3,1.5],[1.8,2.0]]}
    
    rsa_subsample=np.zeros([len(sub_list),84,84])
    
    for n in range(len(sub_list)):
        rsa_subsample[n,:,:], subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict=subsample_matrices(roi_rsa_g[n,:,:], -0.5, 2, intervals_of_interest)
    
    roi_rsa_mean=np.mean(rsa_subsample,0)
    
    
    
    fname_fig_sub=op.join(stat_figure_root, task_info +"_" + analysis_name + roi_name +'_subsample.svg')
    cmap = mpl.cm.RdYlBu_r
    vmin=-0.02
    vmax=0.1
    #plot
    rsa_subsample_plot(roi_rsa_mean, subsampled_time_ref, matrices_delimitations_ref, sub_matrix_dict,vmin,vmax,cmap,fname_fig_sub)
    
    #correlated with theory rdm
    corr_theory(rsa_subsample,analysis_name,decoding_name='Ori')


def stat_cluster_1sample_RDM(gc_mean,test_win_on,test_win_off,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 1 # two-tailed
    n_observations=gc_mean.shape[0]
    stat_time_points=gc_mean[:,test_win_on:test_win_off,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='mask',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    return C1_stat


################
#set data root
if analysis_name=='RSA_Cat':
    analysis_index='RSA_Cat_NoFS'    
    group_deriv_root,stat_data_root,stat_figure_root=set_path_plot(bids_root,visit_id, analysis_index,con_C[0])
else:
    group_deriv_root,stat_data_root,stat_figure_root=set_path_plot(bids_root,visit_id, analysis_name,con_C[0])    


## analysis/task info
if con_T.__len__() == 3:
    con_Tname = 'T_all'
elif con_T.__len__() == 2:
    con_Tname = con_T[0]+'_'+con_T[1]
else:
    con_Tname = con_T[0]

task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
print(task_info)

fname_data=op.join(group_deriv_root, task_info +"_data_group_" + analysis_name +
                   '.pickle')

fr=open(fname_data,'rb')
group_data=pickle.load(fr)




if analysis_name=='RSA_ID':
    #sub_list.remove('SB006')
    #sub_list.remove('SB003')
    # GNW ROI
    roi_name='GNW' 
    RSA_ID_plot(roi_name)
    
    # IIT ROI
    roi_name='IIT' 
    RSA_ID_plot(roi_name)
    
elif analysis_name=='RSA_Cat':
    #sub_list.remove('SB006')
    #sub_list.remove('SB003')
    # GNW ROI
    roi_name='GNW'
    condition='Irrelevant'
    RSA_Cat_plot(roi_name,condition)
    condition='Relevant non-target'
    RSA_Cat_plot(roi_name,condition)
    
    # IIT ROI
    roi_name='IIT' 
    condition='Irrelevant'
    RSA_Cat_plot(roi_name,condition)
    condition='Relevant non-target'
    RSA_Cat_plot(roi_name,condition)

elif analysis_name=='RSA_Ori':
    #sub_list.remove('SB006')
    #sub_list.remove('SB003')
    # GNW ROI
    roi_name='GNW' 
    RSA_Ori_plot(roi_name)
    
    # IIT ROI
    roi_name='IIT' 
    RSA_Ori_plot(roi_name)



