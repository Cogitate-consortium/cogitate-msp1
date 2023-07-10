"""
====================
D98. Group analysis for decoding
genelaization across time (GAT)
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

from scipy.ndimage import gaussian_filter

import matplotlib.patheffects as path_effects

import matplotlib.colors as mcolors


import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root

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
                    default='GAT_Cat',
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
    


# Color parameters:
cmap = "RdYlBu_r"
#color_blind_palette = sns.color_palette("colorblind")
colors = {
    "IIT": [
        0.00392156862745098,
        0.45098039215686275,
        0.6980392156862745
    ],
    "GNW": [
        0.00784313725490196,
        0.6196078431372549,
        0.45098039215686275
    ],    
    "MT": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "FP": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
        ],
    "Relevant to Irrelevant": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "Irrelevant to Relevant": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
        ],
    "Relevant non-target": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "Irrelevant": [
        0.5450980392156862,
        0.16862745098039217,
        0.8862745098039215
    ],
}


time_point = np.array(range(-200,2001, 10))/1000
# set the path for decoding analysis
def set_path_plot(bids_root, visit_id, analysis_name, con_name):

    ### I  Set the group Data Path
    # Set path to decoding derivatives
    decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa')

    data_path=op.join(decoding_path,analysis_name)

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



def df_plot_cluster_GAT(gc_mean,C1_stat,C2_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    
    talk_rc={'lines.linewidth':1,'lines.markersize':1}
    sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    fig, axes = plt.subplots(1, 1,figsize=(10,10),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    
    
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    #cmap = mpl.cm.RdYlBu_r
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                     np.vstack((plt.cm.Blues_r(np.linspace(0, 1, 220) ),
                                                                plt.cm.Blues_r( np.linspace(1, 1, 36) ), 
                                                                plt.cm.Reds( np.linspace(0, 0, 36) ),
                                                                plt.cm.Reds( np.linspace(0, 1, 220) ) ) ) )
    vmin = 0
    vmax = 100
    # bounds = np.linspace(vmin, vmax, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #plot
    GAT_avg=np.mean(gc_mean[0,:,30:251,30:251],axis=0)
    GAT_avg_plot = np.nan * np.ones_like(GAT_avg)
    for c, p_val in zip(C1_stat['cluster'], C1_stat['cluster_p']):
        if p_val <= 0.05:
            GAT_avg_plot[c] = GAT_avg[c]
    
    im = axes.imshow(gaussian_filter(GAT_avg,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,alpha=0.9,
                    extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.contour(GAT_avg_plot > 0, GAT_avg_plot > 0, colors="black", linewidths=1.5, origin="lower",extent=t[[0, -1, 0, -1]])
    im = axes.imshow(GAT_avg_plot, origin='lower', cmap=cmap,aspect='auto',
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.set_xticks([0,1,2])
    axes.set_yticks([0,1,2])
    axes.set_title(task_index[0])
    axes.axvline(0, color='k',linestyle='--')
    axes.axhline(0, color='k',linestyle='--')
    axes.axline((0, 0), slope=1, color='k',linestyle='--')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    cb = axes.figure.axes[-1]
    m = axes.figure.axes[-2]
    pos = m.get_position().bounds
    cb.set_position([pos[2]+pos[0]+0.01, pos[1], 0.1, pos[3]])
    
    
    
    fname_fig_1=op.join(fname_fig+'_'+task_index[0]+'.svg')
    
    fig.savefig(fname_fig_1,format="svg")
    
    
    talk_rc={'lines.linewidth':1,'lines.markersize':1}
    sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    fig, axes = plt.subplots(1, 1,figsize=(10,10),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    #cmap = mpl.cm.RdYlBu_r
    
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                     np.vstack((plt.cm.Blues_r(np.linspace(0, 1, 220) ),
                                                                plt.cm.Blues_r( np.linspace(1, 1, 36) ), 
                                                                plt.cm.Reds( np.linspace(0, 0, 36) ),
                                                                plt.cm.Reds( np.linspace(0, 1, 220) ) ) ) )
    vmin = 0
    vmax = 100
    # bounds = np.linspace(vmin, vmax, 11)
    
    GAT2_avg=np.mean(gc_mean[1,:,30:251,30:251],axis=0)
    GAT2_avg_plot = np.nan * np.ones_like(GAT_avg)
    for c2, p2_val in zip(C2_stat['cluster'], C2_stat['cluster_p']):
        if p2_val <= 0.05:
            GAT2_avg_plot[c2] = GAT2_avg[c2]
    im = axes.imshow(gaussian_filter(np.mean(gc_mean[1,:,30:251,30:251],axis=0), sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,alpha=0.9,
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.contour(GAT2_avg_plot > 0, GAT2_avg_plot > 0, colors="black", linewidths=1.5, origin="lower",extent=t[[0, -1, 0, -1]])
    im = axes.imshow(GAT2_avg_plot, origin='lower', cmap=cmap,aspect='auto',
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.set_xticks([0,1,2])
    axes.set_yticks([0,1,2])
    axes.set_title(task_index[1])
    axes.axvline(0, color='k',linestyle='--')
    axes.axhline(0, color='k',linestyle='--')
    axes.axline((0,0), slope=1, color='k',linestyle='--')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    cb = axes.figure.axes[-1]
    m = axes.figure.axes[-2]
    pos = m.get_position().bounds
    cb.set_position([pos[2]+pos[0]+0.01, pos[1], 0.1, pos[3]])
    
    fname_fig_2=op.join(fname_fig+'_'+task_index[1]+'.svg')
    
    fig.savefig(fname_fig_2,format="svg")
 

def df_plot_cluster_GAT_ori(gc_mean,C1_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    
    
    fig, axes = plt.subplots(1, 1,figsize=(5,5),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    
    
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    
    #cmap = mpl.cm.RdYlBu_r
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                     np.vstack((plt.cm.Blues_r(np.linspace(0, 1, 220) ),
                                                                plt.cm.Blues_r( np.linspace(1, 1, 36) ), 
                                                                plt.cm.Reds( np.linspace(0, 0, 36) ),
                                                                plt.cm.Reds( np.linspace(0, 1, 220) ) ) ) )
    vmin = 0
    vmax = 100
    # bounds = np.linspace(0, 100, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #plot
    GAT_avg=np.mean(gc_mean[0,:,30:251,30:251],axis=0)
    GAT_avg_plot = np.nan * np.ones_like(GAT_avg)
    for c, p_val in zip(C1_stat['cluster'], C1_stat['cluster_p']):
        if p_val <= 0.05:
            GAT_avg_plot[c] = GAT_avg[c]
    
    im = axes.imshow(gaussian_filter(GAT_avg,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,alpha=0.9,
                    extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.contour(GAT_avg_plot > 0, GAT_avg_plot > 0, colors="black", linewidths=1.5, origin="lower",extent=t[[0, -1, 0, -1]])
    im = axes.imshow(GAT_avg_plot, origin='lower', cmap=cmap,aspect='auto',
                   extent=t[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.set_title('GAT_Ori')
    axes.axvline(0, color='k',linestyle='--')
    axes.axhline(0, color='k',linestyle='--')
    axes.axline((0, 0), slope=1, color='k',linestyle='--')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    cb = axes.figure.axes[-1]
    m = axes.figure.axes[-2]
    pos = m.get_position().bounds
    cb.set_position([pos[2]+pos[0]+0.01, pos[1], 0.1, pos[3]])
    
    talk_rc={'lines.linewidth':1,'lines.markersize':1}
    sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    fig.savefig(fname_fig,format="svg")



def stat_cluster_1sample_GAT(gc_mean,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='mask',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
   
    T_obs_2, clusters_2, cluster_p_values_2, H0_2 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[1,:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='indices',verbose=None)
    
    C2_stat=dict()
    C2_stat['T_obs']=T_obs_2
    C2_stat['cluster']=clusters_2
    C2_stat['cluster_p']=cluster_p_values_2

    
    return C1_stat,C2_stat

def stat_cluster_1sample_GAT_ori(gc_mean,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=1000, tail=tail, out_type='mask',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    return C1_stat

    


def dat2gat(dat,roi_name,cond_name,decoding_name):
    roi_gat=np.zeros([2,len(sub_list),251,251])
    for ci, cond in enumerate(cond_name):
        roi_gat_gc=np.zeros([len(sub_list),251,251])
        for i, sbn in enumerate(sub_list):
            roi_gat_gc[i,:,:]=dat[sbn][decoding_name][roi_name][cond]
           
        
        roi_gat[ci,:,:,:]=roi_gat_gc*100
        
    return roi_gat   

def dat2gat_ori(dat,roi_name,cond_name,decoding_name):
    roi_gat=np.zeros([1,len(sub_list),251,251])
    #for ci, cond in enumerate(cond_name):
    roi_gat_gc=np.zeros([len(sub_list),251,251])
    for i, sbn in enumerate(sub_list):
        roi_gat_gc[i,:,:]=dat[sbn][decoding_name][roi_name][cond_name]
           
        
    roi_gat[0,:,:,:]=roi_gat_gc*100
        
    return roi_gat   


def ctccd_plt(group_data,con_name,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=50,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    ROI_gat_g=dat2gat(group_data,roi_name,cond_name=['RE2IR','IR2RE'],decoding_name='ctccd_acc')
    
   
    #cluster based methods
    
    #stat
    C1_stat,C2_stat=stat_cluster_1sample_GAT(ROI_gat_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig_index= op.join(stat_figure_root, roi_name + '_'+str(con_name)+"_acc_CTCCD_cluster" )
    
    #plot
    df_plot_cluster_GAT(ROI_gat_g,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig_index)
    

def ctwcd_plt(group_data,con_name,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=50,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI_gat_g=dat2gat(group_data,roi_name,cond_name=['Irrelevant','Relevant non-target'],decoding_name='ctwcd_acc')
    
    
    
    #cluster based methods
    
    #stat
    C1_stat,C2_stat=stat_cluster_1sample_GAT(ROI_gat_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(con_name)+"_acc_CTWCD_cluster")
    
    #plot
    df_plot_cluster_GAT(ROI_gat_g,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)

    
    



def ctwcd_ori_plt(group_data,con_name,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=33.3,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=conditions_C #Face/Object/Letter/False
    #get decoding data
    ROI_ori_g=dat2gat_ori(group_data,roi_name,conditions_C[0],decoding_name='ctwcd_ori_acc')
    
    
    # #FDR methods
    
    # #stat
    # ts_df_fdr,T1,pval1,T2,pval2=gc2df(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    # #plot
    # fname_fdr_fig= op.join(stat_figure_root, roi_name + '_'+ str(test_win_on) + '_'+ str(test_win_off) +"_acc_WCD_fdr" + '.png')
    
    # sig1_fdr,sig2_fdr=df_plot(ts_df_fdr,T1,pval1,T2,pval2,time_point,test_win_on,
    #                           roi_name,task_index=task_index,
    #                           chance_index=chance_index,y_index=y_index,fname_fig=fname_fdr_fig)
    
    
    
    
    #cluster based methods
    
    #stat
    C1_stat=stat_cluster_1sample_GAT_ori(ROI_ori_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(con_name)+"_acc_CTWCD_ori_cluster" + '.svg')
    
    #plot
    df_plot_cluster_GAT_ori(ROI_ori_g,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)

    


#########
#set data root    
group_deriv_root,stat_data_root,stat_figure_root=set_path_plot(bids_root,visit_id, analysis_name,con_C[0])    


# ########
# #debug
# decoding_path=op.join(bids_root, "derivatives",'decoding')

# data_path=op.join(decoding_path,methods_name)

# # Set path to group analysis derivatives
# group_deriv_root = op.join(data_path, "group", analysis_name)
# if not op.exists(group_deriv_root):
#     os.makedirs(group_deriv_root)




# analysis/task info
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



if analysis_name=='GAT_Cat': 
    #CCD: cross condition decoding
    #GNW
    
    # # 300ms to 500ms
    # ccd_plt(group_data2,roi_name='GNW',test_win_on=130, test_win_off=150,chance_index=50,y_index=15)
    
    # 0ms to 1500ms
    ctccd_plt(group_data,con_Tname,roi_name='GNW',test_win_on=30, test_win_off=251,chance_index=50,y_index=40)
    
    #IIT
    
    # # 300ms to 500ms
    # ccd_plt(group_data2,roi_name='IIT',test_win_on=130, test_win_off=250,chance_index=50,y_index=40)
    
    # 0ms to 1500ms
    ctccd_plt(group_data,con_Tname,roi_name='IIT',test_win_on=30, test_win_off=251,chance_index=50,y_index=40)
    
    
    #WCD: within condition decoding
    #GNW
    
    # # 300ms to 500ms
    # wcd_plt(group_data2,roi_name='GNW',test_win_on=130, test_win_off=150,chance_index=50,y_index=15)
    
    # 0ms to 1500ms
    ctwcd_plt(group_data,con_Tname,roi_name='GNW',test_win_on=30, test_win_off=251,chance_index=50,y_index=40)
    
    #IIT
    
    # # 300ms to 500ms
    # wcd_plt(group_data2,roi_name='IIT',test_win_on=130, test_win_off=250,chance_index=50,y_index=40)
    
    # 0ms to 1500ms
    ctwcd_plt(group_data,con_Tname,roi_name='IIT',test_win_on=30, test_win_off=251,chance_index=50,y_index=40)

   
elif analysis_name=='GAT_Ori':
    
    ctwcd_ori_plt(group_data,con_Tname,roi_name='GNW',test_win_on=30, test_win_off=251,chance_index=33.3,y_index=40)
    ctwcd_ori_plt(group_data,con_Tname,roi_name='IIT',test_win_on=30, test_win_off=251,chance_index=33.3,y_index=40)
        
