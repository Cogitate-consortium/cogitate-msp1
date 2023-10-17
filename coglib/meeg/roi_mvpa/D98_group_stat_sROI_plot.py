"""
====================
D98. Group analysis for decoding
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
import pandas as pd
import seaborn as sns
sns.set_theme(style='ticks')

from mne.stats import fdr_correction


from scipy import stats as stats



import matplotlib as mpl


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
parser.add_argument('--space',
                    type=str,
                    default='surface',
                    help='source space ("surface" or "volume")')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--analysis',
                    type=str,
                    default='Cat',
                    help='the name for anlaysis, e.g. Cat or Ori or GAT_Cat')


opt = parser.parse_args()

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path
analysis_name=opt.analysis


opt = parser.parse_args()
con_C = opt.cC
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
param = plot_param
colors=param['colors']
fig_size = param["figure_size_mm"]
plt.rc('font', size=8)  # controls default text size
plt.rc('axes', labelsize=20)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('xtick.major', width=2, size=4)
plt.rc('ytick.major', width=2, size=4)
plt.rc('legend', fontsize=18)
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

def mm2inch(val):
    return val / 25.4

# Color parameters:
cmap = "RdYlBu_r"


time_point = np.array(range(-200,2001, 10))/1000
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

def df_plot(ts_df,T1,pval1,T2,pval2,time_point,test_win_on,roi_name,task_index,chance_index,y_index,fname_fig):
    if roi_name=='GNW':
       window=[0.3,0.5,0.5,0.3]
    elif roi_name=='IIT':
        window=[0.3,1.5,1.5,0.3]    
    elif roi_name=='MT':
        window=[0.25,0.5,0.5,0.25]    
    elif roi_name=='FP':
        window=[0.3,1.5,1.5,0.3]    
    #plot with sns
    
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    reject_fdr1, pval_fdr1 = fdr_correction(pval1, alpha=0.05, method='indep')
    temp=reject_fdr1.nonzero()
    sig1=np.full(time_point.shape,np.nan)
    if len(temp[0])>=1:
        threshold_fdr1 = np.min(np.abs(T1)[reject_fdr1])
        T11=np.concatenate((np.zeros((test_win_on-30,)),T1))
        clusters1 = np.where(T11 > threshold_fdr1)[0]
        if len(clusters1)>1:
            clusters1 = clusters1[clusters1 > test_win_on-30]
            #times = range(0, 500, 10)
            plt.plot(time_point[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[task_index[0]])
            sig1[clusters1]=1
    
    reject_fdr2, pval_fdr2 = fdr_correction(pval2, alpha=0.05, method='indep')
    temp=reject_fdr2.nonzero()
    sig2=np.full(time_point.shape,np.nan)
    if len(temp[0])>=1:
        threshold_fdr2 = np.min(np.abs(T2)[reject_fdr2])
        T22=np.concatenate((np.zeros((test_win_on-30,)),T2))
        clusters2 = np.where(T22 > threshold_fdr2)[0]
        if len(clusters2)>1:
            clusters2 = clusters2[clusters2 > test_win_on-30]
            #times = range(0, 500, 10)
            plt.plot(time_point[clusters2], np.zeros(clusters2.shape) + 30, 'o', linewidth=3,color=colors[task_index[1]])
            sig2[clusters2]=1
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([15,100])
    plt.xticks([0,0.5,1.0,1.5,2])
    plt.yticks([20,40,60,80,100])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1, sig2

def df_plot_cluster(ts_df,C1_stat,C2_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    if roi_name=='GNW':
       window=[0.3,0.5,0.5,0.3]
    elif roi_name=='IIT':
        window=[0.3,1.5,1.5,0.3]
    elif roi_name=='MT':
        window=[0.25,0.5,0.5,0.25]
    elif roi_name=='FP':
        window=[0.3,1.5,1.5,0.3]
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 40, 'o', linewidth=3,color=colors[task_index[0]])
                sig1[clusters1]=i
            
    temp2=C2_stat['cluster']
    temp_p2=C2_stat['cluster_p']
    sig2=np.full(time_point.shape,np.nan)
    if len(temp2)>=1:
        for i in range(len(temp2)):
            if temp_p2[i]<0.05:# plot the cluster which  p < 0.05
                clusters2=temp2[i][0]
                plt.plot(time_index[clusters2], np.zeros(clusters2.shape) + 30, 'o', linewidth=3,color=colors[task_index[1]])
                sig2[clusters2]=i
    
    
            
    #plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([15,100])
    plt.xticks([0,0.5,1.0,1.5,2])
    plt.yticks([20,40,60,80,100])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1, sig2

def df_plot_cluster_ori(ts_df,C1_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    if roi_name=='GNW':
       window=[0.3,0.5,0.5,0.3]
    elif roi_name=='IIT':
        window=[0.3,1.5,1.5,0.3]
    elif roi_name=='MT':
        window=[0.25,0.5,0.5,0.25]
    elif roi_name=='FP':
        window=[0.3,1.5,1.5,0.3]
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('paper',rc=talk_rc,font_scale=4)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors,legend=False)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    
    # leg = g._legend
    # leg.remove()
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + 30, 'o', linewidth=3,color=colors[task_index[0]])
                sig1[clusters1]=i
               
    
            
    # plt.fill(window,[15,15,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([25,100])
    plt.xticks([0,0.5,1.0,1.5,2])
    plt.yticks([40,60,80,100])
    
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1

def df_plot_ROI_cluster(ts_df,C1_stat,time_point,test_win_on,test_win_off,chance_index,y_index,fname_fig):
    
    window=[0.3,1.5,1.5,0.3]
    
    
    #plot with sns
    
    # talk_rc={'lines.linewidth':2,'lines.markersize':4}
    # sns.set_context('talk',rc=talk_rc,font_scale=1)
    
    
    g = sns.relplot(x="time(s)", y="decoding accuracy(%)", kind="line", data=ts_df,hue='ROI',aspect=2,palette=colors,legend=True)
    g.fig.set_size_inches(mm2inch(fig_size[0]),mm2inch(fig_size[1]))
    #sns.move_legend(g, "upper left", bbox_to_anchor=(.72, .8), frameon=False)
    #leg = g._legend
    # leg.remove()
    #leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    #plt.axvline(0.5, color='gray', linestyle='--')
    #plt.axvline(1, color='gray', linestyle='--')
    #plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-5, 'o', linewidth=3,color=colors['IIT'])
                sig1[clusters1]=i
    
    
            
    #plt.fill(window,[40,40,100,100],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([25,100])
    plt.xticks([0,0.5,1.0,1.5,2])
    plt.yticks([40,60,80,100])
    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    return sig1

def g2gdat(roi_g,time_point,sig1,sig2):
    roi_g_acc=np.mean(roi_g[:,:,30:251],axis=1)
    roi_g_ci=1.96*stats.sem(roi_g[:,:,30:251],axis=1)
    roi_g_dat=np.vstack((time_point,roi_g_acc,roi_g_ci,sig1,sig2))
    return roi_g_dat

def g2gdat_ori(roi_g,time_point,sig1):
    roi_g_acc=np.mean(roi_g[:,:,30:251],axis=1)
    roi_g_ci=1.96*stats.sem(roi_g[:,:,30:251],axis=1)
    roi_g_dat=np.vstack((time_point,roi_g_acc,roi_g_ci,sig1))
    return roi_g_dat

def df2csv(np_data,task_index,csv_fname):
    columns_index=['Time',
                   'ACC (' + task_index[0] + ')','ACC (' + task_index[1] + ')',
                   'CI (' + task_index[0] + ')','CI (' + task_index[1] + ')',
                   'sig (' + task_index[0] + ')','sig (' + task_index[1] + ')']
    df = pd.DataFrame(np_data.T, columns=columns_index)
    df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')
    
def df2csv_ori(np_data,task_index,csv_fname):
    columns_index=['Time',
                   'ACC (' + task_index[0] + ')',
                   'CI (' + task_index[0] + ')',
                   'sig (' + task_index[0] + ')']
    df = pd.DataFrame(np_data.T, columns=columns_index)
    df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')
    
def gc2df(gc_mean,test_win_on,test_win_off,task_index,chance_index):

    df1 = pd.DataFrame(gc_mean[0,:,30:251], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T1, pval1 = stats.ttest_1samp(gc_mean[0,:,test_win_on:test_win_off], chance_index)
    
    df2 = pd.DataFrame(gc_mean[1,:,30:251], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='Task',value=task_index[1])
    
    T2, pval2 = stats.ttest_1samp(gc_mean[1,:,test_win_on:test_win_off], chance_index)
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,T1,pval1,T2,pval2

def stat_cluster_1sample(gc_mean,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    df1 = pd.DataFrame(gc_mean[0,:,30:251], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    df2 = pd.DataFrame(gc_mean[1,:,30:251], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='Task',value=task_index[1])
    
    T_obs_2, clusters_2, cluster_p_values_2, H0_2 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[1,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C2_stat=dict()
    C2_stat['T_obs']=T_obs_2
    C2_stat['cluster']=clusters_2
    C2_stat['cluster_p']=cluster_p_values_2
    
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,C1_stat,C2_stat

def stat_cluster_1sample_ori(gc_mean,test_win_on,test_win_off,task_index,chance_index):
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=gc_mean.shape[1]
    stat_time_points=gc_mean[:,:,test_win_on:test_win_off].shape[2]
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    df1 = pd.DataFrame(gc_mean[0,:,30:251], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='Task',value=task_index[0])
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_1samp_test(
        gc_mean[0,:,test_win_on:test_win_off]-np.ones([n_observations,stat_time_points])*chance_index, 
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    
    ts_df = pd.melt(df1, id_vars=['SUBID','Task'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    return ts_df,C1_stat


def stat_cluster_1sample_roi(ROI1_data,ROI2_data,test_win_on,test_win_off,ROI_name):
    
    # define theresh
    pval = 0.05  # arbitrary
    tail = 0 # two-tailed
    n_observations=ROI1_data.shape[1]

    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    df1 = pd.DataFrame(ROI1_data[:,30:251], columns=time_point)
    df1.insert(loc=0, column='SUBID', value=sub_list)
    df1.insert(loc=0, column='ROI',value=ROI_name[0])
    

    
    df2 = pd.DataFrame(ROI2_data[:,30:251], columns=time_point)
    df2.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='ROI',value=ROI_name[1])
    
    
    df=df1.append(df2)
    
    ts_df = pd.melt(df, id_vars=['SUBID','ROI'], var_name='time(s)', value_name='decoding accuracy(%)', value_vars=time_point)
    
    T_obs_1, clusters_1, cluster_p_values_1, H0_1 = mne.stats.permutation_cluster_test(
        [ROI1_data[:,test_win_on:test_win_off] , ROI2_data[:,test_win_on:test_win_off]],
        threshold=thresh, n_permutations=10000, tail=tail, out_type='indices',verbose=None)
    
    C1_stat=dict()
    C1_stat['T_obs']=T_obs_1
    C1_stat['cluster']=clusters_1
    C1_stat['cluster_p']=cluster_p_values_1
    
    return ts_df,C1_stat
    

def dat2g(dat,roi_name,cond_name,decoding_name):
    roi_ccd_g=np.zeros([2,len(sub_list),251])
    for ci, cond in enumerate(cond_name):
        roi_ccd_gc=np.zeros([len(sub_list),251])
        for i, sbn in enumerate(sub_list):
            roi_ccd_gc[i,:]=dat[sbn][decoding_name][roi_name][cond]
           
        
        roi_ccd_g[ci,:,:]=roi_ccd_gc*100
        
    return roi_ccd_g

def dat2g_PFC(dat,cond_name):
    roi_wcd_g=np.zeros([3,len(sub_list),251])
    for ci, cond in enumerate(cond_name):
        roi_wcd_gc=np.zeros([len(sub_list),251])
        for i, sbn in enumerate(sub_list):
            roi_wcd_gc[i,:]=dat[sbn][cond]
        roi_wcd_g[ci,:,:]=roi_wcd_gc*100
        
    return roi_wcd_g


def dat2g_ori(dat,roi_name,cond_name,decoding_name):
    roi_ccd_g=np.zeros([1,len(sub_list),251])
    roi_ccd_gc=np.zeros([len(sub_list),251])
    for i, sbn in enumerate(sub_list):
        roi_ccd_gc[i,:]=dat[sbn][decoding_name][roi_name][cond_name]
    roi_ccd_g[0,:,:]=roi_ccd_gc*100
        
    return roi_ccd_g


def dat2gat(dat,roi_name,cond_name,decoding_name):
    roi_ccd_g=np.zeros([2,len(sub_list),251,251])
    for ci, cond in enumerate(cond_name):
        roi_ccd_gc=np.zeros([len(sub_list),251,251])
        for i, sbn in enumerate(sub_list):
            roi_ccd_gc[i,:,:]=np.diagonal(dat[sbn][decoding_name][roi_name][cond])
           
        
        roi_ccd_g[ci,:,:,:]=roi_ccd_gc*100
        
    return roi_ccd_g   

def dat2gat2(dat,roi_name,cond_name,decoding_name):
    roi_ccd_g=np.zeros([2,len(sub_list),251])
    for ci, cond in enumerate(cond_name):
        roi_ccd_gc=np.zeros([len(sub_list),251])
        for i, sbn in enumerate(sub_list):
            roi_ccd_gc[i,:]=np.diagonal(dat[sbn][decoding_name][roi_name][cond])
           
        
        roi_ccd_g[ci,:,:]=roi_ccd_gc*100
        
    return roi_ccd_g   


def ccd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    ROI_ccd_g=dat2g(group_data,roi_name,cond_name=['RE2IR','IR2RE'],decoding_name='ccd_acc')
    
    
    
    # #FDR methods
    
    # #stat
    # ts_df_fdr,T1,pval1,T2,pval2=gc2df(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    # #plot
    # fname_fdr_fig= op.join(stat_figure_root, roi_name + '_'+ str(test_win_on) + '_'+ str(test_win_off) +"_acc_CCD_fdr" + '.png')
    
    # sig1_fdr,sig2_fdr=df_plot(ts_df_fdr,T1,pval1,T2,pval2,time_point,test_win_on,
    #                           roi_name,task_index=task_index,
    #                           chance_index=chance_index,y_index=y_index,fname_fig=fname_fdr_fig)
    
    
    
    
    #cluster based methods
    
    #stat
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CCD_cluster" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster=df_plot_cluster(ts_df_cluster,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    #prepare data for plt plot   
    ROI_ccd_g_dat=g2gdat(ROI_ccd_g,time_point,sig1_cluster,sig2_cluster)

    
    csv_fname=op.join(stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_CCD_cluster" + '.csv')

    df2csv(ROI_ccd_g_dat,task_index,csv_fname)
    

def wcd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI_wcd_g=dat2g(group_data,roi_name,cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')
    
    
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
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_wcd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster=df_plot_cluster(ts_df_cluster,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    #prepare data for plt plot   
    ROI_wcd_g_dat=g2gdat(ROI_wcd_g,time_point,sig1_cluster,sig2_cluster)

    
    csv_fname=op.join(stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.csv')

    df2csv(ROI_wcd_g_dat,task_index,csv_fname)

def ROI_wcd_plt(group_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40):
    ROI_name=['IIT','FP']
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI1_data=dat2g(group_data,ROI_name[0],cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')   
    ROI2_data=dat2g(group_data,ROI_name[1],cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')
    
    time_point = np.array(range(-200,2001, 10))/1000
   
    #cluster based methods
    
    #stat
    ts1_df_cluster,C1_stat=stat_cluster_1sample_roi(ROI1_data[0,:,:],ROI2_data[0,:,:],test_win_on,test_win_off,ROI_name)
    
    fname_cluster_fig= op.join(stat_figure_root, task_index[0] + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_FP_P_diff_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts1_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              task_index=task_index[0],
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    #stat
    ts2_df_cluster,C2_stat=stat_cluster_1sample_roi(ROI1_data[1,:,:],ROI2_data[1,:,:],test_win_on,test_win_off,ROI_name)
    
    fname_cluster_fig2= op.join(stat_figure_root, task_index[1] + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_FP_P_diff_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts2_df_cluster,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              task_index=task_index[1],
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig2)

    
    
def ROI_ccd_plt(group_data,decoding_method ='ccd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40):
    ROI_name=['IIT','FP']
    task_index=['Relevant to Irrelevant','Irrelevant to Relevant']
    #get decoding data
    ROI1_data=dat2g(group_data,ROI_name[0],cond_name=['RE2IR','IR2RE'],decoding_name='ccd_acc')   
    ROI2_data=dat2g(group_data,ROI_name[1],cond_name=['RE2IR','IR2RE'],decoding_name='ccd_acc')
    
    time_point = np.array(range(-200,2001, 10))/1000
   
    #cluster based methods
    
    #stat
    ts1_df_cluster,C1_stat=stat_cluster_1sample_roi(ROI1_data[0,:,:],ROI2_data[0,:,:],test_win_on,test_win_off,ROI_name)
    
    fname_cluster_fig= op.join(stat_figure_root, task_index[0] + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_FP_P_diff_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts1_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              task_index=task_index[1],
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    
    #stat
    ts2_df_cluster,C2_stat=stat_cluster_1sample_roi(ROI1_data[1,:,:],ROI2_data[1,:,:],test_win_on,test_win_off,ROI_name)
    
    fname_cluster_fig2= op.join(stat_figure_root, task_index[1] + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_FP_P_diff_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts2_df_cluster,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              task_index=task_index[1],
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig2)



def wcd_ori_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=33.3,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=conditions_C #Face/Object/Letter/False
    #get decoding data
    ROI_ori_g=dat2g_ori(group_data,roi_name,cond_name=conditions_C[0],decoding_name='wcd_ori_acc')
    
    
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
    ts_df_cluster,C1_stat=stat_cluster_1sample_ori(ROI_ori_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_ori_cluster" + '.svg')
    
    #plot
    sig1_cluster=df_plot_cluster_ori(ts_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    #prepare data for plt plot   
    ROI_ori_g_dat=g2gdat_ori(ROI_ori_g,time_point,sig1_cluster)

    
    csv_fname=op.join(stat_data_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_ori_cluster" + '.csv')

    df2csv_ori(ROI_ori_g_dat,task_index,csv_fname)


def ROI_wcd_ori_plt(group_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=33.3,y_index=40):
    ROI_name=['IIT','FP']
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI1_data=dat2g_ori(group_data,ROI_name[0],cond_name=conditions_C[0],decoding_name='wcd_ori_acc')
    ROI2_data=dat2g_ori(group_data,ROI_name[1],cond_name=conditions_C[0],decoding_name='wcd_ori_acc')
    
    time_point = np.array(range(-200,2001, 10))/1000
   
    #cluster based methods
    
    #stat
    ts1_df_cluster,C1_stat=stat_cluster_1sample_roi(ROI1_data[0,:,:],ROI2_data[0,:,:],test_win_on,test_win_off,ROI_name)
    
    fname_cluster_fig= op.join(stat_figure_root, task_index[0] + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_FP_P_diff_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts1_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              task_index=task_index[0],
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)
    


#########
#set data root    
group_deriv_root,stat_data_root,stat_figure_root=set_path_plot(bids_root,visit_id, analysis_name,con_C[0])    


# ########
# #debug
# decoding_path=op.join(bids_root, "derivatives",'decoding')

# data_path=op.join(decoding_path, analysis_name)

# # Set path to group analysis derivatives
# group_deriv_root = op.join(data_path, "group")
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



if analysis_name=='Cat' or analysis_name=='Cat_offset_control':
    #CCD: cross condition decoding
    #GNW
    
    # # 300ms to 500ms
    # ccd_plt(group_data2,roi_name='GNW',test_win_on=130, test_win_off=150,chance_index=50,y_index=15)
    
    # 0ms to 1500ms
    ccd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    
    #IIT
    
    # # 300ms to 500ms
    # ccd_plt(group_data2,roi_name='IIT',test_win_on=130, test_win_off=251,chance_index=50,y_index=40)
    
    # 0ms to 1500ms
    ccd_plt(group_data,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    
    
    #WCD: within condition decoding
    #GNW
    
    # # 300ms to 500ms
    # wcd_plt(group_data2,roi_name='GNW',test_win_on=130, test_win_off=150,chance_index=50,y_index=15)
    
    # 0ms to 1500ms
    wcd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    
    #IIT
    
    # # 300ms to 500ms
    # wcd_plt(group_data2,roi_name='IIT',test_win_on=130, test_win_off=251,chance_index=50,y_index=40)
    
    # 0ms to 1500ms
    wcd_plt(group_data,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)

    #compare IIT with IIT+GNW(FP)
    ROI_ccd_plt(group_data,decoding_method ='ccd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    
    ROI_wcd_plt(group_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40)


elif analysis_name=='Cat_MT_control':
    ccd_plt(group_data,roi_name='MT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    wcd_plt(group_data,roi_name='MT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40) 
    
elif analysis_name=='Cat_baseline':
        
    wcd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
    wcd_plt(group_data,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)

elif analysis_name=='Ori':
    
    wcd_ori_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=33.3,y_index=40)
    wcd_ori_plt(group_data,roi_name='IIT',test_win_on=50, test_win_off=200,chance_index=33.3,y_index=40)
    
elif analysis_name=='Cat_PFC':
    cond_name=['IIT','IITPFC_f','IITPFC_m']
    colors = {
        "IIT": [1,0,0
        ],
        "IITPFC_f": [0,0,1
        ],    
        "IITPFC_m": [0,0,1
            ]}
    decoding_method=analysis_name
    #task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name)   
    
    
    time_point = np.array(range(-200,2001, 10))/1000
   
    #cluster based methods
    test_win_on=50
    test_win_off=200
    #stat
    ts1_df_cluster,C1_stat=stat_cluster_1sample_roi(PFC_data[0,:,:],PFC_data[1,:,:],test_win_on,test_win_off,['IIT','IITPFC_f'])
    
    fname_cluster_fig= op.join(stat_figure_root, decoding_method + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_IITPFC_feature_diff_acc_cluster.svg')
    
    # fname_cluster_fig= op.join(data_path, decoding_method + 
    #                            '_'+str(test_win_on) + '_' + str(test_win_off) +
    #                            '_IITPFC_feature_diff_acc_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts1_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              chance_index=50,y_index=50,
                                              fname_fig=fname_cluster_fig)
    
    #stat
    ts2_df_cluster,C2_stat=stat_cluster_1sample_roi(PFC_data[0,:,:],PFC_data[2,:,:],test_win_on,test_win_off,['IIT','IITPFC_m'])
    
    fname_cluster_fig2= op.join(stat_figure_root, decoding_method + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_IITPFC_model_diff_acc_cluster.svg')
    
    # fname_cluster_fig2= op.join(data_path, decoding_method + 
    #                            '_'+str(test_win_on) + '_' + str(test_win_off) +
    #                            '_IITPFC_model_diff_acc_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts2_df_cluster,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              chance_index=50,y_index=50,
                                              fname_fig=fname_cluster_fig2)


elif analysis_name=='Ori_PFC':
    cond_name=['IIT','IITPFC_f','IITPFC_m']
    colors = {
        "IIT": [1,0,0
        ],
        "IITPFC_f": [0,0,1
        ],    
        "IITPFC_m": [0,0,1
            ]}
    decoding_method=analysis_name
    #task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name)   
    
    
    time_point = np.array(range(-200,2001, 10))/1000
   
    #cluster based methods
    test_win_on=50
    test_win_off=200
    #stat
    ts1_df_cluster,C1_stat=stat_cluster_1sample_roi(PFC_data[0,:,:],PFC_data[1,:,:],test_win_on,test_win_off,['IIT','IITPFC_f'])
    
    fname_cluster_fig= op.join(stat_figure_root, decoding_method + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_IITPFC_feature_diff_acc_cluster.svg')
    
    # fname_cluster_fig= op.join(data_path, decoding_method + 
    #                            '_'+str(test_win_on) + '_' + str(test_win_off) +
    #                            '_IITPFC_feature_diff_acc_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts1_df_cluster,C1_stat,time_point,
                                              test_win_on,test_win_off,
                                              chance_index=33.3,y_index=50,
                                              fname_fig=fname_cluster_fig)
    
    #stat
    ts2_df_cluster,C2_stat=stat_cluster_1sample_roi(PFC_data[0,:,:],PFC_data[2,:,:],test_win_on,test_win_off,['IIT','IITPFC_m'])
    
    fname_cluster_fig2= op.join(stat_figure_root, decoding_method + 
                               '_'+str(test_win_on) + '_' + str(test_win_off) +
                               '_IITPFC_model_diff_acc_cluster.svg')
    
    # fname_cluster_fig2= op.join(data_path, decoding_method + 
    #                            '_'+str(test_win_on) + '_' + str(test_win_off) +
    #                            '_IITPFC_model_diff_acc_cluster.svg')
    
    #plot
    sig1_cluster=df_plot_ROI_cluster(ts2_df_cluster,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              chance_index=33.3,y_index=50,
                                              fname_fig=fname_cluster_fig2)       
    #ROI_wcd_ori_plt(group_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=33.3,y_index=40)    
