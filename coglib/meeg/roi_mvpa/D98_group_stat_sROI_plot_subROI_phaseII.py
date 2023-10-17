"""
====================
D98. Group analysis for decoding pattern
Category decoding
control analysis,
decoding at subROI.
====================

@author: Ling Liu  ling.liu@pku.edu.cn

"""

import os.path as op
import os

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
from matplotlib import cm


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import plot_param
from sublist_phase2 import sub_list

# get the parameters dictionary
param = plot_param
pcolors=param['colors']
fig_size = param["figure_size_mm"]
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('font', size=param["font_size"]*2)  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"]*2)  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"]*2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"]*2)  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"]*2)  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"]*2)  # legend fontsize
plt.rc('figure', titlesize=param["font_size"]*2)  # fontsize of the fi
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

def mm2inch(val):
    return val / 25.4

#set data path    
subjects_dir = r'Y:\HPC\fs'

    
# decoding_path=op.join(bids_root, "derivatives",'decoding')

# data_path=op.join(decoding_path,'roi_mvpa')

data_path=r'D:\COGITATE_xps\data_analysis\MSP\leakage_control'

# Set path to group analysis derivatives
group_deriv_root = op.join(data_path, "group_phase2",)
if not op.exists(group_deriv_root):
    os.makedirs(group_deriv_root)




stat_figure_root =  op.join(group_deriv_root,"figures")
if not op.exists(stat_figure_root):
    os.makedirs(stat_figure_root)
    
    
    
con_C = ['FO']
con_D = ['Irrelevant', 'Relevant non-target']
con_T = ['500ms','1000ms','1500ms']




if con_C[0] == 'FO':
    conditions_C = ['face', 'object']
    print(conditions_C)
elif con_C[0] == 'LF':
    conditions_C = ['letter', 'false']
    print(conditions_C)
    

# analysis/task info
## analysis/task info
if con_T.__len__() == 3:
    con_Tname = 'T_all'
else:
    con_Tname = con_T[0]

task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
print(task_info)

Ffname_data=op.join(data_path,'group_phase2', task_info +"_data_group_Cat_subF_control" + '.pickle')
Pfname_data=op.join(data_path,'group_phase2', task_info +"_data_group_Cat_subP_control" + '.pickle')


Ffr=open(Ffname_data,'rb')
Fgroup_data=pickle.load(Ffr)



Pfr=open(Pfname_data,'rb')
Pgroup_data=pickle.load(Pfr)



# Color parameters:
cmap = "RdYlBu_r"
#color_blind_palette = sns.color_palette("colorblind")
colors = {
    "F1": [
        0.00392156862745098,
        0.45098039215686275,
        0.6980392156862745
    ],
    "F2": [
        0.00784313725490196,
        0.6196078431372549,
        0.45098039215686275
    ],    
    "F3": [
        0.8352941176470589,
        0.3686274509803922,
        0.0
        ],
    "Middle frontal gyrus": [
        0/255,
        0/255,
        130/255
        ],
    "Inferior frontal sulcus": [
        0/255,
        0/255,
        235/255
        ],
    "Superios frontal sulcus": [
        0/255,
        18/255,
        255/255
        ],
    "Intraparietal sulcus & transverse parietal sulci": [
        128/255,
        0/255,
        0/255
    ],
    "Post-central sulcus": [
        255/255,
        149/255,
        0/255
    ],    
    "Post-central gyrus": [
        0/255,
        114/255,
        235/255
        ],
    "Central sulcus": [
        125/255,
        236/255,
        104/255
        ],
    "Central gyrus": [
        0/255,
        120/255,
        255/255
        ],
    'G_and_S_cingul-Ant': [
        0/255,
        0/255,
        203/255
    ],
    'G_and_S_cingul-Mid-Ant': [
        0/255,
        219/255,
        255/255
    ],    
    'G_and_S_cingul-Mid-Post': [
        133/255,
        255/255,
        143/255
        ],
    "Precentral infrior sulcus": [
        210/255,
        255/255,
        51/255
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



def df_plot(ts_df,T1,pval1,T2,pval2,time_point,test_win_on,roi_name,task_index,chance_index,y_index,fname_fig):
    window=[0.3,0.5,0.5,0.3]
    # if roi_name=='GNW':
    #    window=[0.3,0.5,0.5,0.3]
    # elif roi_name=='IIT':
    #     window=[0.3,1.5,1.5,0.3]    
    # elif roi_name=='MT':
    #     window=[0.25,0.5,0.5,0.25]    
    # elif roi_name=='FP':
    #     window=[0.3,1.5,1.5,0.3]    
    # #plot with sns
    
    
    talk_rc={'lines.linewidth':2,'lines.markersize':4}
    sns.set_context('talk',rc=talk_rc,font_scale=1)
    
    
    g = sns.relplot(x="Times(s)", y="Accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors)
    leg = g._legend
    leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.axvline(1, color='gray', linestyle='--')
    plt.axvline(1.5, color='gray', linestyle='--')
    
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
            plt.plot(time_point[clusters1], np.zeros(clusters1.shape) + chance_index-4, 'o', linewidth=3,color=colors[task_index[0]])
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
            plt.plot(time_point[clusters2], np.zeros(clusters2.shape) + chance_index-6, 'o', linewidth=3,color=colors[task_index[1]])
            sig2[clusters2]=1
            
    plt.fill(window,[chance_index-10,chance_index-10,chance_index+y_index,chance_index+y_index],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([chance_index-10,chance_index+y_index])
    
    g.savefig(fname_fig)
    
    return sig1, sig2

def df_plot_cluster(ts_df,C1_stat,C2_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    # if roi_name=='GNW':
    #    window=[0.3,0.5,0.5,0.3]
    # elif roi_name=='IIT':
    #     window=[0.3,1.5,1.5,0.3]
    # elif roi_name=='MT':
    #     window=[0.25,0.5,0.5,0.25]
    # elif roi_name=='FP':
    #     window=[0.3,1.5,1.5,0.3]
    window=[0.3,0.5,0.5,0.3]
    
    #plot with sns
    
    talk_rc={'lines.linewidth':2,'lines.markersize':4}
    sns.set_context('talk',rc=talk_rc,font_scale=1)
    
    
    g = sns.relplot(x="Times(s)", y="Accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors)
    leg = g._legend
    leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.axvline(1, color='gray', linestyle='--')
    plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-4, 'o', linewidth=3,color=colors[task_index[0]])
                sig1[clusters1]=i
            
    temp2=C2_stat['cluster']
    temp_p2=C2_stat['cluster_p']
    sig2=np.full(time_point.shape,np.nan)
    if len(temp2)>=1:
        for i in range(len(temp2)):
            if temp_p2[i]<0.05:# plot the cluster which  p < 0.05
                clusters2=temp2[i][0]
                plt.plot(time_index[clusters2], np.zeros(clusters2.shape) + chance_index-6, 'o', linewidth=3,color=colors[task_index[1]])
                sig2[clusters2]=i
    
    
            
    plt.fill(window,[chance_index-10,chance_index-10,chance_index+y_index,chance_index+y_index],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([chance_index-10,chance_index+y_index])
    
    g.savefig(fname_fig)
    
    return sig1, sig2

def df_plot_cluster_ori(ts_df,C1_stat,time_point,test_win_on,test_win_off,roi_name,task_index,chance_index,y_index,fname_fig):
    # if roi_name=='GNW':
    #    window=[0.3,0.5,0.5,0.3]
    # elif roi_name=='IIT':
    #     window=[0.3,1.5,1.5,0.3]
    # elif roi_name=='MT':
    #     window=[0.25,0.5,0.5,0.25]
    # elif roi_name=='FP':
    #     window=[0.3,1.5,1.5,0.3]
    window=[0.3,0.5,0.5,0.3]
    
    #plot with sns
    
    talk_rc={'lines.linewidth':2,'lines.markersize':4}
    sns.set_context('talk',rc=talk_rc,font_scale=1)
    
    
    g = sns.relplot(x="Times(s)", y="Accuracy(%)", kind="line", data=ts_df,hue='Task',aspect=2,palette=colors)
    leg = g._legend
    leg.set_bbox_to_anchor([0.72,0.8])
    
    plt.axhline(chance_index, color='k', linestyle='-', label='chance')
    plt.axvline(0, color='k', linestyle='-', label='onset')
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.axvline(1, color='gray', linestyle='--')
    plt.axvline(1.5, color='gray', linestyle='--')
    
    
    temp=C1_stat['cluster']
    temp_p=C1_stat['cluster_p']
    sig1=np.full(time_point.shape,np.nan)
    time_index=time_point[(test_win_on-30):(test_win_off-30)]
    if len(temp)>=1:
        for i in range(len(temp)):
            if temp_p[i]<0.05:# plot the cluster which  p < 0.05
                clusters1=temp[i][0]
                plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-4, 'o', linewidth=3,color=colors[task_index[0]])
                sig1[clusters1]=i
               
    
            
    plt.fill(window,[chance_index-10,chance_index-10,chance_index+y_index,chance_index+y_index],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([chance_index-10,chance_index+y_index])
    
    g.savefig(fname_fig)
    
    return sig1

def df_plot_ROI12_cluster(ts_df,time_point,test_win_on,test_win_off,task_index,chance_index,y_index,fname_fig):
    
    #window=[0.3,0.5,0.5,0.3]
    
    
    #plot with sns
    
    talk_rc={'lines.linewidth':2,'lines.markersize':4}
    sns.set_context('talk',rc=talk_rc,font_scale=2)
    
    
    
    
    g = sns.relplot(x="Times(s)", y="Accuracy(%)", kind="line", data=ts_df,col='ROI',hue='ROI',aspect=4,palette=colors,col_wrap=5,legend=False)
    g.map(plt.axhline, y=50, color='k', linestyle='-', label='chance')
    g.map(plt.axvline, x=0, color='k', linestyle='-', label='onset')
    g.map(plt.axvline, x=0.5, color='gray', linestyle='--')
    
  
 
    
    g.fig.set_size_inches(mm2inch(fig_size[0])*5,mm2inch(fig_size[0])*2)
    #leg = g._legend
    #leg.set_bbox_to_anchor([0.72,0.8])
    
   
    
    #g.map( plt.axvline(1, color='gray', linestyle='--'))
    #g.map(plt.axvline(1.5, color='gray', linestyle='--'))
    
    
    # temp=C1_stat['cluster']
    # temp_p=C1_stat['cluster_p']
    # sig1=np.full(time_point.shape,np.nan)
    # time_index=time_point[(test_win_on-30):(test_win_off-30)]
    # if len(temp)>=1:
    #     for i in range(len(temp)):
    #         if temp_p[i]<0.05:# plot the cluster which  p < 0.05
    #             clusters1=temp[i][0]
    #             plt.plot(time_index[clusters1], np.zeros(clusters1.shape) + chance_index-4, 'o', linewidth=3,color=colors[task_index[0]])
    #             sig1[clusters1]=i
    
    
            
    #plt.fill(window,[chance_index-10,chance_index-10,chance_index+y_index,chance_index+y_index],facecolor='g',alpha=0.2)
    plt.xlim([-0.2,2])
    plt.ylim([chance_index-10,chance_index+y_index])
    plt.xticks([0,0.5,1.0,1.5,2])
    plt.yticks([20,40,60,80,100])

    
    g.savefig(fname_fig,format="svg", transparent=True, dpi=300)
    
    

def df2csv(np_data,task_index,csv_fname):
    columns_index=['Time',
                   'ACC (' + task_index[0] + ')','ACC (' + task_index[1] + ')',
                   'CI (' + task_index[0] + ')','CI (' + task_index[1] + ')',
                   'sig (' + task_index[0] + ')','sig (' + task_index[1] + ')']
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
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='Times(s)', value_name='Accuracy(%)', value_vars=time_point)
    
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
    
    ts_df = pd.melt(df, id_vars=['SUBID','Task'], var_name='Times(s)', value_name='Accuracy(%)', value_vars=time_point)
    
    return ts_df,C1_stat,C2_stat




def stat_cluster_1sample_roi(ROI1_data,ROI2_data,ROI3_data,test_win_on,test_win_off,ROI_name):
    
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
    
    
    df3 = pd.DataFrame(ROI2_data[:,30:251], columns=time_point)
    df3.insert(loc=0, column='SUBID', value=sub_list)
    df2.insert(loc=0, column='ROI',value=ROI_name[1])
    
    
    ts_df = pd.melt(df, id_vars=['SUBID','ROI'], var_name='Times(s)', value_name='Accuracy(%)', value_vars=time_point)
    
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


def wcd_plt(group_data,roi_name='GNW',test_win_on=50, test_win_off=200,chance_index=50,y_index=15):


    time_point = np.array(range(-200,2001, 10))/1000
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI_ccd_g=dat2g(group_data,roi_name,cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')
    
    
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
    ts_df_cluster,C1_stat,C2_stat=stat_cluster_1sample(ROI_ccd_g,test_win_on,test_win_off,task_index=task_index,chance_index=chance_index)
    
    fname_cluster_fig= op.join(stat_figure_root, roi_name + '_'+str(test_win_on) + '_' + str(test_win_off)+"_acc_WCD_cluster" + '.svg')
    
    #plot
    sig1_cluster,sig2_cluster=df_plot_cluster(ts_df_cluster,C1_stat,C2_stat,time_point,
                                              test_win_on,test_win_off,
                                              roi_name,task_index=task_index,
                                              chance_index=chance_index,y_index=y_index,
                                              fname_fig=fname_cluster_fig)



def ROI10_wcd_plt(group_data1,group_data2,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=50,y_index=40):
    ROI_name1=['P1','P2','P3','P4','P5','F1','F2','F3','F4','F5']# ,'P6','F6'
    ROI_name2=['Intraparietal sulcus & transverse parietal sulci',
               'Post-central sulcus',
               'Post-central gyrus',
               'Central sulcus',
               'Central gyrus',
               #'Precentral infrior sulcus',
               'G_and_S_cingul-Ant',
               'G_and_S_cingul-Mid-Ant',
               'G_and_S_cingul-Mid-Post',
               'Middle frontal gyrus',
               'Inferior frontal sulcus']
               #'Superios frontal sulcus']
    task_index=['Irrelevant','Relevant non-target']
    #get decoding data
    ROI9_data=np.zeros([10,2,len(sub_list),251])
    for i in range(5):
        print('i=',i)
        ROI9_data[i]=dat2g(group_data1,ROI_name1[i],cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')   
   
    for i2 in range(5):
        print('i2=',i2+5)
        ROI9_data[i2+5]=dat2g(group_data2,ROI_name1[i2+5],cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')    
   
    time_point = np.array(range(-200,2001, 10))/1000
    
    IR_df1 = pd.DataFrame(ROI9_data[0,0,:,30:251], columns=time_point)
    IR_df1.insert(loc=0, column='SUBID', value=sub_list)
    IR_df1.insert(loc=0, column='ROI',value=ROI_name2[0])
    
    IR_df1 = pd.melt(IR_df1, id_vars=['SUBID','ROI'], var_name='Times(s)', value_name='Accuracy(%)', value_vars=time_point)
        

    for i in range(9):
        n=i+1
        IR_df2 = []
        IR_df2 = pd.DataFrame(ROI9_data[n,0,:,30:251], columns=time_point)
        IR_df2.insert(loc=0, column='SUBID', value=sub_list)
        IR_df2.insert(loc=0, column='ROI',value=ROI_name2[n])
        IR_df1=IR_df1.append(IR_df2)
        
    RE_df1 = pd.DataFrame(ROI9_data[0,1,:,30:251], columns=time_point)
    RE_df1.insert(loc=0, column='SUBID', value=sub_list)
    RE_df1.insert(loc=0, column='ROI',value=ROI_name2[0])
        

    for i in range(9):
        n=i+1
        RE_df2 = []
        RE_df2 = pd.DataFrame(ROI9_data[n,1,:,30:251], columns=time_point)
        RE_df2.insert(loc=0, column='SUBID', value=sub_list)
        RE_df2.insert(loc=0, column='ROI',value=ROI_name2[n])
        RE_df1=RE_df1.append(RE_df2)
        
    RE_df1 = pd.melt(RE_df1, id_vars=['SUBID','ROI'], var_name='Times(s)', value_name='Accuracy(%)', value_vars=time_point)
    
   
    #cluster based methods
    
    
    
    # fname_cluster_fig= op.join(stat_figure_root, task_index[0] + 
    #                             '_'+str(test_win_on) + '_' + str(test_win_off) +
    #                             '_ROI9_acc_'+decoding_method + '_cluster.png')
    
    # #plot
    
    # df_plot_ROI9_cluster(IR_df1,time_point,
    #                       test_win_on,test_win_off,
    #                       task_index=task_index[0],
    #                       chance_index=chance_index,
    #                       y_index=y_index,
    #                       fname_fig=fname_cluster_fig)
    
    #stat
    
    
    fname_cluster_fig2= op.join(stat_figure_root, task_index[1] + 
                                '_'+str(test_win_on) + '_' + str(test_win_off) +
                                '_ROI10_acc_'+decoding_method + '_cluster.svg')
    
    #plot
    df_plot_ROI12_cluster(RE_df1,time_point,
                          test_win_on,test_win_off,
                          task_index=task_index[1],
                          chance_index=chance_index,
                          y_index=y_index,
                          fname_fig=fname_cluster_fig2)
    
    return IR_df1,RE_df1

IR_df1,RE_df1=ROI10_wcd_plt(Pgroup_data,Fgroup_data,decoding_method ='wcd', test_win_on=50, test_win_off=200,chance_index=50,y_index=30) 





# #ccd_plt(group_data,roi_name='MT',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F1',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F2',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F3',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F4',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F5',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)
# wcd_plt(Fgroup_data,roi_name='F6',test_win_on=50, test_win_off=200,chance_index=50,y_index=40)

FROI6_g = np.zeros([6, 2, len(sub_list), 251])
FROI_list=['F1','F2','F3','F4','F5','F6']
for ri,rname in enumerate(FROI_list):
    FROI6_g[ri] = dat2g(Fgroup_data,roi_name=rname,cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')
    
FROI6_g_1=np.mean(np.mean(FROI6_g[:,1,:,50:75],axis=2),axis=1)
FROI6_g_2=np.mean(np.mean(FROI6_g[:,1,:,75:100],axis=2),axis=1)

Froi_g=FROI6_g_2
#Froi_g[0:2]=ROI6_g_1[3:5]

#subjects_dir =r'Y:\HPC\fs'


labels_parc_fs = mne.read_labels_from_annot(subject='fsaverage', parc='aparc.a2009s', subjects_dir=subjects_dir)

F1_ts_list=['G_and_S_cingul-Ant']
F2_ts_list=['G_and_S_cingul-Mid-Ant']
F3_ts_list=['G_and_S_cingul-Mid-Post']
F4_ts_list=['G_front_middle']
F5_ts_list=['S_front_inf']
F6_ts_list=['S_front_sup']

F1_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F1_ts_list:
        F1_ts_index.append(ii)
F2_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F2_ts_list:
        F2_ts_index.append(ii)
F3_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F3_ts_list:
        F3_ts_index.append(ii)
F4_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F4_ts_list:
        F4_ts_index.append(ii)
        
F5_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F5_ts_list:
        F5_ts_index.append(ii)
F6_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in F6_ts_list:
        F6_ts_index.append(ii)


for ni, n_label in enumerate(F1_ts_index):
    F1_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF1_label = F1_label
    elif ni == 1:
        lF1_label = F1_label
    elif ni % 2 == 0:
        rF1_label = rF1_label + F1_label  # , hemi="both"
    else:
        lF1_label = lF1_label + F1_label

for ni, n_label in enumerate(F2_ts_index):
    F2_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF2_label = F2_label
    elif ni == 1:
        lF2_label = F2_label
    elif ni % 2 == 0:
        rF2_label = rF2_label + F2_label  # , hemi="both"
    else:
        lF2_label = lF2_label + F2_label
        
for ni, n_label in enumerate(F3_ts_index):
    F3_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF3_label = F3_label
    elif ni == 1:
        lF3_label = F3_label
    elif ni % 2 == 0:
        rF3_label = rF3_label + F3_label  # , hemi="both"
    else:
        lF3_label = lF3_label + F3_label
        
for ni, n_label in enumerate(F4_ts_index):
    F4_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF4_label = F4_label
    elif ni == 1:
        lF4_label = F4_label
    elif ni % 2 == 0:
        rF4_label = rF4_label + F4_label  # , hemi="both"
    else:
        lF4_label = lF4_label + F4_label
        
for ni, n_label in enumerate(F5_ts_index):
    F5_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF5_label = F5_label
    elif ni == 1:
        lF5_label = F5_label
    elif ni % 2 == 0:
        rF5_label = rF5_label + F5_label  # , hemi="both"
    else:
        lF5_label = lF5_label + F5_label
        
for ni, n_label in enumerate(F6_ts_index):
    F6_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rF6_label = F6_label
    elif ni == 1:
        lF6_label = F6_label
    elif ni % 2 == 0:
        rF6_label = rF6_label + F6_label  # , hemi="both"
    else:
        lF6_label = lF6_label + F6_label



PROI6_g = np.zeros([6, 2, len(sub_list), 251])
PROI_list=['P1','P2','P3','P4','P5','P6']
for ri,rname in enumerate(PROI_list):
    PROI6_g[ri] = dat2g(Pgroup_data,roi_name=rname,cond_name=['Irrelevant','Relevant non-target'],decoding_name='wcd_acc')
    
PROI6_g_1=np.mean(np.mean(PROI6_g[:,1,:,50:75],axis=2),axis=1)
PROI6_g_2=np.mean(np.mean(PROI6_g[:,1,:,75:100],axis=2),axis=1)

Proi_g=PROI6_g_2
#Proi_g[0:2]=ROI6_g_1[3:5]

FProi_g=np.zeros([12,])
FProi_g[:6]=FROI6_g_2
FProi_g[6:]=PROI6_g_2
#subjects_dir =r'Y:\HPC\fs'


labels_parc_fs = mne.read_labels_from_annot(subject='fsaverage', parc='aparc.a2009s', subjects_dir=subjects_dir)

P1_ts_list=['S_intrapariet_and_P_trans']
P2_ts_list=['S_postcentral']
P3_ts_list=['G_postcentral']
P4_ts_list=['S_central']
P5_ts_list=['G_precentral']
P6_ts_list=['S_precentral-inf-part']

P1_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P1_ts_list:
        P1_ts_index.append(ii)
P2_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P2_ts_list:
        P2_ts_index.append(ii)
P3_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P3_ts_list:
        P3_ts_index.append(ii)
P4_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P4_ts_list:
        P4_ts_index.append(ii)
        
P5_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P5_ts_list:
        P5_ts_index.append(ii)
P6_ts_index = []
for ii in range(len(labels_parc_fs)):
    label_name = []
    label_name = labels_parc_fs[ii].name
    if label_name[:-3] in P6_ts_list:
        P6_ts_index.append(ii)


for ni, n_label in enumerate(P1_ts_index):
    P1_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP1_label = P1_label
    elif ni == 1:
        lP1_label = P1_label
    elif ni % 2 == 0:
        rP1_label = rP1_label + P1_label  # , hemi="both"
    else:
        lP1_label = lP1_label + P1_label

for ni, n_label in enumerate(P2_ts_index):
    P2_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP2_label = P2_label
    elif ni == 1:
        lP2_label = P2_label
    elif ni % 2 == 0:
        rP2_label = rP2_label + P2_label  # , hemi="both"
    else:
        lP2_label = lP2_label + P2_label
        
for ni, n_label in enumerate(P3_ts_index):
    P3_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP3_label = P3_label
    elif ni == 1:
        lP3_label = P3_label
    elif ni % 2 == 0:
        rP3_label = rP3_label + P3_label  # , hemi="both"
    else:
        lP3_label = lP3_label + P3_label
        
for ni, n_label in enumerate(P4_ts_index):
    P4_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP4_label = P4_label
    elif ni == 1:
        lP4_label = P4_label
    elif ni % 2 == 0:
        rP4_label = rP4_label + P4_label  # , hemi="both"
    else:
        lP4_label = lP4_label + P4_label
        
for ni, n_label in enumerate(P5_ts_index):
    P5_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP5_label = P5_label
    elif ni == 1:
        lP5_label = P5_label
    elif ni % 2 == 0:
        rP5_label = rP5_label + P5_label  # , hemi="both"
    else:
        lP5_label = lP5_label + P5_label
        
for ni, n_label in enumerate(P6_ts_index):
    P6_label = [label for label in labels_parc_fs if label.name == labels_parc_fs[n_label].name][0]
    if ni == 0:
        rP6_label = P6_label
    elif ni == 1:
        lP6_label = P6_label
    elif ni % 2 == 0:
        rP6_label = rP6_label + P6_label  # , hemi="both"
    else:
        lP6_label = lP6_label + P6_label



#250-500ms
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', 'rh', 'pial', subjects_dir=subjects_dir,
                      background='white', size=(800, 800), alpha=1)
# FROI_list=['F1_ts_index','F2_ts_index','F3_ts_index','F4_ts_index','F5_ts_index','F6_ts_index']

for ni, n_label in enumerate(F4_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F4_label, color=cmap(norm(FROI6_g_2[3])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F4_label, color=cmap(norm(FROI6_g_2[3])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(F5_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F5_label, color=cmap(norm(FROI6_g_2[4])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F5_label, color=cmap(norm(FROI6_g_2[4])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"




for ni, n_label in enumerate(F6_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F6_label, color=cmap(norm(FROI6_g_2[5])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F6_label, color=cmap(norm(FROI6_g_2[5])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"

for ni, n_label in enumerate(P1_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(P2_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(P3_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(P4_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P4_label, color=cmap(norm(PROI6_g_2[3])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P4_label, color=cmap(norm(PROI6_g_2[3])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(P5_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P5_label, color=cmap(norm(PROI6_g_2[4])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P5_label, color=cmap(norm(PROI6_g_2[4])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"




for ni, n_label in enumerate(P6_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P6_label, color=cmap(norm(PROI6_g_2[5])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P6_label, color=cmap(norm(PROI6_g_2[5])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"

views=['lateral','lateral']
for view in views:
    brain.show_view(view)
    
pial_brain=brain.screenshot()
    


# before/after results
# fig = plt.figure(figsize=(4, 12))
# axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
# for ax, image, title in zip(
#     axes, [screenshot, cropped_screenshot], ["Before", "After"]
# ):
#     ax.imshow(image)
#     ax.set_title("{} cropping".format(title))

fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[0])])
ax.imshow(pial_brain)
ax.axis("off")
file_path=stat_figure_root
fnamec='pial_brain.svg'
fname_fig_c=op.join(file_path,fnamec)

fig.savefig(fname_fig_c,format="svg", transparent=True, dpi=300)


#250-500ms
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', 'rh', 'inflated', subjects_dir=subjects_dir,
                      background='white', size=(800, 800), alpha=1)
# FROI_list=['F1_ts_index','F2_ts_index','F3_ts_index','F4_ts_index','F5_ts_index','F6_ts_index']
for ni, n_label in enumerate(F1_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(F2_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(F3_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"






for ni, n_label in enumerate(F4_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F4_label, color=cmap(norm(FROI6_g_2[3])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F4_label, color=cmap(norm(FROI6_g_2[3])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(F5_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F5_label, color=cmap(norm(FROI6_g_2[4])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F5_label, color=cmap(norm(FROI6_g_2[4])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"


for ni, n_label in enumerate(F6_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F6_label, color=cmap(norm(FROI6_g_2[5])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F6_label, color=cmap(norm(FROI6_g_2[5])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"

for ni, n_label in enumerate(P1_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P1_label, color=cmap(norm(PROI6_g_2[0])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(P2_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P2_label, color=cmap(norm(PROI6_g_2[1])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(P3_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P3_label, color=cmap(norm(PROI6_g_2[2])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(P4_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P4_label, color=cmap(norm(PROI6_g_2[3])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P4_label, color=cmap(norm(PROI6_g_2[3])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(P5_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P5_label, color=cmap(norm(PROI6_g_2[4])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P5_label, color=cmap(norm(PROI6_g_2[4])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"




for ni, n_label in enumerate(P6_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(P6_label, color=cmap(norm(PROI6_g_2[5])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(P6_label, color=cmap(norm(PROI6_g_2[5])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"

views=['lateral','lateral']
for view in views:
    brain.show_view(view)
    
inflated_brain=brain.screenshot()
    

fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[0])])
ax.imshow(inflated_brain)
ax.axis("off")
file_path=stat_figure_root
fnamec='inflated_brain.svg'
fname_fig_c=op.join(file_path,fnamec)

fig.savefig(fname_fig_c,format="svg", transparent=True, dpi=300)


#plot colorbar
fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[1])])
ax.axis("off")
cax = fig.add_axes([0.5, 0.1, 0.05, 0.8])
norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
fig.colorbar(ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap)), cax=cax,label="Decoding Accuracy (%)")
file_path=stat_figure_root
fnamec='decoding_plot_colorbar.svg'
fname_fig_c=op.join(file_path,fnamec)

fig.savefig(fname_fig_c,format="svg", transparent=True, dpi=300)



#250-500ms
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', 'rh', 'inflated', subjects_dir=subjects_dir,
                      background='white', size=(800, 800), alpha=1)
# FROI_list=['F1_ts_index','F2_ts_index','F3_ts_index','F4_ts_index','F5_ts_index','F6_ts_index']
for ni, n_label in enumerate(F1_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F1_label, color=cmap(norm(FROI6_g_2[0])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F1_label, color=cmap(norm(FROI6_g_2[0])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
for ni, n_label in enumerate(F2_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F2_label, color=cmap(norm(FROI6_g_2[1])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F2_label, color=cmap(norm(FROI6_g_2[1])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
for ni, n_label in enumerate(F3_ts_index):
    cmap = 'jet'
    cmap = cm.get_cmap(cmap)
    #norm = Normalize(vmin=0, vmax=6)
    norm = Normalize(vmin=min(FProi_g), vmax=max(FProi_g))
    if (ni % 2) == 0:
        brain.add_label(F3_label, color=cmap(norm(FROI6_g_2[2])), alpha=1, hemi="rh",
                        borders=False)  # , hemi="both"
    else:
        brain.add_label(F3_label, color=cmap(norm(FROI6_g_2[2])), alpha=1, hemi="lh",
                        borders=False)  # , hemi="both"
        
views=['medial','medial']
for view in views:
    brain.show_view(view)
    
inflated_brain=brain.screenshot()

fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),mm2inch(fig_size[0])])
ax.imshow(inflated_brain)
ax.axis("off")
file_path=stat_figure_root
fnamec='inflated_brain_medial.svg'
fname_fig_c=op.join(file_path,fnamec)

fig.savefig(fname_fig_c,format="svg", transparent=True, dpi=300)
