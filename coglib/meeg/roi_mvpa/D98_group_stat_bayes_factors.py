# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:16:29 2025

@author: Ling_BLCU
"""

"""
====================
D98. Group Statistic Using Bayes Factors
====================

@author: Ling Liu  ling.liu@pku.edu.cn

"""
import os.path as op
import os
import argparse

import pickle



import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
sns.set_theme(style='ticks')

from scipy import stats as stats


from config import bids_root
from sublist import sub_list

from scipy.stats import halfnorm

from bayes_factor_fun import bayes_ttest, bayes_binomtest, sim_decoding_binomial, sim_decoding_binomial_2d, kendall_bf_from_data

parser = argparse.ArgumentParser()
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--cT', type=str, nargs='*', default=['500ms','1000ms','1500ms'],
                    help='condition in Time duration')
parser.add_argument('--cC', type=str, nargs='*', default=['LF'],
                    help='selected decoding category, FO for face and object, LF for letter and false')
parser.add_argument('--analysis',
                    type=str,
                    default='IITGNW',
                    help='the name for anlaysis, e.g. GAT or RSA or IITPFC/IITGNW or WCD')
parser.add_argument('--decoding',
                    type=str,
                    default='Cat',
                    help='the name for anlaysis, e.g. Cat or Ori or ID')


opt = parser.parse_args()

visit_id = opt.visit
analysis_name=opt.analysis
decoding_class=opt.decoding


opt = parser.parse_args()
con_C = opt.cC
con_T = opt.cT

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
    


time_point = np.array(range(-200,2001, 10))/1000
# set the path for decoding analysis
def set_path_plot(bids_root, visit_id, analysis_name,decoding_class,con_name):

    ### I  Set the group Data Path
    # Set path to decoding derivatives
    decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa','BF')

    data_path=op.join(decoding_path, analysis_name, decoding_class)

    # Set path to group analysis derivatives
    group_deriv_root = op.join(data_path, "group")
    if not op.exists(group_deriv_root):
        os.makedirs(group_deriv_root)
        
    
    # Set path to the ROI MVPA output BF results
   
    # 1) output_stat_data
    BF_data_root = op.join(group_deriv_root,"BF",con_name)
    if not op.exists(BF_data_root):
        os.makedirs(BF_data_root)

   

    return group_deriv_root,BF_data_root



def dat2gat(dat,roi_name,cond_name,decoding_name):
    roi_gat=np.zeros([2,len(sub_list),251,251])
    for ci, cond in enumerate(cond_name):
        roi_gat_gc=np.zeros([len(sub_list),251,251])
        for i, sbn in enumerate(sub_list):
            roi_gat_gc[i,:,:]=dat[sbn][decoding_name][roi_name][cond]
           
        
        roi_gat[ci,:,:,:]=roi_gat_gc*100
        
    return roi_gat 

def dat2g_ori(dat,roi_name,cond_name,decoding_name):
    roi_ccd_g=np.zeros([1,len(sub_list),251])
    roi_ccd_gc=np.zeros([len(sub_list),251])
    for i, sbn in enumerate(sub_list):
        roi_ccd_gc[i,:]=dat[sbn][decoding_name][roi_name][cond_name]
    roi_ccd_g[0,:,:]=roi_ccd_gc*100
        
    return roi_ccd_g


def dat2g_PFC(dat,cond_name):
    roi_wcd_g=np.zeros([3,len(sub_list),251])
    for ci, cond in enumerate(cond_name):
        roi_wcd_gc=np.zeros([len(sub_list),251])
        for i, sbn in enumerate(sub_list):
            roi_wcd_gc[i,:]=dat[sbn][cond]
        roi_wcd_g[ci,:,:]=roi_wcd_gc*100
        
    return roi_wcd_g

def rsa2gat(dat,roi_name,cond_name,decoding_name,decoding_class):
    if decoding_class=='ID':
        time_points=201
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
    if decoding_class=='Ori':
        time_points=251
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
    elif decoding_class=='Cat':
        time_points=251
        roi_rsa_g=np.zeros([len(sub_list),time_points,time_points])
        for ci, cond in enumerate(cond_name):
            roi_rsa_gc=np.zeros([len(sub_list),time_points,time_points])
            for i, sbn in enumerate(sub_list):
                roi_rsa_gc[i,:,:]=dat[sbn][roi_name][cond][roi_name][decoding_name]
        roi_rsa_g[:,:,:]=roi_rsa_gc
        
    return roi_rsa_g   

def Cat_GAT_BF(group_data,roi_name='GNW',test_win_on=50, test_win_off=70):
    #get decoding data
    ROI_ctccd_g=dat2gat(group_data,roi_name,cond_name=['IR2RE','RE2IR'],decoding_name='ctccd_acc')
    
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(ROI_ctccd_g[0,:,test_win_on:test_win_off,test_win_on:test_win_off], y=50, paired=False, alternative='greater', r=0.707, return_pval=True)
    bf_ROI_value=np.mean(np.mean(1/bf))
    return bf_ROI_value


def Ori_WCD_BF(group_data,roi_name='GNW',test_win_on=50, test_win_off=70):
    #get decoding data
    ROI_ori_g=dat2g_ori(group_data,roi_name,cond_name=conditions_C[0],decoding_name='wcd_ori_acc')
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(ROI_ori_g[0,:,test_win_on:test_win_off], y=33.3, paired=False, alternative='greater', r=0.707, return_pval=True)
    bf_ROI_value=np.mean(np.mean(1/bf))
    if con_C[0]=='F':
        bf_ROI_value=np.mean(np.mean(bf))
    return bf_ROI_value

def Cat_IITPFC_BF(group_data,test_win_on=50, test_win_off=170):
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name=['IIT','IITPFC_f','IITPFC_m'])
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(PFC_data[0,:,test_win_on:test_win_off],PFC_data[1,:,test_win_on:test_win_off], paired=True, alternative='two-sided', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(1/bf)
    return bf_ROI_value

def Cat_IITGNW_BF(group_data,test_win_on=50, test_win_off=170):
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name=['IIT','IITGNW_f'])
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(PFC_data[0,:,test_win_on:test_win_off],PFC_data[1,:,test_win_on:test_win_off], paired=True, alternative='two-sided', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(1/bf)
    return bf_ROI_value

def Ori_IITPFC_BF(group_data,test_win_on=50, test_win_off=170):
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name=['IIT','IITPFC_f','IITPFC_m'])
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(PFC_data[0,:,test_win_on:test_win_off],PFC_data[1,:,test_win_on:test_win_off], paired=True, alternative='two-sided', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(1/bf)
    return bf_ROI_value

def Ori_IITGNW_BF(group_data,test_win_on=50, test_win_off=170):
    #get decoding data
    PFC_data=dat2g_PFC(group_data,cond_name=['IIT','IITGNW_f'])
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(PFC_data[0,:,test_win_on:test_win_off],PFC_data[1,:,test_win_on:test_win_off], paired=True, alternative='two-sided', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(1/bf)
    return bf_ROI_value

def Cat_RSA_BF(group_data,roi_name='GNW',condition='Irrelevant'):
    if roi_name=='GNW':
        test_win_on=80
        test_win_off=100
    elif roi_name=='IIT':
        test_win_on=80
        test_win_off=200
        
    time_point = np.array(range(-500,2001, 10))/1000
    if condition=='Irrelevant':
        conD='IR'
    elif condition=='Relevant non-target':
        conD='RE'
    
    condN=['rsa']    
    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=condN,decoding_name=condition,decoding_class=decoding_class)
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(roi_rsa_g[:,test_win_on:test_win_off,test_win_on:test_win_off],  y=0, paired=False, alternative='greater', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(np.mean(1/bf))
    return bf_ROI_value


def Ori_RSA_BF(group_data,roi_name='GNW'):
    if roi_name=='GNW':
        test_win_on=80
        test_win_off=100
    elif roi_name=='IIT':
        test_win_on=80
        test_win_off=200
        
    time_point = np.array(range(-500,2001, 10))/1000
   
    
    condN=['rsa']    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=condN,decoding_name='Ori',decoding_class=decoding_class)
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(roi_rsa_g[:,test_win_on:test_win_off,test_win_on:test_win_off],  y=0, paired=False, alternative='greater', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(np.mean(1/bf))
    return bf_ROI_value


def ID_RSA_BF(group_data,roi_name='GNW'):
    if roi_name=='GNW':
        test_win_on=80
        test_win_off=100
    elif roi_name=='IIT':
        test_win_on=80
        test_win_off=200
        
    time_point = np.array(range(-500, 1501, 10))/1000
   
    
    condN=['rsa']    
    #get decoding data
    roi_rsa_g=rsa2gat(group_data,roi_name,cond_name=condN,decoding_name='ID',decoding_class=decoding_class)
    
    # Compute the Bayes factor:
    bf, pval = bayes_ttest(roi_rsa_g[:,test_win_on:test_win_off,test_win_on:test_win_off],  y=0, paired=False, alternative='greater', r=0.707, return_pval=True)
   
    bf_ROI_value=np.mean(np.mean(1/bf))
    return bf_ROI_value
#########
#set data root    
group_deriv_root,BF_data_root=set_path_plot(bids_root,visit_id, analysis_name,decoding_class,con_C[0])    
    

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

if analysis_name=='GAT':
    data_name=analysis_name+'_'+decoding_class
elif analysis_name=='IITPFC':
    data_name=decoding_class + '_PFC'
elif analysis_name=='WCD':
    data_name=decoding_class
elif analysis_name=='RSA':
    data_name=analysis_name+'_'+decoding_class + '_ED'
elif analysis_name=='IITGNW':
    data_name=decoding_class + '_GNW'
        


fname_data=op.join(group_deriv_root, task_info +"_data_group_" + data_name +
                   '.pickle')

fr=open(fname_data,'rb')
group_data=pickle.load(fr)


if analysis_name=='GAT':
    bf_ROI_value=Cat_GAT_BF(group_data,roi_name='GNW',test_win_on=50, test_win_off=70)
    csv_fname=op.join(BF_data_root, 'BF_value_'+con_Tname + '.csv')
    df=pd.DataFrame({'duration':[con_Tname],'BF_value':[bf_ROI_value]})
    df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')    
elif analysis_name=='IITPFC':
    if decoding_class=='Cat':
        bf_ROI_value=Cat_IITPFC_BF(group_data,test_win_on=50, test_win_off=170)
        csv_fname=op.join(BF_data_root, 'BF_value_Cat_'+con_C[0] + '.csv')
        df=pd.DataFrame({'condition':[con_C[0]],'BF_value':[bf_ROI_value]})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')  
    elif decoding_class=='Ori':
        bf_ROI_value=Ori_IITPFC_BF(group_data,test_win_on=50, test_win_off=170)
        csv_fname=op.join(BF_data_root, 'BF_value_Ori_'+con_C[0] + '.csv')
        df=pd.DataFrame({'condition':[con_C[0]],'BF_value':[bf_ROI_value]})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')  
elif analysis_name=='WCD':
    bf_ROI_value=Ori_WCD_BF(group_data,roi_name='GNW',test_win_on=50, test_win_off=70)
    csv_fname=op.join(BF_data_root, 'BF_value_'+conditions_C[0] + '.csv')
    df=pd.DataFrame({'condition':[conditions_C[0]],'BF_value':[bf_ROI_value]})
    df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')    
elif analysis_name=='RSA':
    if decoding_class=='Cat':
        bf_ROI_value=[]
        condition_list=[]
        for roi_name in ['GNW','IIT']:
            for condition in ['Irrelevant','Relevant non-target']:
                c_name=roi_name+'_'+condition
                bf_value=Cat_RSA_BF(group_data,roi_name=roi_name,condition=condition)
                bf_ROI_value.append(bf_value)
                condition_list.append(c_name)
        csv_fname=op.join(BF_data_root, 'BF_value_'+decoding_class + '.csv')
        df=pd.DataFrame({'condition':condition_list,'BF_value':bf_ROI_value})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')         
    elif decoding_class=='Ori':
        bf_ROI_value=[]
        condition_list=[]
        for roi_name in ['GNW','IIT']:
            c_name=roi_name
            bf_value=Ori_RSA_BF(group_data,roi_name=roi_name)
            bf_ROI_value.append(bf_value)
            condition_list.append(c_name)
        csv_fname=op.join(BF_data_root, 'BF_value_'+decoding_class + '.csv')
        df=pd.DataFrame({'condition':condition_list,'BF_value':bf_ROI_value})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')
    elif decoding_class=='ID':
        bf_ROI_value=[]
        condition_list=[]
        for roi_name in ['GNW','IIT']:
            c_name=roi_name
            bf_value=ID_RSA_BF(group_data,roi_name=roi_name)
            bf_ROI_value.append(bf_value)
            condition_list.append(c_name)
        csv_fname=op.join(BF_data_root, 'BF_value_'+decoding_class + '.csv')
        df=pd.DataFrame({'condition':condition_list,'BF_value':bf_ROI_value})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')
elif analysis_name=='IITGNW':
    if decoding_class=='Cat':
        bf_ROI_value=Cat_IITGNW_BF(group_data,test_win_on=50, test_win_off=170)
        csv_fname=op.join(BF_data_root, 'BF_value_Cat_'+con_C[0] + '.csv')
        df=pd.DataFrame({'condition':[con_C[0]],'BF_value':[bf_ROI_value]})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')  
    elif decoding_class=='Ori':
        bf_ROI_value=Ori_IITGNW_BF(group_data,test_win_on=50, test_win_off=170)
        csv_fname=op.join(BF_data_root, 'BF_value_Ori_'+con_C[0] + '.csv')
        df=pd.DataFrame({'condition':[con_C[0]],'BF_value':[bf_ROI_value]})
        df.to_csv(csv_fname,sep=',',index=False,header=True,na_rep='NaN')  










