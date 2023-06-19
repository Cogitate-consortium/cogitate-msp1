# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:43:32 2022

@author: Ling Liu  ling.liu@pku.edu.cn
=================================
Functions for MEG decoding
=================================
"""

import os
import os.path as op


import mne
import numpy as np



from config import l_freq, h_freq, sfreq

from rsa_helper_functions import equate_offset
from mne.minimum_norm import apply_inverse_epochs



# set the path for decoding analysis
def set_path_ROI_MVPA(bids_root,subject_id, visit_id, analysis_name):
    ### I   Set subject information
    # sub and visit info
    sub_info = 'sub-' + subject_id + '_ses-' + visit_id
    print(sub_info)

    ### II  Set the Input Data Path
    # 1 Set path to the data root path
    fpath_root = op.join(bids_root, "derivatives") #data_path

    # 2 Set path to preprocessed sensor (xxx_epo.fif)
    fpath_epo = op.join(fpath_root, "preprocessing",
                        f"sub-{subject_id}", f"ses-{visit_id}", "meg")

    # 2 Set path to the preprocessed source model data
    fpath_fw = op.join(fpath_root,'forward', f"sub-{subject_id}", "ses-" + visit_id, "meg")

    # 3 Set path to the freesufer subjects_dir for source analysis
    fpath_fs=op.join(fpath_root, "fs")
    # subjects_dir = r'/home/user/S10/Cogitate/HPC/fs'


    ### III  Set the Output Data Path
    # Set path to decoding derivatives
    mvpa_deriv_root = op.join(fpath_root, "decoding")
    if not op.exists(mvpa_deriv_root):
        os.makedirs(mvpa_deriv_root)
        
    
    # Set path to the ROI MVPA output(1) data, 2) figures, 3) codes)
    roi_deriv_root = op.join(mvpa_deriv_root, "roi_mvpa", analysis_name)
    if not op.exists(roi_deriv_root):
        os.makedirs(roi_deriv_root)
    # 1) output_data
    roi_data_root = op.join(roi_deriv_root,
                            f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                            "data")
    if not op.exists(roi_data_root):
        os.makedirs(roi_data_root)

    # 2) output_figure
    roi_figure_root = op.join(roi_deriv_root,
                              f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                              "figures")
    if not op.exists(roi_figure_root):
        os.makedirs(roi_figure_root)

    # 3) output_code
    roi_code_root = op.join(roi_deriv_root,
                            f"sub-{subject_id}", f"ses-{visit_id}", "meg",
                            "codes")
    if not op.exists(roi_code_root):
        os.makedirs(roi_code_root)

    return sub_info,fpath_epo,fpath_fw,fpath_fs, roi_data_root,roi_figure_root, roi_code_root

# functions for use both spatial and temporal feature as the decoding feature
def STdata(Xraw):
    #spatial + temporal decoding
    # temporal feature window
    #Xraw=epochs_cd.get_data()
    twd=5  # how many time points will used as temporal feature
    Xtemp=[];
    for twd_index in range(twd):
        if twd_index==0:
            #Xtemp1=np.append(Xraw[:,:,:1],Xraw[:,:,:-1],axis=2)
            Xtemp = Xraw
        else:
            Xtemp1=np.append(Xraw[:,:,:twd_index],Xraw[:,:,:-twd_index],axis=2)
            Xtemp=np.append(Xtemp,Xtemp1,axis=1)
    
    return Xtemp

# sliding windows (twd,) for MEG data
def ATdata(Xraw):

    #Xraw=epochs_cd.get_data()
    twd=5  # how many time points will be used as sliding windows
    [t1,t2,t3]=Xraw.shape
    Xtemp=np.zeros([5,t1,t2,t3]);
    for twd_index in range(twd):
        if twd_index==0:
            #Xtemp1=np.append(Xraw[:,:,:1],Xraw[:,:,:-1],axis=2)
            #Xtemp = np.expand_dims(Xraw, axis=0)
            Xtemp[twd_index,:,:,:] = Xraw
        else:
            Xtemp1=np.append(Xraw[:,:,:twd_index],Xraw[:,:,:-twd_index],axis=2)
            Xtemp[twd_index,:,:,:] = Xtemp1
    
    Xnew=np.mean(Xtemp,axis=0)
    
    return Xnew


def sensor_data_for_ROI_MVPA(fpath_epo,sub_info,con_T,con_C,con_D):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-dur_epo.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')

    ### Choose condition
    # e.g
    # conditions_T=['500ms','1000ms','1500ms']
    # conditions_D = ['Irrelevant', 'Relevant non-target']
    # conditions_C = ['face', 'object'] or conditions_C = ['letter', 'false']


    #1) Select Category
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
    

    epochs_cdc = epochs['Category in {}'.format(conditions_C)]
    del epochs

    #2) Select Duration Time
    conditions_T = con_T
    print(conditions_T)

    epochs_cdd = epochs_cdc['Duration in {}'.format(conditions_T)]
    del epochs_cdc
    #3) Select Task relevance Design

    conditions_D = con_D
    print(conditions_D)

    epochs_cd = epochs_cdd['Task_relevance in {}'.format(conditions_D)]
    del epochs_cdd
    
    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs = epochs_cd.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    epochs_rs.crop(tmin=-0.5, tmax=2,include_tmax=True, verbose=None)
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.0
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)

    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')

    baseline_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=0, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=0, tmax=2, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    ## analysis/task info
    if con_T.__len__() == 3:
        con_Tname = 'T_all'
    elif con_T.__len__() == 2:
        con_Tname = con_T[0]+'_'+con_T[1]
    else:
        con_Tname = con_T[0]

    task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
    print(task_info)

    return epochs_rs, rank, common_cov, conditions_C, conditions_D, conditions_T, task_info

def sensor_data_for_ROI_MVPA_baseline(fpath_epo,sub_info,con_T,con_C,con_D):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-dur_epo.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')

    ### Choose condition
    # e.g
    # conditions_T=['500ms','1000ms','1500ms']
    # conditions_D = ['Irrelevant', 'Relevant non-target']
    # conditions_C = ['face', 'object'] or conditions_C = ['letter', 'false']


    #1) Select Category
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
    

    epochs_cdc = epochs['Category in {}'.format(conditions_C)]
    del epochs

    #2) Select Duration Time
    conditions_T = con_T
    print(conditions_T)

    epochs_cdd = epochs_cdc['Duration in {}'.format(conditions_T)]
    del epochs_cdc
    #3) Select Task relevance Design

    conditions_D = con_D
    print(conditions_D)

    epochs_cd = epochs_cdd['Task_relevance in {}'.format(conditions_D)]
    del epochs_cdd
    
    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs = epochs_cd.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs.filter(l_freq, h_freq, n_jobs=-1)
    
    epochs_rs.crop(tmin=-0.5, tmax=2,include_tmax=True, verbose=None)
    
    # # Baseline correction
    # b_tmin = -.5
    # b_tmax = -.0
    # baseline = (b_tmin, b_tmax)
    # epochs_rs.apply_baseline(baseline=baseline)

    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')

    baseline_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=0, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=0, tmax=2, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    ## analysis/task info
    if con_T.__len__() == 3:
        con_Tname = 'T_all'
    elif con_T.__len__() == 2:
        con_Tname = con_T[0]+'_'+con_T[1]
    else:
        con_Tname = con_T[0]

    task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
    print(task_info)

    return epochs_rs, rank, common_cov, conditions_C, conditions_D, conditions_T, task_info

def sensor_data_for_ROI_MVPA_equal_offset(fpath_epo,sub_info,con_T,con_C,con_D):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-dur_epo.fif')
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')

    ### Choose condition
    # e.g
    # conditions_T=['500ms','1000ms','1500ms']
    # conditions_D = ['Irrelevant', 'Relevant non-target']
    # conditions_C = ['face', 'object'] or conditions_C = ['letter', 'false']


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
        
    epochs_cdc = epochs['Category in {}'.format(conditions_C)]

    #2) Select Duration Time
    conditions_T = con_T
    print(conditions_T)

    epochs_cdd = epochs_cdc['Duration in {}'.format(conditions_T)]

    #3) Select Task relevance Design

    conditions_D = con_D
    print(conditions_D)

    epochs_cd = epochs_cdd['Task_relevance in {}'.format(conditions_D)]

    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs_temp = epochs_cd.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs_temp.filter(l_freq, h_freq, n_jobs=-1)
    
    #equal offset for 1000ms and 1500ms
    equate_offset_dict= {
        "1500ms":{
            "excise_onset": 1.0,
            "excise_offset": 1.5},
        "1000ms":{
          "excise_onset": 1.5,
          "excise_offset": 2}
        }
    
    epochs_rs=equate_offset(epochs_rs_temp, equate_offset_dict)
    
    epochs_rs.crop(tmin=-0.5, tmax=2,include_tmax=True, verbose=None)
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.0
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)
    


    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')

    baseline_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=0, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=0, tmax=2, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    ## analysis/task info
    if con_T.__len__() == 3:
        con_Tname = 'T_all'
    elif con_T.__len__() == 2:
        con_Tname = con_T[0]+'_'+con_T[1]
    else:
        con_Tname = con_T[0]

    task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
    print(task_info)

    return epochs_rs, rank, common_cov, conditions_C, conditions_D, conditions_T, task_info

def sensor_data_for_ROI_MVPA_ID(fpath_epo,sub_info,con_T,con_C,con_D,remove_too_few_trials=True):
    ### Loading the epochs data
    # fname_epo = file_name
    fname_epo=op.join(fpath_epo,sub_info + '_task-dur_epo.fif')
    #fname_epo=op.join(fpath_epo,fname)
    epochs = mne.read_epochs(fname_epo,
                             preload=True,
                             verbose=True).pick('meg')

    ### Choose condition
    # e.g
    # conditions_T=['500ms','1000ms','1500ms']
    # conditions_D = ['Irrelevant', 'Relevant non-target']
    # conditions_C = ['face', 'object'] or conditions_C = ['letter', 'false']


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

    epochs_cdc = epochs['Category in {}'.format(conditions_C)]

    #2) Select Duration Time
    conditions_T = con_T
    print(conditions_T)

    epochs_cdd = epochs_cdc['Duration in {}'.format(conditions_T)]

    #3) Select Task relevance Design

    conditions_D = con_D
    print(conditions_D)

    epochs_cd = epochs_cdd['Task_relevance in {}'.format(conditions_D)]
    
    
    #remove_too_few_trials:
    min_n_repeats=2
    sub_metadata = epochs_cd.metadata.reset_index(drop=True)
    # Find the identity for which we have less than two trials:
    cts = sub_metadata.groupby(["Stim_trigger"])["Stim_trigger"].count()
    id_to_remove = [identity for identity in cts.keys() if cts[identity] < min_n_repeats]
    # Get the indices of the said identity to drop the trials:
    id_idx = sub_metadata.loc[sub_metadata["Stim_trigger"].isin(id_to_remove)].index.values.tolist()
    # Dropping those:
    epochs_cd.drop(id_idx)
        # epochs_cd = remove_too_few_trials(epochs_cd, condition="Stim_trigger", min_n_repeats=2, verbose=False)

    # Downsample and filter to speed the decoding
    # Downsample copy of raw
    epochs_rs_temp = epochs_cd.copy().resample(sfreq, n_jobs=-1)
    # Band-pass filter raw copy
    epochs_rs_temp.filter(l_freq, h_freq, n_jobs=-1)
    
    #equal offset for 1000ms and 1500ms
    equate_offset_dict= {
        "1500ms":{
            "excise_onset": 1.0,
            "excise_offset": 1.5},
        "1000ms":{
          "excise_onset": 1.5,
          "excise_offset": 2}
        }
    
    epochs_rs=equate_offset(epochs_rs_temp, equate_offset_dict)
    
    epochs_rs.crop(tmin=-0.5, tmax=1.5,include_tmax=True, verbose=None)
    
    # Baseline correction
    b_tmin = -.5
    b_tmax = -.0
    baseline = (b_tmin, b_tmax)
    epochs_rs.apply_baseline(baseline=baseline)
    

    

    # projecting sensor-space data to source space   ###TODO:shrunk or ?
    rank = mne.compute_rank(epochs_rs, tol=1e-6, tol_kind='relative')

    baseline_cov = mne.compute_covariance(epochs_rs, tmin=-0.5, tmax=0, method='empirical', rank=rank, n_jobs=-1,
                                          verbose=True)
    active_cov = mne.compute_covariance(epochs_rs, tmin=0, tmax=2, method='empirical', rank=rank, n_jobs=-1,
                                        verbose=True)

    common_cov = baseline_cov + active_cov

    ## analysis/task info
    if con_T.__len__() == 3:
        con_Tname = 'T_all'
    elif con_T.__len__() == 2:
        con_Tname = con_T[0]+'_'+con_T[1]
    else:
        con_Tname = con_T[0]

    task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
    print(task_info)

    return epochs_rs, rank, common_cov, conditions_C, conditions_D, conditions_T, task_info


def source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label):

    # projecting sensor-space data to source space
    # the path of forward solution
    fname_fwd = op.join(fpath_fw, sub_info + "_surface_fwd.fif")

    fwd = mne.read_forward_solution(fname_fwd)
    
    #make inverse operator
    # Make inverse operator
   
    inv = mne.minimum_norm.make_inverse_operator(epochs_rs.info, fwd, common_cov,
                                                 loose=.2,depth=.8,fixed=False,
                                                 rank=rank,use_cps=True)  # cov= baseline + active, compute rank, same as the LCMV
    
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(epochs_rs, inv, 1. / lambda2, 'dSPM', pick_ori="normal", label=surf_label)

    return stcs

def sub_ROI_for_ROI_MVPA(fpath_fs,subject_id,analysis_name):
    
    # prepare the label for extract data
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels_parc_sub = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)
    else:
        labels_parc_sub = mne.read_labels_from_annot(subject=f"sub-{subject_id}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=fpath_fs)

    
    # replace "&" and "_and_" for indisual MRI or fsaverage
    if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        #ROI info, could change ###TODO: the final defined ROI
        GNW_ts_list = ['G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant',
                       'G_and_S_cingul-Mid-Post', 'G_front_middle',
                        'S_front_inf', 'S_front_sup',
                        ]
        
        PFC_ts_list = ['G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant',
                       'G_and_S_cingul-Mid-Post', 'G_front_middle', 'S_front_sup',
                        ] #'S_front_inf' # remove S_front_inf, since this GNW ROI is also in the extented IIT ROI list.
        
        IIT_ts_list = ['G_cuneus',
                       'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Lingual',
                       'Pole_occipital', 'S_calcarine',
                       'S_oc_sup_and_transversal']
        
        MT_ts_list = ['S_central','S_postcentral']
        
        F1_ts_list=['G_and_S_cingul-Ant']
        F2_ts_list=['G_and_S_cingul-Mid-Ant']
        F3_ts_list=['G_and_S_cingul-Mid-Post']
        F4_ts_list=['G_front_middle']
        F5_ts_list=['S_front_inf']
        F6_ts_list=['S_front_sup']
        
        P1_ts_list=['S_intrapariet_and_P_trans']
        P2_ts_list=['S_postcentral']
        P3_ts_list=['G_postcentral']
        P4_ts_list=['S_central']
        P5_ts_list=['G_precentral']
        P6_ts_list=['S_precentral-inf-part']
        
        
    else:
        #ROI info, could change ###TODO: the final defined ROI
        GNW_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                       'G&S_cingul-Mid-Post', 'G_front_middle',
                        'S_front_inf', 'S_front_sup',
                        ]
        
        PFC_ts_list = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant',
                       'G&S_cingul-Mid-Post', 'G_front_middle', 'S_front_sup',
                        ] #'S_front_inf' # remove S_front_inf, since this GNW ROI is also in the extented IIT ROI list.
        
        
        IIT_ts_list = ['G_cuneus',
                       'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Lingual',
                       'Pole_occipital', 'S_calcarine',
                       'S_oc_sup&transversal']
        
        #MT_ts_list = ['S_central','S_postcentral']

        MT_ts_list = ['S_central']
        
        F1_ts_list=['G&S_cingul-Ant']
        F2_ts_list=['G&S_cingul-Mid-Ant']
        F3_ts_list=['G&S_cingul-Mid-Post']
        F4_ts_list=['G_front_middle']
        F5_ts_list=['S_front_inf']
        F6_ts_list=['S_front_sup']
    
        P1_ts_list=['S_intrapariet&P_trans']
        P2_ts_list=['S_postcentral']
        P3_ts_list=['G_postcentral']
        P4_ts_list=['S_central']
        P5_ts_list=['G_precentral']
        P6_ts_list=['S_precentral-inf-part']

    GNW_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in GNW_ts_list:
            GNW_ts_index.append(ii)
            
    PFC_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in PFC_ts_list:
            PFC_ts_index.append(ii)
        
            

    IIT_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in IIT_ts_list:
            IIT_ts_index.append(ii)
            
    MT_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in MT_ts_list:
            MT_ts_index.append(ii)
            
    F1_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F1_ts_list:
            F1_ts_index.append(ii)
    F2_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F2_ts_list:
            F2_ts_index.append(ii)
    F3_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F3_ts_list:
            F3_ts_index.append(ii)
    F4_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F4_ts_list:
            F4_ts_index.append(ii)
            
    F5_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F5_ts_list:
            F5_ts_index.append(ii)
    F6_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in F6_ts_list:
            F6_ts_index.append(ii)
            
            
            
    P1_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P1_ts_list:
            P1_ts_index.append(ii)
    P2_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P2_ts_list:
            P2_ts_index.append(ii)
    P3_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P3_ts_list:
            P3_ts_index.append(ii)
    P4_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P4_ts_list:
            P4_ts_index.append(ii)
            
    P5_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P5_ts_list:
            P5_ts_index.append(ii)
    P6_ts_index = []
    for ii in range(len(labels_parc_sub)):
        label_name = []
        label_name = labels_parc_sub[ii].name
        if label_name[:-3] in P6_ts_list:
            P6_ts_index.append(ii)

    for ni, n_label in enumerate(GNW_ts_index):
        GNW_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rGNW_label = GNW_label
        elif ni == 1:
            lGNW_label = GNW_label
        elif ni % 2 == 0:
            rGNW_label = rGNW_label + GNW_label  # , hemi="both"
        else:
            lGNW_label = lGNW_label + GNW_label
            
    for ni, n_label in enumerate(PFC_ts_index):
        PFC_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rPFC_label = PFC_label
        elif ni == 1:
            lPFC_label = PFC_label
        elif ni % 2 == 0:
            rPFC_label = rPFC_label + PFC_label  # , hemi="both"
        else:
            lPFC_label = lPFC_label + PFC_label
            
    for ni, n_label in enumerate(IIT_ts_index):
        IIT_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rIIT_label = IIT_label
        elif ni == 1:
            lIIT_label = IIT_label
        elif ni % 2 == 0:
            rIIT_label = rIIT_label + IIT_label  # , hemi="both"
        else:
            lIIT_label = lIIT_label + IIT_label
            
    for ni, n_label in enumerate(MT_ts_index):
        MT_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rMT_label = MT_label
        elif ni == 1:
            lMT_label = MT_label
        elif ni % 2 == 0:
            rMT_label = rMT_label + MT_label  # , hemi="both"
        else:
            lMT_label = lMT_label + MT_label
            
    for ni, n_label in enumerate(F1_ts_index):
        F1_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF1_label = F1_label
        elif ni == 1:
            lF1_label = F1_label
        elif ni % 2 == 0:
            rF1_label = rF1_label + F1_label  # , hemi="both"
        else:
            lF1_label = lF1_label + F1_label
    
    for ni, n_label in enumerate(F2_ts_index):
        F2_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF2_label = F2_label
        elif ni == 1:
            lF2_label = F2_label
        elif ni % 2 == 0:
            rF2_label = rF2_label + F2_label  # , hemi="both"
        else:
            lF2_label = lF2_label + F2_label
            
    for ni, n_label in enumerate(F3_ts_index):
        F3_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF3_label = F3_label
        elif ni == 1:
            lF3_label = F3_label
        elif ni % 2 == 0:
            rF3_label = rF3_label + F3_label  # , hemi="both"
        else:
            lF3_label = lF3_label + F3_label
            
    for ni, n_label in enumerate(F4_ts_index):
        F4_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF4_label = F4_label
        elif ni == 1:
            lF4_label = F4_label
        elif ni % 2 == 0:
            rF4_label = rF4_label + F4_label  # , hemi="both"
        else:
            lF4_label = lF4_label + F4_label
            
    for ni, n_label in enumerate(F5_ts_index):
        F5_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF5_label = F5_label
        elif ni == 1:
            lF5_label = F5_label
        elif ni % 2 == 0:
            rF5_label = rF5_label + F5_label  # , hemi="both"
        else:
            lF5_label = lF5_label + F5_label
            
    for ni, n_label in enumerate(F6_ts_index):
        F6_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rF6_label = F6_label
        elif ni == 1:
            lF6_label = F6_label
        elif ni % 2 == 0:
            rF6_label = rF6_label + F6_label  # , hemi="both"
        else:
            lF6_label = lF6_label + F6_label
            
    for ni, n_label in enumerate(P1_ts_index):
        P1_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP1_label = P1_label
        elif ni == 1:
            lP1_label = P1_label
        elif ni % 2 == 0:
            rP1_label = rP1_label + P1_label  # , hemi="both"
        else:
            lP1_label = lP1_label + P1_label
    
    for ni, n_label in enumerate(P2_ts_index):
        P2_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP2_label = P2_label
        elif ni == 1:
            lP2_label = P2_label
        elif ni % 2 == 0:
            rP2_label = rP2_label + P2_label  # , hemi="both"
        else:
            lP2_label = lP2_label + P2_label
            
    for ni, n_label in enumerate(P3_ts_index):
        P3_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP3_label = P3_label
        elif ni == 1:
            lP3_label = P3_label
        elif ni % 2 == 0:
            rP3_label = rP3_label + P3_label  # , hemi="both"
        else:
            lP3_label = lP3_label + P3_label
            
    for ni, n_label in enumerate(P4_ts_index):
        P4_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP4_label = P4_label
        elif ni == 1:
            lP4_label = P4_label
        elif ni % 2 == 0:
            rP4_label = rP4_label + P4_label  # , hemi="both"
        else:
            lP4_label = lP4_label + P4_label
            
    for ni, n_label in enumerate(P5_ts_index):
        P5_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP5_label = P5_label
        elif ni == 1:
            lP5_label = P5_label
        elif ni % 2 == 0:
            rP5_label = rP5_label + P5_label  # , hemi="both"
        else:
            lP5_label = lP5_label + P5_label
            
    for ni, n_label in enumerate(P6_ts_index):
        P6_label = [label for label in labels_parc_sub if label.name == labels_parc_sub[n_label].name][0]
        if ni == 0:
            rP6_label = P6_label
        elif ni == 1:
            lP6_label = P6_label
        elif ni % 2 == 0:
            rP6_label = rP6_label + P6_label  # , hemi="both"
        else:
            lP6_label = lP6_label + P6_label



            
    if analysis_name=='Cat' or analysis_name=='Ori' or analysis_name=='Cat_offset_control':
        surf_label_list = [rGNW_label+lGNW_label, rIIT_label+lIIT_label,rGNW_label+lGNW_label+rIIT_label+lIIT_label]
        ROI_Name = ['GNW', 'IIT','FP']
        
    elif analysis_name=='Cat_MT_control':
        surf_label_list = [rMT_label+lMT_label]
        ROI_Name = ['MT']
        
    elif analysis_name=='Cat_subF_control':
        surf_label_list = [rF1_label+lF1_label,rF2_label+lF2_label,rF3_label+lF3_label,
                           rF4_label+lF4_label,rF5_label+lF5_label,rF6_label+lF6_label]
        ROI_Name = ['F1','F2','F3','F4','F5','F6']
        
    elif analysis_name=='Cat_subP_control':
        surf_label_list = [rP1_label+lP1_label,rP2_label+lP2_label,rP3_label+lP3_label,
                           rP4_label+lP4_label,rP5_label+lP5_label,rP6_label+lP6_label]
        ROI_Name = ['P1','P2','P3','P4','P5','P6']
    
    elif analysis_name=='Cat_PFC' or analysis_name=='Ori_PFC':
        surf_label_list = [rPFC_label+lPFC_label, rIIT_label+lIIT_label,rPFC_label+lPFC_label+rIIT_label+lIIT_label]
        ROI_Name = ['PFC', 'IIT','IITPFC']
        
    else:
        surf_label_list = [rGNW_label+lGNW_label, rIIT_label+lIIT_label]
        ROI_Name = ['GNW', 'IIT']

    return surf_label_list, ROI_Name
