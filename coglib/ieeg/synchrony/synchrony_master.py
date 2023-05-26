#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Synchrony analysis
    @author: Simon Henin
    simon.henin@nyulangone.org
"""

# %%
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs
import os, glob
import pandas as pd
from tqdm import tqdm
from scipy import stats

from nice.algorithms.connectivity import epochs_compute_wsmi

from general_helper_functions.data_general_utilities import load_epochs, cluster_test, find_channels_in_roi, \
    moving_average
from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from freesurfer.wang_labels import get_montage_volume_labels_wang

from synchrony.synchrony_analysis_parameters_class import SynchronyAnalysisParameters
from synchrony.synchrony_helper_functions import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from joblib import Parallel, delayed

import xarray as xr
from frites.conn import conn_dfc, define_windows


# %%
def synchrony_analysis(subjects_list, save_folder="super"):
    # %%
    # Extract config from command line argument
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--sim', type=str, default=None,
                        help="run simulation")
    args = parser.parse_args()
    # If no config was passed, just using them all
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "synchrony", "configs"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)

        param = SynchronyAnalysisParameters(config, sub_id=save_folder)

        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "synchrony")
        
        # %%
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            for roi in param.rois:
                save_path_results = path_generator(param.save_root,
                                                   analysis=analysis_name,
                                                   preprocessing_steps=param.preprocess_steps,
                                                   fig=False, stats=True)
                save_path_fig = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=True)

                # load in category selectivity data
                if isinstance( analysis_parameters["category_selectivity_folder"], list):
                    cat_results = []
                    for cat_select_folders in analysis_parameters["category_selectivity_folder"]:
                        cat_select_files = find_files(Path(param.BIDS_root, "derivatives", "category_selectivity",
                                                           "sub-" + param.SUBJ_ID, "ses-" + param.session, "ieeg", "results",
                                                           cat_select_folders,
                                                           param.preprocess_steps),
                                                      naming_pattern="*category_selectivity_all_results",
                                                      extension=".csv")
                        assert len(cat_select_files) == 1, "ERROR: there wasn't a folder for category selectivty results!"
                        cat_results.append( pd.read_csv(cat_select_files[0]) )

                    # # for testing
                    # cat_selectivity_results = cat_results[0]
                    # cat_selectivity_results['cond1'] = cat_results[0].condition
                    # cat_selectivity_results['cond2'] = cat_results[1].condition
                    # cat_selectivity_results['combo'] = cat_selectivity_results.condition[ ( cat_results[0].condition.str.contains('object') & cat_results[1].condition.str.contains('object')) | 
                    #                                                                           ( cat_results[0].condition.str.contains('face') & cat_results[1].condition.str.contains('face')) | 
                    #                                                                           ( cat_results[0].condition.str.contains('false') & cat_results[1].condition.str.contains('false')) | 
                    #                                                                           ( cat_results[0].condition.str.contains('letter') & cat_results[1].condition.str.contains('letter'))]
                    
                    
                    # merge category slective "condition" using AND operation across ti/tr
                    cat_selectivity_results = cat_results[0]
                    cat_selectivity_results['condition'] = cat_selectivity_results.condition[ ( cat_results[0].condition.str.contains('object') & cat_results[1].condition.str.contains('object')) | 
                                                                                              ( cat_results[0].condition.str.contains('face') & cat_results[1].condition.str.contains('face')) | 
                                                                                              ( cat_results[0].condition.str.contains('false') & cat_results[1].condition.str.contains('false')) | 
                                                                                              ( cat_results[0].condition.str.contains('letter') & cat_results[1].condition.str.contains('letter'))]
                        
                        
                else:
                    cat_select_files = find_files(Path(param.BIDS_root, "derivatives", "category_selectivity",
                                                       "sub-" + param.SUBJ_ID, "ses-" + param.session, "ieeg", "results",
                                                       analysis_parameters["category_selectivity_folder"],
                                                       param.preprocess_steps),
                                                  naming_pattern="*category_selectivity_all_results",
                                                  extension=".csv")
    
                    assert len(cat_select_files) == 1, "ERROR: there wasn't a folder for category selectivty results!"
                    cat_selectivity_results = pd.read_csv(cat_select_files[0])

                # %% Analysis is done on a per-subject basis
                # We will concatenate each connectivity matrix together on a per-connection basis
                # Later, we can average per participant
                cat1_to_cat1 = []
                cat1_to_cat2 = []
                cat1_to_cat1_perm = []
                cat1_to_cat2_perm = []
                category_1_selective = []
                cat2_to_cat1 = []
                cat2_to_cat2 = []
                cat2_to_cat1_perm = []
                cat2_to_cat2_perm = []
                category_2_selective = []
                electrode_list = None # init to None, will be initialized once we know what the categories are
                df_conn = pd.DataFrame([], columns=['category', 'trials', 'seed', 'target', 'conn', 'conn_perm'])
                for subject in subjects_list:

                    sub_results = cat_selectivity_results.loc[cat_selectivity_results["subject"] == subject]

                    # % load in the high-gamma signal first, since we will use this to select active electrodes
                    epochs, mni_coord = load_epochs(param.BIDS_root, 'high_gamma',
                                                    subject,
                                                    session=param.session,
                                                    task_name=param.task_name,
                                                    preprocess_folder=param.preprocessing_folder,
                                                    preprocess_steps=param.preprocess_steps,
                                                    channel_types={"seeg": True, "ecog": True},
                                                    condition=analysis_parameters["conditions"],
                                                    crop_time=None,  # do cropping later, since we might be downsampling
                                                    aseg=param.aseg,
                                                    montage_space=param.montage_space,
                                                    get_mni_coord=True
                                                    )
                    # %
                    if epochs is None:
                        continue
                    mni_coord = mni_coord[ mni_coord.channels.isin(epochs.ch_names) ]
                    
                    # % check if binning/downsampling
                    if analysis_parameters['binning_parameters']['do_binning']:
                        if analysis_parameters['binning_parameters']['downsample'] and \
                                analysis_parameters['binning_parameters']['downsample'] > 0:
                            epochs.resample(float(analysis_parameters['binning_parameters']['downsample']))
                        elif analysis_parameters['binning_parameters']["bins_duration_ms"] is not None:
                            n_samples = int(np.floor(
                                analysis_parameters['binning_parameters']["bins_duration_ms"] * epochs.info[
                                    "sfreq"] / 1000))
                            epochs_data = moving_average(epochs.get_data(), n_samples, axis=-1, overlapping=False)
                            times = moving_average(epochs.times, n_samples)
                            info = epochs.info
                            info['sfreq'] = 1 / (n_samples / epochs.info['sfreq'])
                            epochs = mne.EpochsArray(epochs_data, info, tmin=times[0], events=epochs.events,
                                                     event_id=epochs.event_id, on_missing="warning",
                                                     metadata=epochs.metadata)
                    # get labels & roi_picks for this analysis
                    if roi == "gnw":
                        labels, _ = mne.get_montage_volume_labels(
                            epochs.get_montage(), "sub-" + subject,
                            subjects_dir=Path(param.BIDS_root, "derivatives", "fs"), aseg="aparc.a2009s+aseg")
                        roi_picks = find_channels_in_roi(param.rois[roi], labels)
                    else:
                        labels, _ = get_montage_volume_labels_wang(
                            epochs.get_montage(), "sub-" + subject,
                            subjects_dir=Path(param.BIDS_root, "derivatives", "fs"), aseg='wang15_mplbl')
                        roi_picks = find_channels_in_roi(param.rois[roi], labels)

                    # figure out which categories we are comparing based on unique categories in epochs data
                    categories = np.unique(epochs.metadata.category)
                    
                    if electrode_list is None:
                        # init the electrode list
                        electrode_list = pd.DataFrame([], columns=['channels', 'ch_types', 'x', 'y', 'z', 'roi_pick', categories[0], categories[1]])

                    # get select electrodes
                    selective_category_1 = sub_results.loc[sub_results['condition'] == categories[0]]['channel'].tolist()
                    selective_category_2 = sub_results.loc[sub_results['condition'] == categories[1]]['channel'].tolist()
                    
                    print('Initial list:')
                    print(roi + ': ' + ' '.join(map(str, roi_picks)))
                    print('\tcategory ' + categories[0] + ': ' + ' '.join(map(str, selective_category_1)))
                    print('\tcategory ' + categories[1] + ': ' + ' '.join(map(str, selective_category_2)))
                    
                    
                    #%
                    # identify electrodes in either GNW and/or IIT rois. These electrodes need to be removed from category selectivity if they reside in these ROIS
                    labels, _ = mne.get_montage_volume_labels(
                        epochs.get_montage(), "sub-" + subject,
                        subjects_dir=Path(param.BIDS_root, "derivatives", "fs"), aseg="aparc.a2009s+aseg")
                    gnw_picks = find_channels_in_roi(param.rois["gnw"], labels)
                    labels, _ = get_montage_volume_labels_wang(
                        epochs.get_montage(), "sub-" + subject,
                        subjects_dir=Path(param.BIDS_root, "derivatives", "fs"), aseg='wang15_mplbl')
                    iit_picks = find_channels_in_roi(param.rois["iit"], labels)
                    
                    # create a set of object/face selectve electrodes that are not found in either of the theory rois
                    selective_category_1 = [i for i in selective_category_1 if i not in gnw_picks+iit_picks]
                    selective_category_2 = [i for i in selective_category_2 if i not in gnw_picks+iit_picks]
                    
                    
                    print('Initial list (roi pruned):')
                    print(roi + ': ' + ' '.join(map(str, roi_picks)))
                    print('\tcategory ' + categories[0] + ': ' + ' '.join(map(str, selective_category_1)))
                    print('\tcategory ' + categories[1] + ': ' + ' '.join(map(str, selective_category_2)))
                    
                    #% plot potential roi_picks
                    activation_windows = analysis_parameters['activation_windows'][roi]
                    baseline_win = analysis_parameters['baseline_window']
                    for i,picks_ in enumerate(zip(['roi_picks', 'face_selective', 'object_selective'], [roi_picks, selective_category_1, selective_category_2])):
                        picks = picks_[1]
                        if len(picks)>0:
                            fig, ax = plt.subplots( int(np.ceil(len(picks)/5)), int(np.min( (len(picks), 5))), figsize=(20,5), subplot_kw={'visible': False})
                            if isinstance( ax, np.ndarray):
                                ax = ax.ravel()
                            else:
                                ax = [ax]
                            ep = epochs.copy()
                            ep = ep.pick( picks )
                            if i > 0:
                                ep = ep[ categories[i-1] ]
                            avg = ep.get_data()
                            for i, r in enumerate(picks):
                                m = np.squeeze( np.mean( avg[:, i, :], axis=0) )
                                err = np.squeeze( np.std( avg[:, i, :], axis=0) )/np.sqrt(avg.shape[0])
                                ax[i].plot( epochs.times, m )
                                ax[i].fill_between( epochs.times,  m-err, m+err, alpha=0.3)
                                ax[i].axvspan(baseline_win[0], baseline_win[1], fc='y', alpha=0.3)
                                ax[i].text( baseline_win[0], ax[i].get_ylim()[1]+0.05, r, **{'fontsize': 7})
                                ax[i].set_visible(True)
                                for patches in activation_windows:
                                    tmp = ep.copy()
                                    col = 'b'
                                    if patches[0]>1:
                                        tmp = tmp[['face/1500ms', 'object/1500ms']]
                                        col = 'r'
                                        
                                    mm = tmp.get_data(picks=r, tmin=baseline_win[0], tmax=baseline_win[1]).mean(2)
                                    m = np.mean(mm)
                                    err = np.std( mm )/np.sqrt( mm.shape[0] - 1)
                                    ax[i].errorbar( baseline_win[0]+(baseline_win[1]-baseline_win[0])/2, m, err, fmt='o'+col)
                                        
                                    pp = stats.wilcoxon(
                                        tmp.get_data(picks=r, tmin=patches[0], tmax=patches[1]).mean(2),
                                        tmp.get_data(picks=r, tmin=baseline_win[0], tmax=baseline_win[1]).mean(2), method='exact',
                                        alternative='greater').pvalue
                                    
                                    ax[i].errorbar( patches[0]+(patches[1]-patches[0])/2, np.mean(tmp.get_data(picks=r, tmin=patches[0], tmax=patches[1]).mean(2)), np.std(tmp.get_data(picks=r, tmin=patches[0], tmax=patches[1]).mean(2))/np.sqrt( tmp.get_data(picks=r, tmin=patches[0], tmax=patches[1]).mean(2).shape[0]-1 ) , fmt='o'+col)
                                    ax[i].text( patches[0], ax[i].get_ylim()[1], np.round(pp[0],3), **{'fontsize': 7})
                                    ax[i].axvspan(patches[0], patches[1], fc='g', alpha=0.3)
                            file_name = Path(save_path_fig, subject + "_" + roi + "_" + picks_[0]+".png")
                            plt.savefig(file_name, dpi=150)
                            plt.close();
                    
                                
                    
                    # % now check if they are active using HGP time-window analysis
                    activation_windows = analysis_parameters['activation_windows'][roi]
                    baseline_win = analysis_parameters['baseline_window']
                    for act_win in activation_windows:
                        if act_win[0] > 1:
                            ep = epochs[['face/1500ms', 'object/1500ms']]
                        else:
                            ep = epochs.copy()

                        if len(roi_picks):
                            # return only those electrodes that are "active" against baseline in the 0.3-0.5s window
                            tmp = ep.copy()
                            tmp.pick(roi_picks)
                            p = np.array([])
                            for ch in tmp.ch_names: 
                                p = np.append(p, stats.wilcoxon(
                                    tmp.get_data(picks=ch, tmin=act_win[0], tmax=act_win[1]).mean(2),
                                    tmp.get_data(picks=ch, tmin=baseline_win[0], tmax=baseline_win[1]).mean(2),
                                    alternative='greater').pvalue)
                            roi_picks = np.asarray(roi_picks)[(p < 0.05)].tolist()

                        if len(selective_category_1):
                            # return only those electrodes that are "active" against baseline in the 0.3-0.5s window
                            tmp = ep[categories[0]].pick(selective_category_1)
                            p = np.array([])
                            for ch in tmp.ch_names:
                                p = np.append(p, stats.wilcoxon(
                                    tmp.get_data(picks=ch, tmin=act_win[0], tmax=act_win[1]).mean(2),
                                    tmp.get_data(picks=ch, tmin=baseline_win[0], tmax=baseline_win[1]).mean(2),
                                    alternative='greater').pvalue)
                            print(p)
                            selective_category_1 = np.asarray(selective_category_1)[(p < 0.05)].tolist()

                        if len(selective_category_2):
                            # return only those electrodes that are "active" against baseline in the 0.3-0.5s window
                            tmp = ep[categories[1]].pick(selective_category_2)
                            p = np.array([])
                            for ch in tmp.ch_names:
                                p = np.append(p, stats.wilcoxon(
                                    tmp.get_data(picks=ch, tmin=act_win[0], tmax=act_win[1]).mean(2),
                                    tmp.get_data(picks=ch, tmin=baseline_win[0], tmax=baseline_win[1]).mean(2),
                                    alternative='greater').pvalue)
                            selective_category_2 = np.asarray(selective_category_2)[(p < 0.05)].tolist()


                    # if args.sim is not None:
                    #     # just take a few electrodes for the simulation
                    #     roi_picks = roi_picks[0:3]
                    #     selective_category_1 = selective_category_1[0:20]
                    #     selective_category_2 = selective_category_2[0:20]

                    # % anything to analyze?
                    print(roi + ': ' + ' '.join(map(str, roi_picks)))
                    print('\tcategory ' + categories[0] + ': ' + ' '.join(map(str, selective_category_1)))
                    print('\tcategory ' + categories[1] + ': ' + ' '.join(map(str, selective_category_2)))

                    # % if we have electrodes to analyze, let's move forward
                    if (len(roi_picks) and (len(selective_category_1) or len(selective_category_2))):
                        print('\nContinue to connectivity analysis...\n')
                    else:
                        print('\n!!! Nothing to analyze...moving on !!!\n')
                        continue
                    # electrode_list[subject] = {'roi_picks': roi_picks, categories[0]: selective_category_1, categories[1]: selective_category_2};    
                    
                    # store the electrodes for offline plotting
                    df_roi = mni_coord[ mni_coord.channels.isin( roi_picks ) ].copy().reset_index(drop=True)
                    df_roi['roi_pick'] = True
                    df_roi[categories[0]] = False
                    df_roi[categories[1]] = False
                    
                    df_cat1 = mni_coord[ mni_coord.channels.isin( selective_category_1 ) ].copy().reset_index(drop=True)
                    df_cat1['roi_pick'] = False
                    df_cat1[categories[0]] = True
                    df_cat1[categories[1]] = False
                    
                    df_cat2 = mni_coord[ mni_coord.channels.isin( selective_category_2 ) ].copy().reset_index(drop=True)
                    df_cat2['roi_pick'] = False
                    df_cat2[categories[0]] = False
                    df_cat2[categories[1]] = True
                    
                    electrode_list = pd.concat((electrode_list, df_roi, df_cat1, df_cat2)).reset_index(drop=True)
                    
                    
                    
                    
                    

                        
                    # %%
                    epochs, _ = load_epochs(param.BIDS_root, analysis_parameters["signal"],
                                            subject,
                                            session=param.session,
                                            task_name=param.task_name,
                                            preprocess_folder=param.preprocessing_folder,
                                            preprocess_steps=param.preprocess_steps,
                                            channel_types={"seeg": True, "ecog": True},
                                            condition=analysis_parameters["conditions"],
                                            crop_time=None,  # do cropping later, since we might be downsampling
                                            aseg=param.aseg,
                                            montage_space=param.montage_space,
                                            get_mni_coord=False
                                            )

                    if analysis_parameters["subtract_evoked"]:
                        epochs.subtract_evoked()
                    if analysis_parameters["regress_evoked"]:
                        epochs = regress_evoked(epochs)

                    # do connectivity on a subset of trials from conditions list
                    if 'test_group' in analysis_parameters.keys():
                        epochs = epochs[ analysis_parameters['test_group'] ]
                    
                    # %% loop through face and object-selective electrodes and compute connectivity with roi
                    ep_ = [epochs[categories[0]], epochs[categories[1]]]
                    chs_roi = [epochs.ch_names.index(n) for n in roi_picks]
                    for c, selective_electrodes in enumerate([selective_category_1, selective_category_2]):
                        if len(selective_electrodes):
                            # convert to indices
                            chs_ = [epochs.ch_names.index(n) for n in selective_electrodes]
                            indices = (np.repeat(chs_, len(chs_roi)), np.tile(chs_roi, len(chs_)))

                            for i in range(2):  # loop over ep0/ep1

                                method = analysis_parameters["method"]
                                method_params = analysis_parameters["method_params"][method]
                                
                                keep_trials = False
                                if ('keep_trials' in method_params):
                                    keep_trials = method_params['keep_trials']
                                    
                                # check if single electrode stats required
                                conndat_perm = []
                                if 'single_electrode_stats' in analysis_parameters.keys():
                                    single_elec_stats = analysis_parameters['single_electrode_stats']
                                    n_jobs = 1
                                else:
                                    single_elec_stats = False
                                    n_jobs = -1
                                    
                                

                                if method == 'ppc':
                                    analysis_time = ep_[i].times
                                    freqs = np.asarray(method_params["freqs"])
                                    conn = spectral_connectivity_epochs(
                                        ep_[i], method=method, mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                        cwt_freqs=np.asarray(method_params["freqs"]),
                                        cwt_n_cycles=np.asarray(method_params["n_cycles"]),
                                        indices=indices,
                                        faverage=False, mt_adaptive=False, n_jobs=n_jobs)
                                    conndat = conn.get_data()
                                    analysis_time = conn.times
                                    freqs = conn.freqs
                                    conn = None
                                    
                                    if single_elec_stats:
                                        # perms = np.transpose(perms, [1,2,3,0]) # nodes x freqs x times x nperms
                                        conndat_perm = np.zeros_like(conndat)*np.nan
                                        print('running single electrode permutations...')
                                        for r, index in enumerate(zip( indices[0], indices[1])):
                                            
                                            # tmp = Parallel(n_jobs=1)(delayed( spectral_connectivity_epochs_shuffle)(ep_[i], method=method, mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                            #     cwt_freqs=np.asarray(method_params["freqs"]),
                                            #     cwt_n_cycles=np.asarray(method_params["n_cycles"]),
                                            #     indices=((np.asarray([index[0]]), np.asarray([index[1]]))),
                                            #     faverage=False, mt_adaptive=False, n_jobs=-1, verbose='ERROR') for _ in tqdm(range(200)))
                                            # perms = np.squeeze(np.asarray( tmp )) # nperms x nodes x freqs x times
                                            
                                            # v2
                                            # perms = np.zeros((200, conndat.shape[1], conndat.shape[2]))
                                            # for p in range(200):
                                            #     tmp = spectral_connectivity_epochs_shuffle(
                                            #         ep_[i], method=method, mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                            #         cwt_freqs=np.asarray(method_params["freqs"]),
                                            #         cwt_n_cycles=np.asarray(method_params["n_cycles"]),
                                            #         indices=((np.asarray([index[0]]), np.asarray([index[1]]))),
                                            #         faverage=False, mt_adaptive=False, n_jobs=-1, verbose=False)
                                            #     perms[p, :, :] = np.squeeze( tmp )
                                            
                                            #v3 (should be fastest w/ paralleization)
                                            perms = spectral_connectivity_epochs_shuffle(
                                                ep_[i], method=method, mode='cwt_morlet', sfreq=epochs.info['sfreq'],
                                                cwt_freqs=np.asarray(method_params["freqs"]),
                                                cwt_n_cycles=np.asarray(method_params["n_cycles"]),
                                                indices=((np.asarray([index[0]]), np.asarray([index[1]]))),
                                                faverage=False, mt_adaptive=False, n_jobs=1, verbose=False, n_perms=200)
                                           
                                            stats_ = cluster_test( conndat[r,:,:], perms, z_threshold=1.96, tail=1)
                                            perms = None # deallocate
                                            if stats_[4].shape == 0:
                                                conndat_perm[r, :, :] = np.ones_like(conndat_perm[r, :, :] )
                                            else:
                                                conndat_perm[r, :, :] = stats_[4] # store the p-values only
                                        
                                elif method == 'dfc':
                                    #%
                                    slwin_len = method_params['window_len']    # windows of length 500ms
                                    slwin_step = method_params['step']  # 20ms step between each window (or 480ms overlap)
                                    times = ep_[i].times
                                    trials = np.arange(ep_[i].get_data().shape[0])
                                    # define the sliding windows
                                    sl_win = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]
                                    
                                    # conndat = np.empty((len(indices[0]), 1, len(sl_win)))
                                    # for i_, ind_ in enumerate(zip( indices[0], indices[1])):
                                    #     x = ep_[i].get_data()[:, [ind_[0], ind_[1]], :]
                                    #     rr = ['r0', 'r1']
                                    #     x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                                    #                      coords=(trials, rr, times))
                                        
                                    #     # compute the DFC on sliding windows
                                    #     dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win, verbose=False)
                                    #     conndat[i_, 0, :] = dfc.mean('trials').squeeze().data
                                    
                                    # parallelized version
                                    # get the relevant data
                                    x = ep_[i].get_data()[:, np.concatenate((chs_, chs_roi )), :]
                                    rr = ['selective' for k in chs_]+['roi' for k in chs_roi]
                                    x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                                                     coords=(trials, rr, times))
                                    dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win, n_jobs=-1, verbose=False, **{'roi_relation': 'inter'})
                                    # dfc is returned by in order of selective channels -> roi
                                    conndat = np.expand_dims( dfc.mean('trials').data, axis=1) # expand to conform with general conndat shape chan x freq (1) x time
                                    
                                    print(indices )
                                    print( conndat.shape )
                                    analysis_time = dfc['times'].values
                                    freqs = None
                                    #%
                                elif method == 'dfc_tfr':
                                    #%
                                    slwin_len = method_params['window_len']    # windows of length 500ms
                                    slwin_step = method_params['step']  # 20ms step between each window (or 480ms overlap)
                                    times = ep_[i].times
                                    trials = np.arange(ep_[i].get_data().shape[0])
                                    # define the sliding windows
                                    sl_win = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]
                                    
                                    #np.concatenate( (np.arange(2, 30, 1), np.arange(30, 180, 5)) ),
                                    #np.concatenate( (np.tile(4, len(np.arange(2, 30, 1))), np.round( np.linspace(8, 45, len(np.arange(30, 180, 5))) ) )),
                                    tfr = mne.time_frequency.tfr_multitaper(
                                        ep_[i],
                                        freqs=method_params['freqs'], 
                                        n_cycles=method_params['n_cycles'],
                                        use_fft=True,
                                        return_itc=False,
                                        average=False,
                                        time_bandwidth=2.,
                                        n_jobs=-1,
                                        verbose=True)
                                    
                                    # conndat = np.empty((len(indices[0]), len(tfr.freqs), len(sl_win)))
                                    # for f, freq in enumerate( tfr.freqs ):
                                    #     for i_, ind_ in enumerate(zip( indices[0], indices[1])):
                                    #         x = np.squeeze( tfr.data[:, [ind_[0], ind_[1]], f, :] )
                                    #         rr = ['r0', 'r1']
                                    #         x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                                    #                          coords=(trials, rr, times))
                                            
                                    #         # compute the DFC on sliding windows
                                    #         dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win, n_jobs=-1, verbose=False)
                                    #         conndat[i_, f, :] = dfc.mean('trials').squeeze().data
                                    
                                    # # parallelized version
                                    # conndat = np.empty((len(indices[0]), len(tfr.freqs), len(sl_win)))
                                    # for f, freq in enumerate( tfr.freqs ):
                                    #     # get data from relevant channels (selective first, roi second)
                                    #     x = np.squeeze( tfr.data[:, np.concatenate((chs_, chs_roi )), f, :] )
                                    #     rr = ['selective' for k in chs_]+['roi' for k in chs_roi]
                                    #     x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                                    #                      coords=(trials, rr, times))
                                    #     dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win, n_jobs=-1, verbose=False, **{'roi_relation': 'inter'})
                                    #     # dfc is returned by in order of selective channels -> roi (which is compatible with the conndat averaging procedure below)
                                    #     conndat[:, f, :] = dfc.mean('trials').squeeze().data
                                        
                                        
                                        
                                    if keep_trials:
                                        conndat = np.empty(( len(indices[0]), len(ep_[i]), len(tfr.freqs), len(sl_win)))
                                    else:
                                        conndat = np.empty(( len(indices[0]), len(tfr.freqs), len(sl_win)))
                                    for f, freq in enumerate( tfr.freqs ):
                                        # get data from relevant channels (selective first, roi second)
                                        x = np.squeeze( tfr.data[:, np.concatenate((chs_, chs_roi )), f, :] )
                                        rr = ['selective' for k in chs_]+['roi' for k in chs_roi]
                                        x = xr.DataArray(x, dims=('trials', 'space', 'times'),
                                                         coords=(trials, rr, times))
                                        dfc = conn_dfc(x, times='times', roi='space', win_sample=sl_win, n_jobs=-1, verbose=False, **{'roi_relation': 'inter'})
                                        # dfc is returned by in order of selective channels -> roi (which is compatible with the conndat averaging procedure below)
                                        if keep_trials:
                                            conndat[:, :, f, :] = np.transpose(dfc.data, [1, 0, 2]) # put indices on the first dimension
                                        else:
                                            conndat[:, f, :] = dfc.mean('trials').squeeze().data
                                    
                                    analysis_time = dfc['times'].values
                                    freqs = tfr.freqs       
                                    #%
                                elif method == 'smi':

                                    tmp = ep_[i]
                                    nwin = round(0.1 * epochs.info['sfreq']) + 1
                                    wins = np.arange(0, len(epochs.times) - nwin, nwin / 2, dtype=int)
                                    conndat = np.empty((len(indices[0]), 1, len(wins)))
                                    counter = 0

                                    info = mne.create_info(
                                        [epochs.ch_names[ii] for ii in np.concatenate((chs_, chs_roi))],
                                        epochs.info['sfreq'], ch_types='eeg')
                                    epochs_ = mne.EpochsArray(tmp.get_data()[:, np.concatenate((chs_, chs_roi)), :],
                                                              info, tmin=epochs.times[0])
                                    tcount = 0
                                    analysis_time = np.array([])
                                    freqs = method_params["tau"]
                                    for n in wins:
                                        wsmi, smi, _, _ = epochs_compute_wsmi(
                                            epochs_, tmin=epochs_.times[n], tmax=epochs_.times[n + nwin],
                                            kernel=method_params["kernel"], tau=method_params["tau"],
                                            method_params={'nthreads': 'auto', 'bypass_csd': True})
                                        # unravel the data and only keep the necessary connections
                                        wsmi = wsmi.mean(2)
                                        tmp = np.empty((len(indices[0])))
                                        for ii, _ in enumerate(chs_):
                                            tmp[ii + (ii * (len(chs_roi) - 1)):((ii + 1) * (len(chs_roi)))] = \
                                                wsmi[ii, len(chs_)::]

                                        analysis_time = np.append(analysis_time, np.mean(epochs_.times[n:n + nwin]))
                                        conndat[:, 0, tcount] = tmp
                                        tcount += 1

                                # append results to main connectivity dataframe
                                for k,inds in enumerate(zip(indices[0], indices[1])):
                                    if isinstance(conndat_perm, np.ndarray):
                                        df_conn = df_conn.append({'category': categories[c], 'trials': categories[i], 'seed': epochs.ch_names[inds[0]], 'target': epochs.ch_names[inds[1]], 'conn': conndat[k,::], 'conn_perm': conndat_perm[k,::]}, ignore_index=True)
                                    else:
                                        df_conn = df_conn.append({'category': categories[c], 'trials': categories[i], 'seed': epochs.ch_names[inds[0]], 'target': epochs.ch_names[inds[1]], 'conn': conndat[k,::], 'conn_perm': []}, ignore_index=True)
                                        
                                    
                                # % for each category-selective electrode (seed), we take the average connectivity to
                                # an roi-electrode and append each result
                                for seed in np.unique(indices[0]):
                                    idx_ = (indices[0] == seed)
                                    tmp = np.mean(conndat[idx_, ::], axis=0)
                                    tmp_r = []
                                    if single_elec_stats:
                                        tmp_r = np.mean(conndat_perm[idx_, ::], axis=0)
                                    if c == 0:
                                        if i == 0:
                                            category_1_selective.append(subject)
                                            cat1_to_cat1.append(tmp)
                                            cat1_to_cat1_perm.append(tmp_r)
                                        else:
                                            cat1_to_cat2.append(tmp)
                                            cat1_to_cat2_perm.append(tmp_r)
                                    else:
                                        if i == 0:
                                            category_2_selective.append(subject)
                                            cat2_to_cat1.append(tmp)
                                            cat2_to_cat1_perm.append(tmp_r)
                                        else:
                                            cat2_to_cat2.append(tmp)
                                            cat2_to_cat2_perm.append(tmp_r)

                #%% convert results lists to arrays
                cat1_to_cat1 = np.asarray(cat1_to_cat1)
                cat1_to_cat2 = np.asarray(cat1_to_cat2)
                cat2_to_cat1 = np.asarray(cat2_to_cat1)
                cat2_to_cat2 = np.asarray(cat2_to_cat2)
                
                cat1_to_cat1_perm = np.asarray(cat1_to_cat1_perm)
                cat1_to_cat2_perm = np.asarray(cat1_to_cat2_perm)
                cat2_to_cat1_perm = np.asarray(cat2_to_cat1_perm)
                cat2_to_cat2_perm = np.asarray(cat2_to_cat2_perm)
                
                # save the raw results
                file_name = Path(save_path_results, param.files_prefix + roi + "_" + method + ".tar.gz")
                df_conn.to_pickle(file_name)
                file_name = Path(save_path_results, param.files_prefix + roi + "_" + method + "_electrodes.tar.gz")
                electrode_list.to_pickle(file_name)

                #%% Plotting
                if keep_trials:
                    # stats done offline on individual electrode trial data
                    p_values_obj_selective=[]
                    p_values_fac_selective=[]
                else:
                    if np.size(cat1_to_cat1, 1) == 1:
                        # %
                        plt.figure()
                        ax = plt.subplot(211)
                        lines = []
                        for tmp_ in zip(categories, [cat1_to_cat1, cat1_to_cat2]):
                            if tmp_[1].ndim > 1:
                                mean = np.squeeze(tmp_[1].mean(0))
                                sem = np.squeeze(tmp_[1].std(0) / np.sqrt(np.size(tmp_[1], 0) - 1))
                                line, = plt.plot(analysis_time, mean, label=tmp_[0] + '-trials')
                                plt.fill_between(analysis_time, mean + sem, mean - sem)
                                lines.append(line)
                        # stats 
                        if cat1_to_cat1.ndim > 1 and cat1_to_cat2.ndim > 1:
                            tmp_ = cat1_to_cat1 - cat1_to_cat2
                            pval = analysis_parameters["cluster_p_value"]
                            thresh = stats.t.ppf(1 - pval / 2, df=tmp_.shape[0] - 1)
                            F_obs, clusters, cluster_p_values, H0 = \
                                mne.stats.permutation_cluster_1samp_test(tmp_,
                                                                         n_permutations=
                                                                         analysis_parameters[
                                                                             "n_permutations"],
                                                                         threshold=thresh,
                                                                         tail=0)
                            p_values_fac_selective = np.ones_like(F_obs)
                            for cluster, pval in zip(clusters, cluster_p_values):
                                p_values_fac_selective[cluster] = pval
                            sig_mask = (p_values_fac_selective < 0.05)[0]
                            plt.plot(analysis_time[sig_mask],
                                     np.max(cat1_to_cat1.mean(0)) * 1.05 * np.ones((np.sum(sig_mask),)), 'ko')
    
                        plt.title(categories[0] + '-Selective')
                        plt.legend()
    
                        plt.subplot(212)
                        for tmp_ in zip(categories, [cat2_to_cat1, cat2_to_cat2]):
                            if tmp_[1].ndim > 1:
                                mean = np.squeeze(tmp_[1].mean(0))
                                sem = np.squeeze(tmp_[1].std(0) / np.sqrt(np.size(tmp_[1], 0) - 1))
                                plt.plot(analysis_time, mean, label=tmp_[0] + '-trials')
                                plt.fill_between(analysis_time, mean + sem, mean - sem)
    
                                # stats
                        if cat2_to_cat1.ndim > 1 and cat2_to_cat2.ndim > 1:
                            tmp_ = cat2_to_cat1 - cat2_to_cat2
                            pval = analysis_parameters["cluster_p_value"]
                            thresh = stats.t.ppf(1 - pval / 2, df=tmp_.shape[0] - 1)
                            F_obs, clusters, cluster_p_values, H0 = \
                                mne.stats.permutation_cluster_1samp_test(tmp_,
                                                                         n_permutations=
                                                                         analysis_parameters[
                                                                             "n_permutations"],
                                                                         threshold=thresh,
                                                                         tail=0)
                            p_values_obj_selective = np.ones_like(F_obs)
                            for cluster, pval in zip(clusters, cluster_p_values):
                                p_values_obj_selective[cluster] = pval
                            sig_mask = (p_values_obj_selective < 0.05)[0]
                            plt.plot(analysis_time[sig_mask],
                                     np.max(cat2_to_cat1.mean(0)) * 1.05 * np.ones((np.sum(sig_mask),)), 'ko')
    
                        plt.title(categories[1] + '-Selective')
                        plt.legend()
    
                        # %%
                    else:
                        # %%time-freq plots
                        plt.figure(figsize=(12, 6))
                        ax = plt.subplot(231)
                        plt.pcolormesh(analysis_time, freqs, cat1_to_cat1.mean(0))
                        plt.title(categories[0] + '-trials')
                        plt.colorbar()
    
                        plt.subplot(232)
                        plt.pcolormesh(analysis_time, freqs, cat1_to_cat2.mean(0))
                        plt.title(categories[1] + '-trials')
                        plt.colorbar()
    
                        # %
                        plt.subplot(233)
                        tmp_ = cat1_to_cat1 - cat1_to_cat2
    
                        pval = analysis_parameters["cluster_p_value"]
                        thresh = stats.t.ppf(1 - pval / 2, df=tmp_.shape[0] - 1)
                        F_obs, clusters, cluster_p_values, H0 = \
                            mne.stats.permutation_cluster_1samp_test(tmp_,
                                                                     n_permutations=
                                                                     analysis_parameters[
                                                                         "n_permutations"],
                                                                     threshold=thresh,
                                                                     tail=0)
                        p_values_fac_selective = np.ones_like(F_obs)
                        for cluster, pval in zip(clusters, cluster_p_values):
                            p_values_fac_selective[cluster] = pval
    
                        sig_mask = (p_values_fac_selective < analysis_parameters['significance_p_value'])
                        plt.pcolormesh(analysis_time, freqs, tmp_.mean(0))
                        plt.title(categories[0] + '-' + categories[1])
                        plt.colorbar()
                        plt.contour(analysis_time, freqs, sig_mask, colors='k')
    
                        # %
                        plt.subplot(234)
                        plt.pcolormesh(analysis_time, freqs, cat2_to_cat1.mean(0))
                        plt.title(categories[0] + '-trials')
                        plt.colorbar()
    
                        plt.subplot(235)
                        plt.pcolormesh(analysis_time, freqs, cat2_to_cat2.mean(0))
                        plt.title(categories[1] + '-trials')
                        plt.colorbar()
    
                        plt.subplot(236)
                        tmp_ = cat2_to_cat1 - cat2_to_cat2
    
                        pval = analysis_parameters["cluster_p_value"]
                        thresh = stats.t.ppf(1 - pval / 2, df=tmp_.shape[0] - 1)
                        F_obs, clusters, cluster_p_values, \
                        H0 = mne.stats.permutation_cluster_1samp_test(tmp_,
                                                                      n_permutations=
                                                                      analysis_parameters[
                                                                          "n_permutations"],
                                                                      threshold=thresh,
                                                                      tail=0)
                        p_values_obj_selective = np.ones_like(F_obs)
                        for cluster, pval in zip(clusters, cluster_p_values):
                            p_values_obj_selective[cluster] = pval
                        sig_mask = (p_values_obj_selective < analysis_parameters['significance_p_value'])
    
                        plt.pcolormesh(analysis_time, freqs, tmp_.mean(0))
                        plt.title(categories[0] + '-' + categories[1])
                        plt.colorbar()
                        plt.contour(analysis_time, freqs, sig_mask, colors='k')

                    file_name = Path(save_path_fig, param.files_prefix + roi + "_" + method + "_synchrony.png")
                    plt.savefig(file_name, dpi=150)
                    
                # %%
                file_name = Path(save_path_results, param.files_prefix + roi + "_" + method + ".npz")
                np.savez(file_name, category_1_selective=category_1_selective, category_2_selective=category_2_selective, 
                         cat1_to_cat1=cat1_to_cat1,cat1_to_cat2=cat1_to_cat2, 
                         cat2_to_cat1=cat2_to_cat1, cat2_to_cat2=cat2_to_cat2,
                         cat1_to_cat1_perm=cat1_to_cat1_perm,cat1_to_cat2_perm=cat1_to_cat2_perm, 
                         cat2_to_cat1_perm=cat2_to_cat1_perm, cat2_to_cat2_perm=cat2_to_cat2_perm,
                         p_values_obj_selective=p_values_obj_selective, p_values_fac_selective=p_values_fac_selective,
                         analysis_time=analysis_time, freqs=freqs, electrode_list=electrode_list)


            # %%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--sim', type=str, default=None,
                        help="run simulation")
    args = parser.parse_args()
    # check if simulation mode
    if args.sim is None:
        synchrony_analysis(None, save_folder="super")
    else:
        subjects_list = ["sim1"]
        synchrony_analysis(subjects_list, save_folder='sim')
