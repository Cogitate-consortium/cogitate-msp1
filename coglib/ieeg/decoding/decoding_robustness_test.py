#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Decoding analysis - robustness test
    This script subsampling N electrodes, to make sure that the results are not driven from a single subject
    @author: Simon Henin
    simon.henin@nyulangone.org
"""

# %%
import warnings
import argparse
import re
from tqdm import tqdm

from general_helper_functions.data_general_utilities import load_epochs, cluster_test, moving_average
from general_helper_functions.pathHelperFunctions import find_files, path_generator, get_subjects_list
from decoding.decoding_analysis_parameters_class import DecodingAnalysisParameters
from decoding.decoding_helper_functions import *

from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

N_REPEATS = 100
N_CHANNELS = 200
VERBOSE = False


# %%
def decoding_analysis(subjects_list, save_folder="super"):
    # %% Extract config from command line argument
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--sim', type=str, default=None,
                        help="run simulation")
    args = parser.parse_args()
    # If no config was passed, just using them all
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "decoding", "configs"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]
    for config in configs:
        print("-" * 40)
        print("Running population analysis with this config file: \n" + config)

        # %%
        param = DecodingAnalysisParameters(config, sub_id=save_folder)

        if subjects_list is None:
            subjects_list = get_subjects_list(param.BIDS_root, "decoding")



        # loop over each analysis in the configuration
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            # for each analysis, loop over the ROI provided in the configuration file
            for roi in param.rois:
                
                # ======================================================================================================
                # Results/Figure paths
                save_path_results = path_generator(param.save_root,
                                                   analysis=analysis_name,
                                                   preprocessing_steps=param.preprocess_steps,
                                                   fig=False, stats=True)
                save_path_fig = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=True)


                # ======================================================================================================
                # Loading in the data of each subject:
                data = []
                rois = []
                for subject in subjects_list:
                    epochs, mni_coord = load_epochs(param.BIDS_root, analysis_parameters["signal"],
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
                                                    get_mni_coord=False,
                                                    picks_roi=param.rois[roi]
                                                    )

                    if epochs is None:
                        continue

                    if analysis_parameters['crop_time']:
                        epochs.crop(tmin=analysis_parameters['crop_time'][0], tmax=analysis_parameters['crop_time'][1])
                        # check if binning/downsampling
                    if analysis_parameters['binning_parameters']['do_binning']:
                        if analysis_parameters['binning_parameters']['downsample'] and \
                                analysis_parameters['binning_parameters']['downsample'] > 0:
                            epochs.resample(float(analysis_parameters['binning_parameters']['downsample']))
                            epochs_data = epochs.get_data()
                            times = epochs.times
                        elif analysis_parameters['binning_parameters']["bins_duration_ms"] is not None:
                            n_samples = int(np.floor(
                                analysis_parameters['binning_parameters']["bins_duration_ms"] * epochs.info[
                                    "sfreq"] / 1000))
                            epochs_data = moving_average(epochs.get_data(), n_samples, axis=-1, overlapping=False)
                            times = moving_average(epochs.times, n_samples)
                    else:
                        epochs_data = epochs.get_data()
                        times = epochs.times

                    # % get relevant trials and stack them
                    idx = []
                    num_durations = len(np.unique([re.findall('[0-9]+', x) for x in analysis_parameters["conditions"]]))
                    if num_durations == 3:  # if combining all durations, then combine to maximize the number of
                        # trials
                        for task in epochs.metadata.task_relevance.sort_values().unique():
                            for cat in epochs.metadata.category.sort_values().unique():
                                for orientation in epochs.metadata.orientation.sort_values().unique():
                                    idx_ = np.where(epochs.metadata['category'].str.contains(cat) & epochs.metadata[
                                        'orientation'].str.contains(orientation) & epochs.metadata[
                                                        'task_relevance'].str.contains(task))[0]
                                    print('task: %s, category: %s, orientation: %s, trials: %i' % (
                                        task, cat, orientation, len(idx_)))
                                    idx.extend(idx_)
                    else:
                        for task in epochs.metadata.task_relevance.sort_values().unique():
                            for cat in epochs.metadata.category.sort_values().unique():
                                for dur in epochs.metadata.duration.sort_values().unique():
                                    idx_ = np.where(epochs.metadata['category'].str.contains(cat) & epochs.metadata[
                                        'duration'].str.contains(dur) & epochs.metadata['task_relevance'].str.contains(
                                        task))[0]
                                    idx_ = idx_[0:26]  # grab the first 26
                                    print('task: %s, category: %s, duration: %s, trials: %i' % (
                                        task, cat, dur, len(idx_)))
                                    idx.extend(idx_)
                    data.append(epochs_data[idx, :, :])

                    # % get the assigned roi (needed for ROI specificity analsysis)
                    roi_, _ = mne.get_montage_volume_labels(
                        epochs.get_montage(), "sub-" + subject, subjects_dir=Path(param.BIDS_root, "derivatives", "fs"),
                        aseg=param.aseg)
                    rois_ = []
                    for ch in roi_.keys():
                        # Looping through each label of this specific channel:
                        for label in roi_[ch]:
                            if label in param.rois[roi]:
                                rois_.append(label)
                                break
                    rois.append(rois_)
                data = np.concatenate(data, axis=1)
                rois = np.concatenate(rois, axis=0)
                time = times
                info = epochs.metadata.iloc[idx]  # sorted



                # ======================================================================================================
                # -- decoder setup --%
                y = info[analysis_parameters['decoding_target']].values
                if analysis_parameters['grouping_condition']:
                    groups = info[analysis_parameters['grouping_condition']].to_numpy()
                else:
                    groups = None

                # initiliaze classifier pipeline
                classifier_parameters = analysis_parameters["classifier_parameters"]
                clf_steps = []
                if classifier_parameters['scaler']:
                    clf_steps.append(StandardScaler())
                if classifier_parameters["do_feature_selection"]:
                    if classifier_parameters["feature_selection_parameters"]["prop_channels"] < 1:
                        k = int(
                            np.size(data, 1) * classifier_parameters["feature_selection_parameters"]["prop_channels"])
                    else:
                        k = analysis_parameters["classifier_parameters"]["feature_selection_parameters"][
                            "prop_channels"]
                    clf_steps.append(SelectKBest(f_classif, k=k))
                clf_steps.append(svm.SVC(kernel='linear', class_weight='balanced'))
                clf = make_pipeline(*clf_steps)





                # ======================================================================================================
                # -- Main Analysis Section --#
                # % check if special ROI-specificity.
                if analysis_parameters['roi_specificity']["do_roi_analysis"]:
                    
                    # for this analysis we loop over all rois and get ACC within the whole time window, save and exit
                    # for robustness tests, this is a bit trickier, since the ROIs depend on the channels randomly selected
                    # therefore, we need to subselect channels first, then loop over rois in analysis list, and append results per roi
                    tm = analysis_parameters['roi_specificity']['time_window'][roi]  # get the time window for the roi
                    data = data[:, :, np.where((time >= tm[0]) & (time <= tm[1]))[0]]
                    rois_combined = np.unique(
                        [r.replace('ctx_lh_', '').replace('ctx_rh_', '') for r in param.rois[roi]])
                        
                    # initialize a dict of empty lists for each roi
                    decoding_scores = {r: [] for r in rois_combined}     
                    for n in tqdm(range(N_REPEATS)):
                        # select N_CHANNELS
                        chs_ = np.random.choice(data.shape[1], N_CHANNELS, replace=False)
                        rois_ = rois[chs_].tolist() 
                        data_ = data[:, chs_, :]    
                        
                        # loop of possible rois
                        for i, r in enumerate(rois_combined):
                            # get all eletrodes in the roi
                            chs = np.where(np.core.defchararray.find(rois_, r) > 0)[0]
                            print('%i chs in %s' % (len(chs), r))
                            if len(chs):
                                data__ = np.reshape(data_[:, chs, :], (np.size(data, 0), len(chs) * np.size(data_, 2)))
                                data__ = np.expand_dims(data__, 2)
                                scores, coefs = \
                                    zip(*Parallel(n_jobs=param.classifier_n_jobs)(delayed(
                                        temporal_generalization_decoding)(clf, data__, y,
                                                                          analysis_parameters[
                                                                              "cross_validation_parameters"],
                                                                          metric=classifier_parameters['metric'],
                                                                          train_group=analysis_parameters["train_group"],
                                                                          test_group=analysis_parameters["test_group"],
                                                                          groups=groups,
                                                                          do_only_diag=True,
                                                                          classifier_n_jobs=1, verbose=False) for _ in range(classifier_parameters['repeats'])))
                                # add the result to the appropriate roi list
                                decoding_scores[ r ].append( np.mean( np.concatenate(scores, 0) ) )                                          


                    # % convert from lists of decoding scores & save the results
                    rois_combined = np.array(list(decoding_scores.keys()))
                    # average across repeats
                    decoding_scores = np.asarray([np.mean(decoding_scores[key]) for key in decoding_scores.keys()])
                                                          
                        
                    file_name = Path(save_path_results, param.files_prefix + roi + "_decoding_roi.npz")
                    np.savez(file_name, decoding_scores=decoding_scores, rois=rois_combined,
                             analysis_parameters=analysis_parameters)

                    # % roi decoding plot
                    file_name = Path(save_path_fig, param.files_prefix + roi + "_decoding_specificity.png")
                    cmaps = ["Oranges", "Purples"]
                    if analysis_parameters["test_group"] is not None:
                        cmap = cmaps[["Relevant non-target", "Irrelevant"].index(analysis_parameters["test_group"])]
                    else:
                        cmap = cmaps[0]
                    plot_roi_specificity(param.BIDS_root, rois_combined, decoding_scores, cmap=cmap,
                                         vmax=np.round(np.max(decoding_scores), 2), filename=file_name)

                else:
                    # perform N_REPEATS of the decoding analysis, subselecting N_CHANNELS each time using n_channel_subsample
                    decoding_scores, coefs = \
                        zip(*Parallel(n_jobs=param.classifier_n_jobs)(delayed(
                            temporal_generalization_decoding)(clf, data, y,
                                                              analysis_parameters["cross_validation_parameters"],
                                                              metric=classifier_parameters['metric'],
                                                              train_group=analysis_parameters["train_group"],
                                                              test_group=analysis_parameters["test_group"],
                                                              groups=groups,
                                                              do_only_diag=analysis_parameters["do_only_diagonal"],
                                                              classifier_n_jobs=1,
                                                              n_channel_subsample = N_CHANNELS,
                                                              verbose=False) for _ in tqdm(range(N_REPEATS))))
                    decoding_scores = np.concatenate(decoding_scores, axis=0)


                    # %% save the results
                    file_name = Path(save_path_results, param.files_prefix + roi + "_decoding.npz")
                    np.savez(file_name, time=time, decoding_scores=decoding_scores,
                             analysis_parameters=analysis_parameters)

                    # %% Plot average decoding scores of 5 splits
                    if decoding_scores.ndim == 3:
                        # do temporal generalization plot first
                        fig, ax = plt.subplots(1, 1)
                        im = ax.imshow(np.squeeze(decoding_scores.mean(0)), interpolation='lanczos', origin='lower',
                                       cmap='RdBu_r',
                                       extent=time[[0, -1, 0, -1]], vmin=0., vmax=1., alpha=1.)
                                       
                        ax.set_xlabel('Testing Time (s)')
                        ax.set_ylabel('Training Time (s)')
                        ax.set_title('Temporal generalization')
                        ax.axvline(0, color='k')
                        ax.axhline(0, color='k')
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('ACC')
                        file_name = Path(save_path_fig, param.files_prefix + roi + "_decoding_tempgen.png")
                        plt.savefig(file_name, dpi=150)
                        decoding_scores = np.diagonal(decoding_scores.T)

                    # plot the diagonal results    
                    fig, ax = plt.subplots(1)
                    ax.plot(time, decoding_scores.mean(0), label='score')
                    ci = 1.96 * decoding_scores.std(0) / np.sqrt(np.size(decoding_scores, 0))
                    ax.fill_between(time, decoding_scores.mean(0) - ci, decoding_scores.mean(0) + ci, alpha=0.5)
                    ax.axhline(1 / len(np.unique(y)), color='k', linestyle='--', label='chance')
                    ax.axvline(0, color='k')
                    plt.grid()
                    plt.legend()
                    # %% save the figure
                    file_name = Path(save_path_fig, param.files_prefix + roi + "_decoding.png")
                    plt.savefig(file_name, dpi=150)


if __name__ == "__main__":

        decoding_analysis(None, save_folder="super_robustness")
