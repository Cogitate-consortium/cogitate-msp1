"""
    Decoding analysis - control analysis comparing IIT vs. IIT+GNW ROI decoding
    @author: Simon Henin
    simon.henin@nyulangone.org
"""

# %% import
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


import matplotlib.pyplot as plt

from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# for calibrated classifier

from sklearn.preprocessing import label_binarize
from decoding.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV as CCCV

#%%
def decoding_analysis_iit_vs_iitgnw(subjects_list=None, save_folder="super"):
    
    # configs to analyze in this analysis
    # configs = ['decoding/configs/decoding_category_roi_basic.json', 'decoding/configs/decoding_orientation_roi.json']
    configs = find_files(Path(os.getcwd(), "decoding", "configs"), naming_pattern="*", extension=".json")
    
    for config in configs:
        # this control analysis only runs on ther basic category decoding config
        param = DecodingAnalysisParameters(config, sub_id=save_folder)
        subjects_list = get_subjects_list(param.BIDS_root, "decoding")
    
        # only do this analysis on the main category decoding (e.g. faces vs objects relevant, faces vs. objects irrelevant), not on cross-decoding etc...
        valid_analysis_names = ["category_decoding_faces_objects_relevant", "category_decoding_faces_objects_irrelevant", "category_decoding_letter_false_relevant", "category_decoding_letter_false_irrelevant", 
                                "category_decoding_face_orientation_irrelevant", "category_decoding_object_orientation_irrelevant", "category_decoding_letter_orientation_irrelevant", "category_decoding_false_orientation_irrelevant",
                                "category_decoding_faces_objects_relevant_no_fs", "category_decoding_faces_objects_irrelevant_no_fs", "category_decoding_letter_false_relevant_no_fs", "category_decoding_letter_false_irrelevant_no_fs", 
                                "category_decoding_face_orientation_irrelevant_no_fs", "category_decoding_object_orientation_irrelevant_no_fs", "category_decoding_letter_orientation_irrelevant_no_fs", "category_decoding_false_orientation_irrelevant_no_fs",
                                "category_decoding_face_orientation", "category_decoding_object_orientation", "category_decoding_letter_orientation", "category_decoding_false_orientation",
                                "category_decoding_face_orientation_irrelevant", "category_decoding_object_orientation_irrelevant", "category_decoding_letter_orientation_irrelevant", "category_decoding_false_orientation_irrelevant"
                                ]
    
        for analysis_name, analysis_parameters in param.analysis_parameters.items():
            if analysis_name not in valid_analysis_names:
                print('--- not performing analysis on profile: %s ---\n\t continuing...' % analysis_name)
                continue
        
            # append control analysis name to the analysis name
            analysis_name = analysis_name+"_control_iit_vs_iitgnw"
        
            save_path_results = path_generator(param.save_root,
                                               analysis=analysis_name,
                                               preprocessing_steps=param.preprocess_steps,
                                               fig=False, stats=True)
            save_path_fig = path_generator(param.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=param.preprocess_steps,
                                           fig=True)
            #%% load in al the subject data first
            ## The same data will be used across all analyses
            data_iit = []
            data_gnw = []
            for subject in subjects_list:
                # load in data for each roi
                for roi in ['iit', 'gnw']:
                    
                    # remove S_front_inf, since this ROI is a confound
                    rois = [x for x in param.rois[roi] if x.find('S_front_inf') == -1]
                    epochs, mni_coords = load_epochs(param.BIDS_root, analysis_parameters["signal"],
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
                                                    picks_roi=rois
                                                    )
                    if epochs is None:
                        continue
            
                    # preprocessing
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
                                
                    if roi == 'iit':
                        data_iit.append(epochs_data[idx, :, :])
                    elif roi == 'gnw':
                        data_gnw.append(epochs_data[idx, :, :])
                    
            data_iit = np.concatenate(data_iit, axis=1)
            data_gnw = np.concatenate(data_gnw, axis=1) 
            data_iit_gnw = np.concatenate((data_iit, data_gnw), axis=1)
            time = times
            info = epochs.metadata.iloc[idx]  # sorted
            
            #%% run the decoding analysis
            # ==== much of this is custom decoding (e.g. combine posterior probabilities across samples, and therefore does not rely much on utility functions === #
        
            # Get the classes:
            y = info[analysis_parameters['decoding_target']].values
            n_classes = len(np.unique( y ))
        
            # Setting the cross validation:
            if analysis_parameters['cross_validation_parameters']["n_folds"] == "leave_one_out":
                n_folds = len(y)
            else:
                n_folds = analysis_parameters['cross_validation_parameters']["n_folds"]
        
            repeats = 10 #analysis_parameters['classifier_parameters']['repeats']
    
            # Creating cross val iterator, use the same folds for each calibration method for equitability
            skf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats)
            # Getting the indices of the test and train sets from cross folder validation:
            cv_iter = list(skf.split(data_iit, y))
        
            #%% run a custom sliding estimator, using the same cv folds across time
            calibration_methods = ['none', 'sigmoid', 'isotonic', 'beta']
            for calibration_method in calibration_methods:
                
                # initiliaze classifier pipeline for each model (e.g., iit & gnw)
                clf = {}
                classifier_parameters = analysis_parameters["classifier_parameters"]
                for roi in ['iit', 'gnw', 'iit_gnw']:
                    data = eval("data_"+roi)
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
                    if calibration_method == 'none':
                        # Probability need to be set to True in order to obtain posterior probabilities
                        # however, calibration_method = "None" is kind of a misnomer, as setting probability True to return probabilities performs internal platt scaling calibration with default cv=3
                        clf_steps.append(svm.SVC(kernel='linear', class_weight='balanced', probability=True))
                    else:
                        clf_steps.append(svm.SVC(kernel='linear', class_weight='balanced'))
                
                    clf[roi] = make_pipeline(*clf_steps)
                
                # initialize storage
                decoding_scores_iit = np.empty((repeats*n_folds, len(times)))
                decoding_scores_iit_gnw = np.empty((repeats*n_folds, len(times)))
                decoding_scores_comb = np.empty((repeats*n_folds, len(times)))
                decoding_scores_comb_bayes = np.empty((repeats*n_folds, len(times)))
                proba_iit = np.zeros((repeats, len(y), n_classes, len(times)))*np.nan 
                proba_gnw = np.zeros((repeats, len(y), n_classes, len(times)))*np.nan     
                
                r = -1
                for ind, train_test_ind in enumerate(cv_iter):
                    if np.mod(ind, n_folds) == 0:
                        r += 1
                    y_train = y[train_test_ind[0]]
                    y_test = y[train_test_ind[1]]
                    for t, time in enumerate(times):
                        x_train_iit = data_iit[train_test_ind[0], :, t]
                        x_test_iit = data_iit[train_test_ind[1], :, t]
                    
                        x_train_gnw = data_gnw[train_test_ind[0], :, t]
                        x_test_gnw = data_gnw[train_test_ind[1], :, t]
                        
                        x_train_iit_gnw = data_iit_gnw[train_test_ind[0], :, t]
                        x_test_iit_gnw = data_iit_gnw[train_test_ind[1], :, t]
                    
                        # # original code w/o calibration
                        # # regular prediction for iit-alone
                        # mdl_iit = clf['iit'].fit(x_train_iit, y_train)
                        # mdl_gnw = clf['gnw'].fit(x_train_gnw, y_train)
                        
                        # y_pred = mdl_iit.predict(x_test_iit)
                        # decoding_scores_iit[ind,t] = balanced_accuracy_score(y_test,  y_pred )
                
                        # iit+gnw feature model
                        mdl_iit_gnw = clf['iit_gnw'].fit( x_train_iit_gnw, y_train)
                        
                        if calibration_method == 'none':
                            mdl_iit = clf['iit'].fit(x_train_iit, y_train)
                            mdl_gnw = clf['gnw'].fit(x_train_gnw, y_train)                            
                        else:
                            # train model using calibratedClassifierCV
                            mdl_iit = CalibratedClassifierCV( base_estimator=clf['iit'], cv=3, method=calibration_method)
                            mdl_iit.fit(x_train_iit, y_train)
                            mdl_gnw = CalibratedClassifierCV( base_estimator=clf['gnw'], cv=3, method=calibration_method)
                            mdl_gnw.fit(x_train_gnw, y_train)
    
                            # #using sklearn 
                            # mdl_iit = CCCV( clf['iit'], cv=3, method='sigmoid')
                            # mdl_iit.fit(x_train_iit, y_train)
                            # mdl_gnw = CCCV( clf['gnw'], cv=3, method='sigmoid')
                            # mdl_gnw.fit(x_train_gnw, y_train)
                        
                        # iit-only
                        y_pred = mdl_iit.predict(x_test_iit)
                        decoding_scores_iit[ind,t] = balanced_accuracy_score(y_test,  y_pred )
                        
                        # iit+gnw feature model
                        y_pred = mdl_iit_gnw.predict( x_test_iit_gnw )
                        decoding_scores_iit_gnw[ind,t] = balanced_accuracy_score(y_test,  y_pred )
                        
                        # for iit+gnw model, get posterior probabilities, sum them, then norm the result (softmax), and predict the label
                        mdl_prob_iit = mdl_iit.predict_proba( x_test_iit )
                        mdl_prob_gnw = mdl_gnw.predict_proba( x_test_gnw )
    
                        # store the probabilities
                        proba_iit[r, train_test_ind[1], :, t] = mdl_prob_iit
                        proba_gnw[r, train_test_ind[1], :, t] = mdl_prob_gnw
                        
                        psum = mdl_prob_iit+mdl_prob_gnw
                        softmx = np.exp(psum) / np.expand_dims( np.sum(np.exp(psum),1),1)
                        ypred_combined = np.argmax( softmx, 1)
                        decoding_scores_comb[ind,t] = balanced_accuracy_score(y_test, mdl_iit.classes_[ ypred_combined ] )
    
                        # p_post = 1/( 1 + exp(log((1-Pgnw)/Pgnw) - log(Piit/(1-Piit)) ) )
                        Piit = mdl_prob_iit
                        Pgnw = mdl_prob_gnw
                        bayes_int = 1/( 1 + np.exp(np.log((1-Pgnw)/Pgnw) - np.log(Piit/(1-Piit)) ) )
                        ypred_combined = np.argmax( bayes_int, 1)
                        decoding_scores_comb_bayes[ind,t] = balanced_accuracy_score(y_test, mdl_iit.classes_[ ypred_combined ] )
                
                
                
                #%% analyze calibration performance all time bins between 0-0.5
                idx_ = np.arange(15, 36, dtype=int)
                idx_ = np.arange(0, proba_iit.shape[-1], dtype=int)
                idx_ = np.where( (times > 0) & (times < 0.5))[0]
    
                plt.figure();
                plt.plot([0, 1], [0, 1], linestyle='--')
    
                tmp = np.squeeze(np.mean(proba_iit[:, :, 1, idx_],0))
                tmp = tmp.T.reshape(-1)
                yy = np.tile(label_binarize(y, classes=np.unique(y)).T, len(idx_)) 
                fop, mpv = calibration_curve(  yy[0, :], tmp, n_bins=20)            
                plt.plot(mpv, fop, label=( ('IIT: brier score: %2.2f') % (brier_score_loss(yy[0, :], tmp)) ) )
                
                tmp = np.squeeze(np.mean(proba_gnw[:, :, 1, idx_],0))
                tmp = tmp.T.reshape(-1)
                yy = np.tile(label_binarize(y, classes=np.unique(y)).T, len(idx_)) 
                fop, mpv = calibration_curve(  yy[0, :], tmp, n_bins=20)            
                plt.plot(mpv, fop, label=( ('GNW: brier score: %2.2f') % (brier_score_loss(yy[0, :], tmp)) ) )
                
                plt.legend()
                plt.xlabel('Mean predicted probability (Positive class: 1)')
                plt.ylabel('Fraction of positives (Positive class: 1)')
                plt.title(calibration_method)
                # plt.savefig('/Users/simonhenin/Desktop/calibration-fit.png') 
                file_name = Path(save_path_fig, param.files_prefix + "_decoding_calibration_fit_"+calibration_method+".png")
                plt.savefig(file_name, dpi=150)
                
                #%% plot the result
                fig = plt.figure(figsize=(14,6))
                ax = plt.subplot(211)
                ax.plot( times, decoding_scores_iit.mean(0), label='IIT-only'); 
                ci = 1.96 * decoding_scores_iit.std(0) / np.sqrt(np.size(decoding_scores_iit, 0))
                ax.fill_between(times, decoding_scores_iit.mean(0) - ci, decoding_scores_iit.mean(0) + ci, alpha=0.5)
                ax.plot( times, decoding_scores_comb.mean(0), label='IIT+GNW'); 
                ci = 1.96 * decoding_scores_comb.std(0) / np.sqrt(np.size(decoding_scores_comb, 0))
                ax.fill_between(times, decoding_scores_comb.mean(0) - ci, decoding_scores_comb.mean(0) + ci, alpha=0.5)
            
                # compute variance corrected p-values across the whole time window 
                p_values_iit_v_comb = np.empty((len(times),))       # (iit vs. comb)
                p_values_iit_v_iit_gnw = np.empty((len(times),))    # iit vs. iit+gnw
                # degrees of freedom and train/test split counts
                df = decoding_scores_iit.shape[0] - 1
                n_train = len( cv_iter[0][0] )
                n_test = len( cv_iter[0][1] )
                for t, time in  enumerate(times):
                    # perform a single-tailed test of the hypothesis that combined model is better than IIT-alone
                    mdl_diff = decoding_scores_comb[:,t] - decoding_scores_iit[:,t]
                    t_stat, p_values_iit_v_comb[t] = compute_corrected_ttest(mdl_diff, df, n_train, n_test)
                    
                    mdl_diff = decoding_scores_iit_gnw[:,t] - decoding_scores_iit[:,t]
                    t_stat, p_values_iit_v_iit_gnw[t] = compute_corrected_ttest(mdl_diff, df, n_train, n_test)
                
                sig_mask = (p_values_iit_v_comb < 0.05)
                ax.plot(times[sig_mask], np.ones_like(sig_mask)[sig_mask] * np.min(decoding_scores_iit.mean(0)), 'ko')
            
                ax.axhline(1 / len(np.unique(y)), color='k', linestyle='--')
                ax.legend()
                ax.set_ylabel('ACC')
            
            
                # bayesian analysis
                p_values_bayes = np.empty((len(times),2))
                for t, time in  enumerate(times):
                    # perform a single-tailed test of the hypothesis that combined model (IIT+GNW) is better than IIT-alone
                    mdl_diff = decoding_scores_comb[:,t] - decoding_scores_iit[:,t]
                    # initialize random variable
                    t_post = stats.t(
                        df, loc=np.mean(mdl_diff), scale=corrected_std(mdl_diff, n_train, n_test)
                    )
                    better_prob = 1 - t_post.cdf(0)
                    p_values_bayes[t,0] = better_prob
                    p_values_bayes[t,1] = 1-better_prob
                
                ax = plt.subplot(212)
                ax.plot(times, p_values_bayes[:,1], label='IIT-only > IIT+GNW')  
                ax.plot(times, p_values_bayes[:,0], label='IIT+GNW > IIT-only')
                ax.legend(loc='upper right')
                ax.set_ylabel('Probability');
                
    
                #%%
                file_name = Path(save_path_results, param.files_prefix + "_decoding_"+calibration_method+".npz")
                np.savez(file_name, times=times, decoding_scores_iit=decoding_scores_iit, decoding_scores_iit_gnw=decoding_scores_iit_gnw,
                         decoding_scores_comb=decoding_scores_comb, p_values_iit_v_comb=p_values_iit_v_comb, p_values_bayes=p_values_bayes, p_values_iit_v_iit_gnw=p_values_iit_v_iit_gnw, cv_iter=cv_iter, y=y, proba_iit=proba_iit, proba_gnw=proba_gnw)
            
                file_name = Path(save_path_fig, param.files_prefix + "_decoding_"+calibration_method+".png")
                plt.savefig(file_name, dpi=150)

if __name__ == "__main__":
    # run the analysis on al available subjects
    decoding_analysis_iit_vs_iitgnw(None, save_folder="super")





