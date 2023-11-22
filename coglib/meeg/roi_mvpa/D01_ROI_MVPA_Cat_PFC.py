
"""
====================
D01. Decoding for MEG on source space of ROI
Category decoding
control analysis,
compare decoding performance with vs without PFC region.
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CCD: Cross Condition Decoding
classifier: SVM (linear)
feature: spatial pattern (S)

compare the decodeing performance of postior region with or without prefrontal region

"""
import warnings
import os.path as op
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import matplotlib as mpl

import argparse





from skimage.measure import block_reduce

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Vectorizer,StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score


# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold



from scipy.ndimage import gaussian_filter1d

import matplotlib.patheffects as path_effects


#from config import no_eeg_sbj
#from config import site_id, subject_id, file_names, visit_id, data_path, out_path
# from config import l_freq, h_freq, sfreq
# from config import (bids_root, tmin, tmax)
import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root, plot_param

from D_MEG_function import set_path_ROI_MVPA, ATdata, sensor_data_for_ROI_MVPA
from D_MEG_function import source_data_for_ROI_MVPA, sub_ROI_for_ROI_MVPA

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


####if need pop-up figures
# %matplotlib qt5
#mpl.use('Qt5Agg')

parser=argparse.ArgumentParser()
parser.add_argument('--sub',type=str,default='CA101',help='subject_id')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--cT',type=str,nargs='*', default=['500ms','1000ms','1500ms'], help='condition in Time duration')
parser.add_argument('--cC',type=str,nargs='*', default=['FO'],
                    help='selected decoding category, FO for face and object, LF for letter and false,'
                         'F for face ,O for object, L for letter, FA for false')
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
parser.add_argument('--out_fw',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward',
                    help='Path to the forward (derivative) directory')
parser.add_argument('--nF',
                    type=int,
                    default=30,
                    help='number of feature selected for source decoding')
parser.add_argument('--nT',
                    type=int,
                    default=5,
                    help='number of trial averaged for source decoding')
parser.add_argument('--nPCA',
                    type=float,
                    default=0.95,
                    help='percentile of PCA selected for source decoding')
# parser.add_argument('--coreg_path',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/coreg',
#                     help='Path to the coreg (derivative) directory')


opt = parser.parse_args()
con_C = opt.cC
con_D = opt.cD
con_T = opt.cT
select_F = opt.nF
n_trials = opt.nT
nPCA = opt.nPCA


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path

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

# Color parameters:
cmap = "RdYlBu_r"


def Category_PFC(fpath_fw,rank,common_cov,sub_info,surf_label_list,
                 epochs_rs,conditions_C,conditions_D,conditions_T,task_info):
    #get data
    stcs_PFC = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[0])
    stcs_IIT = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[1])
    stcs_IITPFC = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[2])




    # setup SVM classifier
    select_Fn=[30,30,60]
    clf={}
    for n,roi in enumerate(['PFC', 'IIT','IITPFC']):
        clf[roi] = make_pipeline(Vectorizer(),
                                 StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
                                 SelectKBest(f_classif,k=select_Fn[n]),
                                 sklearn.svm.SVC(kernel='linear',probability=True))   #LogisticRegression(),

    # # The scorers can be either one of the predefined metric strings or a scorer
    # # callable, like the one returned by make_scorer
    # #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # # score methods could be AUC or Accuracy
    # # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    #sliding = SlidingEstimator(clf, scoring=make_scorer(accuracy_score), n_jobs=-1)


    print(' Creating evoked datasets')

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Category'] == conditions_C[0]] = 1  # face
    temp[epochs_rs.metadata['Category'] == conditions_C[1]] = 2 # object

    times = epochs_rs.times

    y = temp
    X_PFC=np.array([stc.data for stc in stcs_PFC])
    X_IIT=np.array([stc.data for stc in stcs_IIT])
    X_IITPFC=np.array([stc.data for stc in stcs_IITPFC])

    # cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
    # #         # Find indices of Irrelevant trials
    # cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]

    wcd=dict()

    con_index=np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0] # only analysis Irrelevant condition
    group_x_PFC=X_PFC[con_index]
    group_x_IIT=X_IIT[con_index]
    group_x_IITPFC=X_IITPFC[con_index]
    group_y=y[con_index]


    scores_per_IIT=np.zeros([100,len(times)])
    scores_per_IITPFC=np.zeros([100,len(times)])
    # scores_per_comb=np.zeros([100,len(times)])
    # scores_per_comb_bayes=np.zeros([100,len(times)])
    for num_per in range(100):
    # do the average trial
        new_x_PFC = []
        new_x_IIT = []
        new_x_IITPFC = []
        new_y = []
        for label in range(2):
            #block_size
            #array_like or int
            #Array containing down-sampling integer factor along each axis. Default block_size is 2.

            # funccallable
            # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.

            # cvalfloat
            # Constant padding value if image is not perfectly divisible by the block size.

            #PFC
            data_PFC = group_x_PFC[np.where(group_y == label+1)]
            data_PFC = np.take(data_PFC, np.random.permutation(data_PFC.shape[0]), axis=0)
            avg_x_PFC = block_reduce(data_PFC, block_size=tuple([n_trials, *[1] * len(data_PFC.shape[1:])]),
                                  func=np.nanmean, cval=np.nan)
            new_x_PFC.append(avg_x_PFC)

            #IIT
            data_IIT = group_x_IIT[np.where(group_y == label+1)]
            data_IIT = np.take(data_IIT, np.random.permutation(data_IIT.shape[0]), axis=0)
            avg_x_IIT = block_reduce(data_IIT, block_size=tuple([n_trials, *[1] * len(data_IIT.shape[1:])]),
                                  func=np.nanmean, cval=np.nan)
            new_x_IIT.append(avg_x_IIT)


            #IITPFC
            data_IITPFC = group_x_IITPFC[np.where(group_y == label+1)]
            data_IITPFC = np.take(data_IITPFC, np.random.permutation(data_IITPFC.shape[0]), axis=0)
            avg_x_IITPFC = block_reduce(data_IITPFC, block_size=tuple([n_trials, *[1] * len(data_IITPFC.shape[1:])]),
                                  func=np.nanmean, cval=np.nan)
            new_x_IITPFC.append(avg_x_IITPFC)


            # Now generating the labels and group:
            new_y += [label] * avg_x_PFC.shape[0]

        new_x_PFC = np.concatenate((new_x_PFC[0],new_x_PFC[1]),axis=0)
        new_x_IIT = np.concatenate((new_x_IIT[0],new_x_IIT[1]),axis=0)
        new_x_IITPFC = np.concatenate((new_x_IITPFC[0],new_x_IITPFC[1]),axis=0)
        new_y = np.array(new_y)

        # average temporal feature (5 point average)
        new_x_PFC=ATdata(new_x_PFC)
        new_x_IIT=ATdata(new_x_IIT)
        new_x_IITPFC=ATdata(new_x_IITPFC)

        skf = StratifiedKFold(n_splits=5)
        # Getting the indices of the test and train sets from cross folder validation:
        cv_index = list(skf.split(new_x_PFC, new_y))


        # n_classes=2
        n_folds=5
        # initialize storage
        decoding_scores_IIT = np.empty((n_folds, len(times)))
        decoding_scores_IITPFC = np.empty((n_folds, len(times)))
        # decoding_scores_comb = np.empty((n_folds, len(times)))
        # decoding_scores_comb_bayes = np.empty((n_folds, len(times)))
        # proba_IIT = np.zeros((len(new_y), n_classes, len(times)))*np.nan
        # proba_PFC = np.zeros((len(new_y), n_classes, len(times)))*np.nan




        for ind, train_test_ind in enumerate(cv_index):
            y_train = new_y[train_test_ind[0]]
            y_test = new_y[train_test_ind[1]]
            for t, time in enumerate(times):
                x_train_PFC = new_x_PFC[train_test_ind[0], :, t]
                x_test_PFC = new_x_PFC[train_test_ind[1], :, t]

                x_train_IIT = new_x_IIT[train_test_ind[0], :, t]
                x_test_IIT = new_x_IIT[train_test_ind[1], :, t]

                x_train_IITPFC = new_x_IITPFC[train_test_ind[0], :, t]
                x_test_IITPFC = new_x_IITPFC[train_test_ind[1], :, t]

                # # original code w/o calibration
                # # regular prediction for iit-alone
                # mdl_iit = clf['iit'].fit(x_train_iit, y_train)
                # mdl_gnw = clf['gnw'].fit(x_train_gnw, y_train)

                # y_pred = mdl_iit.predict(x_test_iit)
                # decoding_scores_iit[ind,t] = balanced_accuracy_score(y_test,  y_pred )

                # iit+gnw feature model
                mdl_IITPFC = clf['IITPFC'].fit(x_train_IITPFC, y_train)

                mdl_IIT = clf['IIT'].fit(x_train_IIT, y_train)
                mdl_PFC = clf['PFC'].fit(x_train_PFC, y_train)

                # iit-only
                y_pred = mdl_IIT.predict(x_test_IIT)
                decoding_scores_IIT[ind,t] = accuracy_score(y_test,  y_pred )

                # iit+pfc feature model
                y_pred = mdl_IITPFC.predict( x_test_IITPFC )
                decoding_scores_IITPFC[ind,t] = accuracy_score(y_test,  y_pred )

                # # for iit+pfc model, get posterior probabilities, sum them, then norm the result (softmax), and predict the label
                # mdl_prob_IIT = mdl_IIT.predict_proba( x_test_IIT )
                # mdl_prob_PFC = mdl_PFC.predict_proba( x_test_PFC )

                # # store the probabilities
                # proba_IIT[train_test_ind[1], :, t] = mdl_prob_IIT
                # proba_PFC[train_test_ind[1], :, t] = mdl_prob_PFC

                # psum = mdl_prob_IIT+mdl_prob_PFC
                # softmx = np.exp(psum) / np.expand_dims( np.sum(np.exp(psum),1),1)
                # ypred_combined = np.argmax( softmx, 1)
                # decoding_scores_comb[ind,t] = accuracy_score(y_test, mdl_IIT.classes_[ ypred_combined ] )

                # # p_post = 1/( 1 + exp(log((1-Pgnw)/Pgnw) - log(Piit/(1-Piit)) ) )
                # PIIT = mdl_prob_IIT
                # PPFC = mdl_prob_PFC
                # bayes_int = 1/( 1 + np.exp(np.log((1-PPFC)/PPFC) - np.log(PIIT/(1-PPFC)) ) )
                # ypred_combined = np.argmax( bayes_int, 1)
                # decoding_scores_comb_bayes[ind,t] = accuracy_score(y_test, mdl_IIT.classes_[ ypred_combined ] )





        scores_per_IIT[num_per,:]=np.mean(decoding_scores_IIT, axis=0)
        scores_per_IITPFC[num_per,:]=np.mean(decoding_scores_IITPFC, axis=0)
        # scores_per_comb[num_per,:]=np.mean(decoding_scores_comb, axis=0)
        # scores_per_comb_bayes[num_per,:]=np.mean(decoding_scores_comb_bayes, axis=0)

    wcd['IIT']=np.mean(scores_per_IIT, axis=0)
    wcd['IITPFC_f']=np.mean(scores_per_IITPFC, axis=0) # feature combine score
    # wcd['IITPFC_m']=np.mean(scores_per_comb, axis=0) # model combine score
    # wcd['IITPFC_m_bayes']=np.mean(scores_per_comb_bayes, axis=0)  # model combine score with bayes methods







    return wcd




# =============================================================================
# RUN
# =============================================================================


# run roi decoding analysis

if __name__ == "__main__":

    #opt INFO

    # subject_id = 'CB085'
    #
    # visit_id = 'V1'
    # space = 'surface'
    #

    # analysis info

    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']


    analysis_name='Cat_PFC'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    roi_data_root, roi_figure_root, roi_code_root = set_path_ROI_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    # 2 Get Sub ROI
    surf_label_list, ROI_Name = sub_ROI_for_ROI_MVPA(fpath_fs, subject_id,analysis_name)



    # 3 prepare the sensor data
    epochs_rs, \
    rank, common_cov, \
    conditions_C, conditions_D, conditions_T, task_info = sensor_data_for_ROI_MVPA(fpath_epo,
                                                                                   sub_info,
                                                                                   con_T,
                                                                                   con_C,
                                                                                   con_D)

    roi_ccd_acc = dict()
    #roi_ccd_auc = dict()
    roi_wcd_acc = dict()


    fname_fig = op.join(roi_figure_root,sub_info  + task_info + '_'  + "IITPFC_acc_WCD" + '.png')

    wcd_acc=Category_PFC(fpath_fw,rank,common_cov,sub_info,surf_label_list,
                         epochs_rs,conditions_C,conditions_D,conditions_T,task_info)



    fname_data=op.join(roi_data_root, sub_info + '_' + task_info +"_IITPFC_data_Cat" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(wcd_acc,fw)
    fw.close()



    fig, ax = plt.subplots(1)
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Ti_name in wcd_acc.items():
        ax.plot(t, gaussian_filter1d(Ti_name,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title('WCD_IIT_PFC')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig)


# Save code
#    shutil.copy(__file__, roi_code_root)
