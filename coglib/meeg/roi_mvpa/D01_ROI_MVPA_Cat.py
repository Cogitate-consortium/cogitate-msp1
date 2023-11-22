
"""
====================
D01. Decoding for MEG on source space of ROI
Category decoding
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CCD: Cross Condition Decoding
classifier: SVM (linear)
feature: spatial pattern (S)

feature selection methods test

"""

import os.path as op

import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np

import argparse


from mne.decoding import (Vectorizer, SlidingEstimator, cross_val_multiscore)
# import a linear classifier from mne.decoding
from mne.decoding import LinearModel

from skimage.measure import block_reduce

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import SelectPercentile, chi2
#from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold



from scipy.ndimage import gaussian_filter1d
import matplotlib.patheffects as path_effects


from config import bids_root

from D_MEG_function import set_path_ROI_MVPA, ATdata,sensor_data_for_ROI_MVPA
from D_MEG_function import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA

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


    # Now we define a function to decoding condition for one subject
    # Category_CCD, train on condition A, test on condition B
def Category_CCD(epochs_rs,stcs,conditions_C,conditions_D,select_F,n_trials,roi_name,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #PCA(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    sliding = SlidingEstimator(clf, scoring=score_methods, n_jobs=1)


    print(' Creating evoked datasets')

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Category'] == conditions_C[0]] = 1  # face
    temp[epochs_rs.metadata['Category'] == conditions_C[1]] = 2 # object

    y = temp
    X=np.array([stc.data for stc in stcs])

    cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
            # Find indices of Irrelevant trials
    cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]



    # # Run cross-validated decoding analyses:
    # scores_a = cross_val_multiscore(sliding,X=X[cond_a], y=y[cond_a],cv=5,n_jobs=-1)
    # # Run cross-validated decoding analyses:
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=-1)

    ccd=dict()
    cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
            # Find indices of Irrelevant trials
    cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]

    group_xa=X[cond_a]
    group_ya=y[cond_a]
    group_xb=X[cond_b]
    group_yb=y[cond_b]

    scores_ab_per=np.zeros([100,group_xa.shape[2]])
    scores_ba_per=np.zeros([100,group_xb.shape[2]])
    for num_per in range(100):
        # do the average trial
        new_xa = []
        new_ya = []
        new_xb = []
        new_yb = []
        for label in range(2):
            # Extract the data:
            data_a = group_xa[np.where(group_ya == label+1)]
            data_a = np.take(data_a, np.random.permutation(data_a.shape[0]), axis=0)
            avg_xa = block_reduce(data_a, block_size=tuple([n_trials, *[1] * len(data_a.shape[1:])]),
                                 func=np.nanmean, cval=np.nan)
            #block_size
            #array_like or int
            #Array containing down-sampling integer factor along each axis. Default block_size is 2.

            # funccallable
            # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.

            # cvalfloat
            # Constant padding value if image is not perfectly divisible by the block size.

            # Now generating the labels and group:
            new_xa.append(avg_xa)
            new_ya += [label] * avg_xa.shape[0]

            # Extract the data:
            data_b = group_xb[np.where(group_yb == label+1)]
            data_b = np.take(data_b, np.random.permutation(data_b.shape[0]), axis=0)
            avg_xb = block_reduce(data_b, block_size=tuple([n_trials, *[1] * len(data_b.shape[1:])]),
                                 func=np.nanmean, cval=np.nan)
            #block_size
            #array_like or int
            #Array containing down-sampling integer factor along each axis. Default block_size is 2.

            # funccallable
            # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.

            # cvalfloat
            # Constant padding value if image is not perfectly divisible by the block size.

            # Now generating the labels and group:
            new_xb.append(avg_xb)
            new_yb += [label] * avg_xb.shape[0]

        new_xa = np.concatenate((new_xa[0],new_xa[1]),axis=0)
        new_ya = np.array(new_ya)

        # average temporal feature (5 point average)
        new_xa=ATdata(new_xa)

        new_xb = np.concatenate((new_xb[0],new_xb[1]),axis=0)
        new_yb = np.array(new_yb)

        # average temporal feature (5 point average)
        new_xb=ATdata(new_xb)

        # First: train condition a (cond_a) and Test on condition b (cond_b) cross condition decoding
        # Fit
        sliding.fit(X=new_xa, y=new_ya)
        # Test
        scores_ab = sliding.score(X=new_xb, y=new_yb)


        scores_ab_per[num_per,:]=scores_ab

        # Then: train condition b (cond_b) and Test on condition a (cond_a) cross condition decoding
        # Fit
        sliding.fit(X=new_xb, y=new_yb)
        # Test
        scores_ba = sliding.score(X=new_xa, y=new_ya)


        scores_ba_per[num_per,:]=scores_ba




    # ccd['IR'] = np.mean(scores_a, axis=0)
    # ccd['RE'] = np.mean(scores_b, axis=0)
    ccd['IR2RE'] = np.mean(scores_ab_per, axis=0)
    ccd['RE2IR'] = np.mean(scores_ba_per, axis=0)



    fig, ax = plt.subplots(1)
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Sccd in ccd.items():
        ax.plot(t, gaussian_filter1d(Sccd,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title(f'CCD_ {roi_name}')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    mne.viz.tight_layout()
    # Save figure
    fig.savefig(fname_fig)

    return ccd

def Category_WCD(epochs_rs,stcs,
                 conditions_C,conditions_D,
                 select_F,
                 n_trials,
 #                nPCA,
                 roi_name,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}#

    sliding = SlidingEstimator(clf, scoring=score_methods, n_jobs=1)


    print(' Creating evoked datasets')

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Category'] == conditions_C[0]] = 1  # face
    temp[epochs_rs.metadata['Category'] == conditions_C[1]] = 2 # object

    y = temp
    X=np.array([stc.data for stc in stcs])

    # cond_a = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[0])[0]
    # #         # Find indices of Irrelevant trials
    # cond_b = np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[1])[0]

    wcd=dict()
    for condi in range(2):
        con_index=np.where(epochs_rs.metadata['Task_relevance'] == conditions_D[condi])[0]
        group_x=X[con_index]
        group_y=y[con_index]

        scores_per=np.zeros([100,group_x.shape[2]])
        for num_per in range(100):
        # do the average trial
            new_x = []
            new_y = []
            for label in range(2):
                # Extract the data:
                data = group_x[np.where(group_y == label+1)]
                data = np.take(data, np.random.permutation(data.shape[0]), axis=0)
                avg_x = block_reduce(data, block_size=tuple([n_trials, *[1] * len(data.shape[1:])]),
                                     func=np.nanmean, cval=np.nan)
                #block_size
                #array_like or int
                #Array containing down-sampling integer factor along each axis. Default block_size is 2.

                # funccallable
                # Function object which is used to calculate the return value for each local block. This function must implement an axis parameter. Primary functions are numpy.sum, numpy.min, numpy.max, numpy.mean and numpy.median. See also func_kwargs.

                # cvalfloat
                # Constant padding value if image is not perfectly divisible by the block size.

                # Now generating the labels and group:
                new_x.append(avg_x)
                new_y += [label] * avg_x.shape[0]

            new_x = np.concatenate((new_x[0],new_x[1]),axis=0)
            new_y = np.array(new_y)

            # average temporal feature (5 point average)
            new_x=ATdata(new_x)

            scores= cross_val_multiscore(sliding, X=new_x, y=new_y, cv=5, n_jobs=1)
            scores_per[num_per,:]=np.mean(scores, axis=0)

        wcd[conditions_D[condi]]=np.mean(scores_per, axis=0)



    # wcd=dict()
    # scores_a= cross_val_multiscore(sliding, X=X[cond_a], y=y[cond_a], cv=5, n_jobs=1)
    # wcd[conditions_D[0]]=np.mean(scores_a, axis=0)
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=1)
    # wcd[conditions_D[1]] = np.mean(scores_b, axis=0)

    # pattern = dict()
    # pattern['IR'] = coef_a
    # pattern['RE'] = coef_b


    fig, ax = plt.subplots(1)
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    for condi, Ti_name in wcd.items():
        ax.plot(t, gaussian_filter1d(Ti_name,sigma=4), linewidth=1, label=str(condi), path_effects=pe)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0, color='k')
    ax.legend(loc='upper right')
    ax.set_title(f'WCD_ {roi_name}')
    ax.set(xlabel='Time(ms)', ylabel='decoding score')
    mne.viz.tight_layout()
    # Save figure

    fig.savefig(fname_fig)

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


    analysis_name='Cat'

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
    #roi_wcd_auc = dict()


    for nroi, roi_name in enumerate(ROI_Name):

        # 4 Get Source Data for each ROI
        stcs = []
        stcs = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[nroi])


        fname_fig_acc = op.join(roi_figure_root,
                            sub_info + task_info + '_'+ roi_name
                            + "_acc_CCD" + '.png')


        score_methods=make_scorer(accuracy_score)
        ccd_acc = Category_CCD(epochs_rs, stcs,
                                        conditions_C, conditions_D,
                                        select_F,
                                        n_trials,
                                        # nPCA,
                                        roi_name, score_methods,
                                        fname_fig_acc)

        roi_ccd_acc[roi_name] = ccd_acc


        ### WCD


        fname_fig_acc = op.join(roi_figure_root,
                            sub_info  + task_info + '_' + roi_name + "_acc_WCD" + '.png')


        score_methods=make_scorer(accuracy_score)
        wcd_acc= Category_WCD(epochs_rs, stcs,
                              conditions_C, conditions_D,
                              select_F,
                              n_trials,
                              # nPCA,
                              roi_name, score_methods,
                              fname_fig_acc)

        roi_wcd_acc[roi_name] = wcd_acc




    roi_data=dict()
    roi_data['ccd_acc']=roi_ccd_acc

    roi_data['wcd_acc']=roi_wcd_acc



    fname_data=op.join(roi_data_root, sub_info + '_' + task_info +"_ROIs_data_Cat" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(roi_data,fw)
    fw.close()


# Save code
#    shutil.copy(__file__, roi_code_root)
