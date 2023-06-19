
"""
====================
D04. Decoding for MEG on source space of ROI : genelaization across time (GAT)
Orienation Decoding
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

import numpy as np
import matplotlib as mpl

import argparse



from mne.decoding import (Vectorizer, cross_val_multiscore)
# import a linear classifier from mne.decoding
from mne.decoding import LinearModel
from mne.decoding import GeneralizingEstimator


from skimage.measure import block_reduce

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold




from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects



from config import bids_root

from D_MEG_function import set_path_ROI_MVPA, ATdata,sensor_data_for_ROI_MVPA
from D_MEG_function import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA

####if need pop-up figures
# %matplotlib qt5
#mpl.use('Qt5Agg')

parser=argparse.ArgumentParser()
parser.add_argument('--sub',type=str,default='SA101',help='subject_id')
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




def Orientation_CTWCD(epochs_rs,stcs,conditions_C,select_F,n_trials,roi_name,score_methods,fname_fig):
    # setup SVM classifier
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        SelectKBest(f_classif,k=select_F),
        #SelectPercentile(chi2,k=select_p),
        #PCA(n_components=nPCA),
        LinearModel(sklearn.svm.SVC(
            kernel='linear',class_weight='balanced')))   #LogisticRegression(),

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    #scoring = {"Accuracy": make_scorer(accuracy_score)}#"AUC": "roc_auc",
    # score methods could be AUC or Accuracy
    # {"AUC": "roc_auc","Accuracy": }#
    # For multivariable decoding(e,g, 1,2,3, could not use roc_auc),
    # deal with unbalanced trial number, score should use make_scorer(balanced_accuracy_score)

    sliding = GeneralizingEstimator(clf, scoring=score_methods, n_jobs=-1)


    print(' Creating evoked datasets')

    conditions_O = ['Center', 'Left', 'Right']

    temp = epochs_rs.events[:, 2]
    temp[epochs_rs.metadata['Orientation'] == conditions_O[0]] = 1  # center, straight
    temp[epochs_rs.metadata['Orientation'] == conditions_O[1]] = 2  # left,side orientation
    temp[epochs_rs.metadata['Orientation'] == conditions_O[2]] = 3  # right, also side orientation


    y = temp
    X=np.array([stc.data for stc in stcs])

    # cond_a = np.where(epochs_rs.metadata['Category'] == conditions_C[0])[0]
    #         # Find indices of Object trials
    # cond_b = np.where(epochs_rs.metadata['Category'] == conditions_C[1])[0]

    # # # Run cross-validated decoding analyses:
    # scores_a = cross_val_multiscore(sliding,X=X[cond_a], y=y[cond_a],cv=5,n_jobs=1)
    # # # Run cross-validated decoding analyses:
    # scores_b = cross_val_multiscore(sliding, X=X[cond_b], y=y[cond_b], cv=5, n_jobs=1)

    # #
    # wcd = dict()
    # # ccd['IR'] = np.mean(scores_a, axis=0)
    # # ccd['RE'] = np.mean(scores_b, axis=0)
    # wcd[conditions_C[0]] = np.mean(scores_a, axis=0)
    # wcd[conditions_C[1]] = np.mean(scores_b, axis=0)
    
    ctwcd=dict()
    for condi in range(len(conditions_C)):
        con_index=np.where(epochs_rs.metadata['Category'] == conditions_C[condi])[0]
        group_x=X[con_index]
        group_y=y[con_index]
        
        scores_per=np.zeros([100,group_x.shape[2],group_x.shape[2]])
        for num_per in range(100):
            # do the average trial
            new_x = []
            new_y = []
            for label in range(len(conditions_O)):
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

            new_x = np.concatenate((new_x[0],new_x[1],new_x[2]),axis=0)
            new_y = np.array(new_y)
            
            # average temporal feature (5 point average)
            new_x=ATdata(new_x)
            
            scores= cross_val_multiscore(sliding, X=new_x, y=new_y, cv=5, n_jobs=-1)
            scores_per[num_per,:,:]=np.mean(scores, axis=0)
            
        ctwcd[conditions_C[condi]]=np.mean(scores_per, axis=0)       
    
    

    
    fig, axes = plt.subplots(1)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    fig.suptitle('ORI CTWCD')
    
    t = 1e3 * epochs_rs.times
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    cmap = mpl.cm.jet
    vmin = 0.5
    vmax = 0.7
    bounds = np.linspace(vmin, vmax, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # diff setting
    vmind = -0.15
    vmaxd = 0.15
    boundsd = np.linspace(vmind, vmaxd, 11)
    normd = mpl.colors.BoundaryNorm(boundsd, cmap.N)
    #plot
    im = axes.imshow(gaussian_filter(ctwcd[conditions_C[0]],sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                   extent=epochs_rs.times[[0, -1, 0, -1]], vmin=vmin, vmax=vmax)
    axes.set_xlabel('Testing Time (s)')
    axes.set_ylabel('Training Time (s)')
    axes.axvline(0, color='k')
    axes.axhline(0, color='k')
    axes.axline((0, 0), slope=1, color='k')
    plt.colorbar(im, ax=axes,fraction=0.03, pad=0.05)
    
    #fname_fig = op.join('/home/user/S10/Cogitate/HPC/mvpa/ROI/figure/',ROI_Name[nroi] +"_CCD" + '.png')
    fig.savefig(fname_fig)

    return ctwcd
# =============================================================================
# RUN
# =============================================================================


# run roi decoding analysis

if __name__ == "__main__":
    
    #opt INFO
    
    # subject_id = 'SB085'
    #
    # visit_id = 'V1'
    # space = 'surface'
    #

    # analysis info
    
    # con_C = ['face']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']

    analysis_name='GAT_Ori'

    # 1 Set Path
    sub_info, \
    fpath_epo, fpath_fw, fpath_fs, \
    roi_data_root, roi_figure_root, roi_code_root = set_path_ROI_MVPA(bids_root,
                                                                      subject_id,
                                                                      visit_id,
                                                                      analysis_name)

    # 2 Get Sub ROI
    surf_label_list, ROI_Name = sub_ROI_for_ROI_MVPA(fpath_fs, subject_id, analysis_name)

    # 3 prepare the sensor data
    epochs_rs, \
    rank, common_cov, \
    conditions_C, conditions_D, conditions_T, task_info = sensor_data_for_ROI_MVPA(fpath_epo,
                                                                                   sub_info,
                                                                                   con_T,
                                                                                   con_C,
                                                                                   con_D)

    roi_ctwcd_ori_acc = dict()

    for nroi, roi_name in enumerate(ROI_Name):

        # 4 Get Source Data for each ROI
        stcs = []
        stcs = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, 
                                        rank, common_cov, 
                                        sub_info, surf_label_list[nroi])
        
       
        ### wcd_orientation
        #1 scoring methods with balanced accuracy score
        fname_fig_acc = op.join(roi_figure_root, 
                            sub_info  + task_info + '_' + roi_name + "_acc_CTWCD_ori" + '.png')

        
        score_methods=make_scorer(balanced_accuracy_score)
        ctwcd_ori_acc= Orientation_CTWCD(epochs_rs, stcs, 
                                      conditions_C,
                                      select_F,
                                      n_trials,
                                      # nPCA,
                                      roi_name, score_methods,
                                      fname_fig_acc)

        roi_ctwcd_ori_acc[roi_name] = ctwcd_ori_acc
        
    roi_data=dict()
    
    roi_data['ctwcd_ori_acc']=roi_ctwcd_ori_acc

    fname_data=op.join(roi_data_root, sub_info + '_' + task_info +"_ROIs_data_GAT_Ori" + '.pickle')
    fw = open(fname_data,'wb')
    pickle.dump(roi_data,fw)
    fw.close()

    # #load
    # fr=open(fname_data,'rb')
    # d2=pickle.load(fr)
    # fr.close()

    # stc_mean=stc_feat_b.copy().crop(tmin=0, tmax=0.5).mean()
    # brain_mean = stc_mean.plot(views='lateral',subject=f'sub-{subject_id}',hemi='lh',size=(800,400),subjects_dir=subjects_dir)



# Save code
#    shutil.copy(__file__, roi_code_root)
