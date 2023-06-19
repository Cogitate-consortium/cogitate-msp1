
"""
====================
D05. RSA for MEG on source space of ROI
Category RSA
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CTWCD: Cross Time Within Condition Decoding
classifier: SVM (linear)
feature: spatial pattern (S)

"""
#import os
import os.path as op
#import joblib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import argparse


from joblib import Parallel, delayed
from tqdm import tqdm


#from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects


#from config import no_eeg_sbj
#from config import site_id, subject_id, file_names, visit_id, data_path, out_path



from rsa_helper_functions_meg import pseudotrials_rsa_all2all
from config import bids_root

from D_MEG_function import set_path_ROI_MVPA, sensor_data_for_ROI_MVPA
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
parser.add_argument('--cT',type=str,nargs='*', default=['1500ms'], help='condition in Time duration')
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
parser.add_argument('--nPerm',
                    type=int,
                    default=100,
                    help='number of Permuation for pseudo-trials, if debug, could set to 2')

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
per_num=opt.nPerm
# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path

def Category_RSA(epochs, stcs, conditions_C,conD,n_features=None, n_pseudotrials=5, n_iterations=20, n_jobs=-1, feat_sel_diag=True):
    # Find the indices of the relevant conditions:
    epochs.metadata["order"] = list(range(len(epochs.metadata)))
    meta_data = epochs.metadata
    trials_inds = meta_data.loc[((meta_data["Category"] ==  conditions_C[0]) | (meta_data["Category"] == conditions_C[1])) &
                                (meta_data["Task_relevance"] == conD), "order"].to_list()
    # Extract these trials:
    epochs = epochs[trials_inds]
    # Aggregate single trials stcs into a numpy array:
    X = np.array([stc.data for stc in stcs])
    # Select the trials of interest:
    X = X[trials_inds, :, :]
    # Get the labels:
    y = epochs.metadata["Category"].to_numpy()

    rsa_results, rdm_diag, sel_features = \
        zip(*Parallel(n_jobs=n_jobs)(delayed(pseudotrials_rsa_all2all)(
            X, y, n_pseudotrials, epochs.times, sample_rdm_times=None,n_features=n_features,metric="correlation",
             fisher_transform=True,feat_sel_diag=feat_sel_diag
        ) for i in tqdm(range(n_iterations))))
    return rsa_results, rdm_diag, sel_features


def Plot_RSA(rsa, sample, roi_name,fname_fig):      
    ##RSA plot##

    fig, axes = plt.subplots(1, 2,figsize=(10,3),sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    fig.suptitle(f'RSA_Cat_ {roi_name}')
    time_point = np.array(range(-500,2001, 10))/1000
    t = time_point
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
    cmap = mpl.cm.jet
#    bounds = np.linspace(vmin, vmax, 11)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # diff setting
    # vmind = -0.15
    # vmaxd = 0.15
#    boundsd = np.linspace(vmind, vmaxd, 11)
#    normd = mpl.colors.BoundaryNorm(boundsd, cmap.N)
    #plot
    for condi, Ti_name in rsa.items():
        if condi=='Irrelevant':
            im = axes[0].imshow(gaussian_filter(Ti_name,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                            extent=t[[0, -1, 0, -1]])
            axes[0].set(xlabel='First', ylabel='Second')
            axes[0].set_title(f'RSA_IR_ {roi_name}')
            axes[0].axvline(0, color='k')
            axes[0].axhline(0, color='k')
            axes[0].axline((0, 0), slope=1, color='k')
            plt.colorbar(im, ax=axes[0],fraction=0.03, pad=0.05)
        elif condi=='Relevant non-target':
            im = axes[1].imshow(gaussian_filter(Ti_name,sigma=2), interpolation='lanczos', origin='lower', cmap=cmap,
                            extent=t[[0, -1, 0, -1]])
            axes[1].set(xlabel='First', ylabel='Second')
            axes[1].set_title(f'RSA_RE_ {roi_name}')
            axes[1].axvline(0, color='k')
            axes[1].axhline(0, color='k')
            axes[1].axline((0, 0), slope=1, color='k')
            plt.colorbar(im, ax=axes[1],fraction=0.03, pad=0.05)

    # Save figure

    fig.savefig(op.join(fname_fig+ "_rsa_Cat" + '.png'))

     
#     ##sample RDM plot##

#     fig, axes = plt.subplots(1, 2,figsize=(10,3),sharex=True,sharey=True)
#     plt.subplots_adjust(wspace=0.5, hspace=0)
#     fig.suptitle(f'Sample_RDM_ {roi_name}')
    
    
#     pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5), path_effects.Normal()]
#     cmap = mpl.cm.jet
# #    bounds = np.linspace(vmin, vmax, 11)
#     # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     # diff setting
# #    vmind = -0.15
# #    vmaxd = 0.15
# #    boundsd = np.linspace(vmind, vmaxd, 11)
# #    normd = mpl.colors.BoundaryNorm(boundsd, cmap.N)
#     #plot
#     for condi, Ti_name in sample.items():
#         if condi=='Irrelevant':
#             im = axes[0].imshow(Ti_name, interpolation='lanczos', origin='lower', cmap=cmap,
#                             extent=t[[0, -1, 0, -1]])
#             axes[0].set(xlabel='First', ylabel='Second')
#             axes[0].set_title(f'Sample_IR_ {roi_name}')
#             axes[0].axvline(0, color='k')
#             axes[0].axhline(0, color='k')
#             axes[0].axline((0, 0), slope=1, color='k')
#             plt.colorbar(im, ax=axes[0],fraction=0.03, pad=0.05)
#         elif condi=='Relevant non-target':
#             im = axes[1].imshow(Ti_name, interpolation='lanczos', origin='lower', cmap=cmap,
#                             extent=t[[0, -1, 0, -1]])
#             axes[1].set(xlabel='First', ylabel='Second')
#             axes[1].set_title(f'Sample_RE_ {roi_name}')
#             axes[1].axvline(0, color='k')
#             axes[1].axhline(0, color='k')
#             axes[1].axline((0, 0), slope=1, color='k')
#             plt.colorbar(im, ax=axes[1],fraction=0.03, pad=0.05)

#     # Save figure

#     fig.savefig(op.join(fname_fig+ "_sample_rdm_Cat" + '.png'))
    
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
    
    # con_C = ['LF']
    # con_D = ['Irrelevant', 'Relevant non-target']
    # con_T = ['500ms','1000ms','1500ms']
    #metric="correlation" or metric='euclidean'

    analysis_name='RSA_Cat'

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


    
    roi_rsa = dict()
    roi_sample = dict()
    roi_feature = dict()





    for nroi, roi_name in enumerate(ROI_Name):

        # 4 Get Source Data for each ROI
        stcs = []
        stcs = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[nroi])
        
        
        
        
        
        if roi_name=='GNW':
            sample_times=[0.3, 0.5]
        else:
            sample_times=[0.3, 1.5]
            
        cT_rsa = dict()
        cT_sample = dict()
        cT_features = dict()
        
        for nd, conD in enumerate(conditions_D):
            rsa, sample, sel_features = Category_RSA(epochs_rs, stcs, conditions_C, conD, n_features=None)
            
            # converting dictionary to
            # numpy array
            rsa_array = np.asarray(rsa)
            sample_array = np.asarray(sample)
            features_array = np.asarray(sel_features)
            
            
        
            cT_rsa[conD] = np.mean(rsa_array, axis=0)
            cT_sample[conD] = np.mean(sample_array, axis=0)
            cT_features[conD] = features_array
        
        roi_rsa[roi_name]=cT_rsa
        roi_sample[roi_name] = cT_sample
        roi_feature[roi_name] = cT_features
        
        roi_data=dict()
        roi_data['rsa']=roi_rsa
        roi_data['sample']=roi_sample
        roi_data['feature']=roi_feature
        

        fname_data=op.join(roi_data_root, sub_info + '_' + task_info + roi_name + "_ROIs_data_RSA_Cat" + '.pickle')
        fw = open(fname_data,'wb')
        pickle.dump(roi_data,fw)
        fw.close()
        
        #pot results
        # #1 scoring methods with accuracy score
        fname_fig = op.join(roi_figure_root, 
                            sub_info + task_info + '_'+ roi_name
                            )
        Plot_RSA(cT_rsa, cT_sample, roi_name,fname_fig)



            

    # #load
    # fr=open(fname_data,'rb')
    # d2=pickle.load(fr)
    # fr.close()

    # stc_mean=stc_feat_b.copy().crop(tmin=0, tmax=0.5).mean()
    # brain_mean = stc_mean.plot(views='lateral',subject=f'sub-{subject_id}',hemi='lh',size=(800,400),subjects_dir=subjects_dir)



# Save code
#    shutil.copy(__file__, roi_code_root)
