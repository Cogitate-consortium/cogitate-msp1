"""
====================
D01. Decoding for MEG on source space of ROI, 
Category decoding
control analysis,
decoding at subROI.
====================
@author: ling liu ling.liu@pku.edu.cn

decoding methods:  CCD: Cross Condition Decoding
classifier: SVM (linear)
feature: spatial pattern (S)

feature selection methods test

"""
#import os
import os.path as op

import pickle


import argparse



from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


from config import bids_root
from D_MEG_function import set_path_ROI_MVPA, sensor_data_for_ROI_MVPA
from D_MEG_function import source_data_for_ROI_MVPA,sub_ROI_for_ROI_MVPA

from D01_ROI_MVPA_Cat import Category_WCD


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
#nPCA = opt.nPCA


# =============================================================================
# SESSION-SPECIFIC SETTINGS
# =============================================================================



subject_id = opt.sub

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path





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
    ROI_index = ['subF','subP']
    
    for analysis_index in ROI_index:
        analysis_name='Cat_' + analysis_index + '_control'
    
        
    
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
    
        #roi_ccd_acc = dict()
        #roi_ccd_auc = dict()
        roi_wcd_acc = dict()
        #roi_wcd_auc = dict()
        
    
        for nroi, roi_name in enumerate(ROI_Name):
    
            # 4 Get Source Data for each ROI
            stcs = []
            stcs = source_data_for_ROI_MVPA(epochs_rs, fpath_fw, rank, common_cov, sub_info, surf_label_list[nroi])
            
            
            # fname_fig_acc = op.join(roi_figure_root, 
            #                     sub_info + task_info + '_'+ roi_name
            #                     + "_acc_CCD" + '.png')
    
            
            # score_methods=make_scorer(accuracy_score)
            # ccd_acc = Category_CCD(epochs_rs, stcs, 
            #                                 conditions_C, conditions_D,
            #                                 select_F,
            #                                 n_trials,
            #                                 # nPCA,
            #                                 roi_name, score_methods, 
            #                                 fname_fig_acc)
    
            # roi_ccd_acc[roi_name] = ccd_acc
    
    
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
        # roi_data['ccd_acc']=roi_ccd_acc
        
        roi_data['wcd_acc']=roi_wcd_acc
        
    
    
        fname_data=op.join(roi_data_root, sub_info + '_' + task_info +"_ROIs_data_" + analysis_name + '.pickle')
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
