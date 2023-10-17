"""
====================
D99. Group analysis for decoding pattern
prepare for ploting
====================

@author: ling liu ling.liu@pku.edu.cn



"""

import os.path as op
import os
import argparse

import pickle

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root

from sublist_phase2 import sub_list

parser = argparse.ArgumentParser()
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--cT', type=str, nargs='*', default=['500ms','1000ms','1500ms'],
                    help='condition in Time duration:  [500ms],[1000ms],[1500ms]')
parser.add_argument('--cC', type=str, nargs='*', default=['FO'],
                    help='selected decoding category, FO for face and object, LF for letter and false')
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
parser.add_argument('--analysis',
                    type=str,
                    default='Cat',
                    help='the name for anlaysis, e.g. Cat or Ori or GAT_Cat')
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


opt = parser.parse_args()

visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path
analysis_name=opt.analysis


opt = parser.parse_args()
con_C = opt.cC
con_D = opt.cD
con_T = opt.cT

# if analysis_name=='Cat' or analysis_name=='Ori':
#     if methods_name=='T_all':
#         con_T=['500ms','1000ms','1500ms']
#     else:
#         con_T = methods_name[0]
    

select_F = opt.nF
n_trials = opt.nT
nPCA = opt.nPCA


visit_id = opt.visit
space = opt.space
subjects_dir = opt.fs_path

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
    
#1) Select time duration
if con_T[0] == 'T_all':
    con_T = ['500ms', '1000ms','1500ms']
    print(con_T)
elif con_T[0] == 'ML':# middle and long
    con_T = ['1000ms','1500ms']
    print(con_T)

    
    
decoding_path=op.join(bids_root, "derivatives",'decoding','roi_mvpa')

data_path=op.join(decoding_path,analysis_name)

# Set path to group analysis derivatives
group_deriv_root = op.join(data_path, "group_phase2")
if not op.exists(group_deriv_root):
    os.makedirs(group_deriv_root)

# evokeds_group = []
# stc_group = []


sb_list=sub_list



# analysis/task info
## analysis/task info

## analysis/task info
if con_T.__len__() == 3:
    con_Tname = 'T_all'
elif con_T.__len__() == 2:
    con_Tname = con_T[0]+'_'+con_T[1]
else:
    con_Tname = con_T[0]

task_info = "_" + "".join(con_Tname) + "_" + "".join(con_C[0])
print(task_info)

group_data=dict()
for i, sbn in enumerate(sb_list):
    # if 'SB' in sbn:
    # sub and visit info
    sub_info = 'sub-' + sbn + '_ses-' + visit_id
    
    sub_data_root = op.join(data_path,
                            f"sub-{sbn}", f"ses-{visit_id}", "meg",
                            "data")
    # if analysis_name=='Cat':
    #     pkl_name = "_ROIs_data_Cat"
    # elif analysis_name=='Ori':
    #     pkl_name = "_ROIs_data_Ori"
    # elif analysis_name=='GAT_Cat':
    #     pkl_name = "_ROIs_data_GAT_Cat"
    # elif analysis_name=='GAT_Ori':
    #     pkl_name = "_ROIs_data_GAT_Cat"
    # elif analysis_name=='RSA_Cat':
    #     pkl_name = "_ROIs_RSA_Cat"
    # elif analysis_name=='RSA_Ori':
    #     pkl_name = "_ROIs_RSA_Ori"
    # elif analysis_name=='RSA_ID':
    #     pkl_name = "_ROIs_RSA_ID"
    rsa_data=dict()
    if analysis_name == "RSA_Cat" or analysis_name=="RSA_Ori" or analysis_name=="RSA_ID":
        for ri,roi_name in enumerate(['GNW','IIT']):
            fname_data=op.join(sub_data_root, sub_info + '_' + task_info + roi_name +'_ROIs_data_'+ analysis_name +'.pickle')
            fr=open(fname_data,'rb')
            rsa_data[roi_name]=pickle.load(fr)
        group_data[sbn]=rsa_data
        
    elif analysis_name == "Cat_PFC":
        fname_data=op.join(sub_data_root, sub_info + '_' + task_info +'_IITPFC_data_Cat.pickle')
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
        
    elif analysis_name == "Ori_PFC":
        fname_data=op.join(sub_data_root, sub_info + '_' + task_info +'_IITPFC_data_Ori.pickle')
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data
            
    else:       
        fname_data=op.join(sub_data_root, sub_info + '_' + task_info +'_ROIs_data_'+analysis_name +'.pickle')
        fr=open(fname_data,'rb')
        roi_data=pickle.load(fr)
        group_data[sbn]=roi_data

fname_data=op.join(group_deriv_root, task_info +"_data_group_" + analysis_name +
                   '.pickle')
fw = open(fname_data,'wb')
pickle.dump(group_data,fw)
fw.close()


