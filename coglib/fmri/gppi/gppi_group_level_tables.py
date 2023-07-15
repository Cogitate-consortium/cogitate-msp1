# Generates ROI tables based on gppi stats maps

"""
Created by Yamil Vidal and modified by Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 04-11-2023
Date modified: 04-26-2023
"""

import sys
import numpy as np
import pandas as pd
import os

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'


# data path
data_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/gppi/second level_combined_nonparametric/PPI_FFA_gPPI_300/face-object'


projectRoot = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed'

#subject_list_type = 'phase2_V1'
subject_list_type = 'phase3_V1_subset'
#subject_list_type = 'debug'

bids_dir = projectRoot + '/bids'  
code_dir = projectRoot + '/bids/code' 

output_dir = data_dir

mask_dir = bids_dir + '/derivatives/masks/ICBM2009c_asym_nlin'
conjunct_pattern = data_dir + '/%(conj)s_conjunction_not_A_or_B.nii.gz'
group_mask = bids_dir + '/derivatives/fslFeat/group/ses-V1/' + subject_list_type + '/ses-V1_task-Dur_analysis-3rdGLM_space-MNI152NLin2009cAsym_desc-cope15.gfeat/cope1.feat/mask.nii.gz'
input_nifti= os.path.join(data_dir, 'gppi_group_stats_map.nii')
output_fname_pattern_cvs = data_dir + '/gppi_group_level_table.csv'
# load helper functions / code dir
sys.path.append(code_dir)
from helper_functions_MRI import load_mri

# %% ROIs
roi_list = ['G_and_S_cingul-Mid-Post',
            'Lat_Fis-ant-Horizont',
            'Lat_Fis-ant-Vertical',
            'G_and_S_cingul-Ant',
            'G_and_S_cingul-Mid-Ant',
            'G_front_inf-Opercular',
            'G_front_inf-Orbital',
            'G_front_inf-Triangul',
            'G_front_middle',
            'S_front_middle',
            'S_front_sup',
            'G_and_S_frontomargin',
            'G_and_S_transv_frontopol',
            'G_front_sup',
            'G_rectus',
            'G_subcallosal',
            'S_orbital_lateral',
            'S_orbital_med-olfact',
            'S_orbital-H_Shaped',
            'S_suborbital',
            'G_and_S_occipital_inf',
            'G_oc-temp_lat-fusifor',
            'G_occipital_middle',
            'S_oc_middle_and_Lunatus',
            'G_cuneus',
            'G_occipital_sup',
            'G_oc-temp_med-Lingual',
            'G_oc-temp_med-Parahip',
            'G_temporal_inf',
            'Pole_occipital',
            'Pole_temporal',
            'S_calcarine',
            'S_intrapariet_and_P_trans',
            'S_oc_sup_and_transversal',
            'S_temporal_sup',
            'S_front_inf',
            'G_orbital',
            'G_pariet_inf-Angular',
            'G_pariet_inf-Supramar',
            'G_precentral',
            'G_temp_sup-Lateral',
            'G_temp_sup-Plan_tempo',
            'G_temporal_middle',
            'S_interm_prim-Jensen',
            'S_occipital_ant',
            'S_oc-temp_lat',
            'S_precentral-inf-part',
            'S_temporal_inf',
            'GNW',
            'IIT',
            'IIT_extended',
            'IIT_excluded']



# Should only bilateral masks be processed? these are assumed to be label with 
# a 'bh' (both hemispheres) in the file name (as created by 01_create_ROI_masks.py)
process_only_bilateral_masks = True


# load all anatomical ROIs
def get_mask_list(sub_mask_dir):
    """
    Get list of paths to all masks for current subject. Prints how many masks
    are found or prints warning (not an error) if none have been found.
    sub_mask_dir: path to mask dir
    Returns: list of paths to all masks
    """
    from glob import glob
    if process_only_bilateral_masks:
        mask_list = glob(sub_mask_dir + '/*_bh_*' + 'MNI152NLin2009cAsym.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of bilateral masks only. Found ' + str(n_masks) + ' masks')
    else:
        mask_list = glob(sub_mask_dir + '/*.nii.gz')
        n_masks = len(mask_list)
        print('. Getting list of all masks. Found ' + str(n_masks) + ' masks')
    if not mask_list:
        print('! Warning no masks found for current subject !!!')
    return mask_list

def load_a_rois():
    """
    Load all anatomica ROIs of a subject into a dictionary
    
    sub_id: Subject ID
    Returns: Dictionary containing all the anatomical ROIs of the subject
    """
    
    mask_paths = get_mask_list(mask_dir)
    mask_paths.sort()
    
    sub_mask_list = [l[115:] for l in mask_paths]
    sub_mask_list = [l[:-33] for l in sub_mask_list]
    
    # empty dictionary that will contain all masks of a subject
    a_rois = {}
    
    for n in range(0,len(mask_paths)):
        mask = mask_paths[n]
        m = load_mri(mask, group_mask)
        a_rois[sub_mask_list[n]] = m
        
    return a_rois, sub_mask_list


def count_voxels_in_roi(a_roi,conjunction):
    
    # Set to zero the voxels that don't belong to the ROI
    
    c = conjunction.copy()
    c[a_roi == 0] = 0
    
    n_voxels = int(np.sum(c > 0))
    
    return n_voxels

# %% run


    
voxel_count = dict(zip(roi_list,np.zeros((len(roi_list),2), dtype=float)))

# Load the relevant functional maps
conjunction = np.squeeze(load_mri(input_nifti,group_mask))
    
a_rois, sub_roi_list = load_a_rois()
        
for roi_name in roi_list:
        

    a_roi = np.squeeze(a_rois[roi_name])

    n_voxels = count_voxels_in_roi(a_roi,conjunction)
    #print(str(n_voxels))
    #print(str(n_voxels/sum(a_roi)))
    voxel_count[roi_name][0] = n_voxels
    voxel_count[roi_name][1] = n_voxels/sum(a_roi)
    
voxel_df = pd.DataFrame.from_dict(voxel_count, orient='index')
# Save gppi ROI table
voxel_df.to_csv(output_fname_pattern_cvs, header=False)
