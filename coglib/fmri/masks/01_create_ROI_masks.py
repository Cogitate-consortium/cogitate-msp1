#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create anatomical ROI masks using FreeSurfer (segmentation & atlas) for each 
participant.

First gets subject list from bids_dir, then finds FS dir and anatomical scan's 
path, before running bbregister for subject and creating volumetric ROI masks 
from list of fs_labels and from atlas (atlas) annotations (atlas_annotations) 
for each hemisphere (hemispheres), and if two hemispheres are supplied merges
masks into bilateral masks including both hemispheres (bh). If any 
remove_overlap_strategy is selected removes any voxels with overlap from final 
masks (see strategy options). Finally, writes all masks to mask_output_dir per 
subject.

Depending on the number of masks to be created and the selected options 
(dilation and overlap removal) script may take ~15min per participant.

Make sure freesufer & FSL are loaded before executing the python script here. 
I.e., in the shell run, before executing the python script:
module load FreeSurfer
module load FSL

Created on Tue Aug 24 13:39:06 2021

@author: David Richter, Yamil Vidal
@tag: prereg_v4.2
"""

import os, sys


#%% Paths and Parameters

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'

#subject_list_type = 'phase2_V1'
subject_list_type = 'debug'

# session labels (checks for anat file in first session, if none is found checks second session; i.e. allows exceptions where some participants have the anat in session 2)
session_labels = ['ses-V1', 'ses-V2']


# ROIs of interest (these must correspond to FS label/annotations!)
# labels (FS labels procuded by recon-all)
#fs_labels = ['V1_exvivo']
fs_labels = []

# create EVC mask? If True merges FS V1 and/or V2 labels to create a combined EVC mask (assumes that V1 and V2 are also supplied as fs_labels above; otherwise only V1 or V2 is used, depending on which mask is defined in fs_labels)
create_EVC_mask = False


# atlas & the associated annotations (FS)
# atlas = 'aparc.annot'
# atlas_annotations = ['fusiform', 'lateraloccipital',
#                      'frontalpole', 'medialorbitofrontal', 
#                      'lateralorbitofrontal', 'rostralmiddlefrontal', 
#                      'superiorfrontal','rostralanteriorcingulate', 
#                      'caudalanteriorcingulate', 'parsorbitalis', 
#                      'parstriangularis', 'parsopercularis']

atlas = 'aparc.annot.a2009s'
atlas_annotations = ['G&S_occipital_inf',
                      'G&S_cingul-Ant',
                      'G&S_cingul-Mid-Ant',
                      'G&S_cingul-Mid-Post',
                      'G_front_inf-Opercular',
                      'G_front_inf-Orbital',
                      'G_front_inf-Triangul',
                      'G_front_middle',
                      'G_occipital_middle',
                      'Lat_Fis-ant-Horizont',
                      'Lat_Fis-ant-Vertical',
                      'S_front_inf',
                      'S_front_middle',
                      'S_front_sup',
                      'S_interm_prim-Jensen',
                      'G_cuneus',
                      'G_occipital_sup',
                      'G_oc-temp_lat-fusifor',
                      'G_oc-temp_med-Lingual',
                      'G_oc-temp_med-Parahip',
                      'G_orbital',
                      'G_pariet_inf-Angular',
                      'G_pariet_inf-Supramar',
                      'G_precentral',
                      'G_temp_sup-Lateral',
                      'G_temp_sup-Plan_tempo',
                      'G_temporal_inf',
                      'G_temporal_middle',
                      'Pole_occipital',
                      'Pole_temporal',
                      'S_calcarine',
                      'S_intrapariet&P_trans',
                      'S_oc_middle&Lunatus',
                      'S_oc_sup&transversal',
                      'S_occipital_ant',
                      'S_oc-temp_lat',
                      'S_precentral-inf-part',
                      'S_temporal_inf',
                      'S_temporal_sup',
                      'G&S_frontomargin',
                      'G&S_transv_frontopol',
                      'G_front_sup',
                      'G_rectus',
                      'G_subcallosal',
                      'S_orbital_lateral',
                      'S_orbital_med-olfact',
                      'S_orbital-H_Shaped',
                      'S_suborbital']

# atlas_annotations = ['G&S_frontomargin',
#                     'G&S_transv_frontopol',
#                     'G_front_sup',
#                     'G_rectus',
#                     'G_subcallosal',
#                     'S_orbital_lateral',
#                     'S_orbital_med-olfact',
#                     'S_orbital-H_Shaped',
#                     'S_suborbital']


# hemispheres (FS labels are returned per hemisphere; we combine all masks into a bilateral mask if lh and rh are defined here)
hemispheres = ['lh', 'rh']


# fill holes in masks? If True fills holes in masks (i.e. adds voxels to mask that are surrounded by it; -fillh). If mask is dilated, filling holes is not necessary
fill_holes = False

# dilate masks? If True dilates masks using a default box kernel of 3x3x3 with maximum filtering of voxel values; -dilF
dilate_mask = True

# remove overlap between masks in list? removes voxels from masks if they cannot be uniquely attributed to one mask (merges fs_labels & atlas_annotations to form list of masks and includes EVC mask if create_EVC_mask is true)
# Options:
# 'Remove' = removes non-unique voxels from all masks, hence creating empty borders around masks (usually not desirable)
# 'Smaller' = attributes non-unique voxels to the smaller of two overlapping masks, while removing voxels from the bigger mask
# 'None' = does not remove any overlapping voxels
remove_overlap_strategy = 'Smaller'


# %% Additional paths/parameters. These should be identical across BIDS datasets

# freesurfer path pattern
fs_path = bids_dir + '/derivatives/freesurfer'
fs_subject_pattern = fs_path + '/%(sub)s'

# T1w file path pattern
anat_file_pattern =  bids_dir + '/derivatives/fmriprep/%(sub)s/%(ses)s/anat/%(sub)s_%(ses)s_acq-anat_run-1_desc-preproc_T1w.nii.gz'

# path where custom masks are written to
mask_output_pattern = bids_dir + '/derivatives/masks/%(sub)s'

# Pattern to check if a mask already exists
fname_pattern = '%(mask_output_dir)s/%(sub_id)s_%(fill_label)s.nii.gz'

# load helper functions / code dir
code_dir_with_helperfunctions = bids_dir + '/code'
sys.path.append(code_dir_with_helperfunctions)
from helper_functions_MRI import load_mri, get_subject_list


# %% utility functions

# run shell command line using subprocess
def run_subprocess(full_cmd):
    """
    Runs shell command given in full_cmd using subprocess.
    full_cmd: full shell command line to run.
    Returns: stdout
    """
    import subprocess
    # execute command
    subprocess_return = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE)
    return subprocess_return.stdout.decode('utf-8')
    

# %% main fs mask creation functions
    
def create_fs_register_dat(fs_path_sub, sub_id, anat_fname):
    """
    Creates FS register.dat file using bbregister (run in shell), given 
    subject's FS paths, ID and T1 file.
    fs_path_sub: Path to subject's freesurfer dir.
    sub_id: subject ID.
    anat_fname: Path to subject's T1w (anatomical) scan.
    Returns: nothing, but writes register.dat to fs_path_sub
    """
    reg_dat_fname = fs_path_sub + '/register.dat'
    if not os.path.isfile(reg_dat_fname):
        print('. creating register.dat file')
        # create shell command for bbregister for current subject
        register_dat_cmd_pattern = 'export SUBJECTS_DIR=%(fs_path)s; bbregister --s %(sub_id)s --mov %(anat_fname)s --reg %(fs_path_sub)s/register.dat --init-header --t1'
        full_cmd = register_dat_cmd_pattern%{'fs_path':fs_path, 'sub_id':sub_id, 'anat_fname':anat_fname, 'fs_path_sub':fs_path_sub}
        # execute command
        run_subprocess(full_cmd)
    else:
        print('. register.dat file already exists! Skipping')

def create_masks_from_labels(fs_labels, hemispheres, fs_path, fs_path_sub, sub_id, mask_output_dir):
    """
    Creates volumetric ROI masks from existing FS labels. If two hemispheres 
    are supplied, also merges left+right hemisphere mask into bilateral mask.
    fs_labels: Freesurfer labels to create volumetric masks from.
    hemispheres: hemisphere labels (rh, lh)
    fs_path: Freesurfer dir.
    fs_path_sub: Path to subject's freesurfer dir.
    sub_id: subject ID.
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates ROI masks per label & hemisphere in 
    mask_output_dir
    """
    for label in fs_labels:
        print('. creating mask from label: ' + label)
        # run label for both hemispheres
        for hemi in hemispheres:
            # create shell command for mri_label2vol for current subject & label
            label2vol_cmd_pattern = 'export SUBJECTS_DIR=%(fs_path)s; mri_label2vol --label %(fs_path_sub)s/label/%(hemi)s.%(label)s.label --temp %(fs_path_sub)s/mri/orig.mgz --subject %(sub_id)s --hemi %(hemi)s --o %(mask_output_dir)s/%(sub_id)s_%(hemi)s_%(label)s.nii.gz --proj frac 0 1 0.01 --fillthresh 0 --identity'
            full_cmd = label2vol_cmd_pattern%{'fs_path':fs_path, 'fs_path_sub':fs_path_sub, 'hemi':hemi, 'label':label, 'sub_id':sub_id, 'mask_output_dir':mask_output_dir}
            # execute command
            run_subprocess(full_cmd)
            # fill holes in mask
            fill_label = hemi + '_' + label
            fill_holes_and_dilate_masks(mask_output_dir, sub_id, fill_label)
        # merge left and right hemispheres
        if len(hemispheres) > 1:
            merge_hemispheres_for_bilateral_mask(mask_output_dir, sub_id, label)

def create_masks_from_annotations(atlas, atlas_annotations, hemispheres, fs_path, fs_path_sub, sub_id, mask_output_dir):
    """
    Creates volumetric ROI masks from FS atlas annotations. If two hemispheres 
    are supplied, also merges left+right hemisphere mask into bilateral mask.
    atlas: Freesurfer atlas (e.g. aparc)
    atlas_annotations: Freesurfer atlas annotations corresponding to masks in 
    the atlas.
    hemispheres: hemisphere labels (rh, lh).
    fs_path: Freesurfer dir.
    fs_path_sub: Path to subject's freesurfer dir.
    sub_id: subject ID.
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates ROI masks per label & hemisphere in 
    mask_output_dir
    """
    # create labels from atlas
    print('. creating labels from annotation in atlas: ' + atlas)
    for hemi in hemispheres:
        # create shell command for mri_label2vol for current subject & label
        #annot2label_cmd_pattern = 'export SUBJECTS_DIR=%(fs_path)s; mri_annotation2label  --subject %(sub_id)s --hemi %(hemi)s --labelbase %(fs_path_sub)s/label/%(atlas)s-%(hemi)s'
        annot2label_cmd_pattern = 'export SUBJECTS_DIR=%(fs_path)s; mri_annotation2label  --subject %(sub_id)s --hemi %(hemi)s --labelbase %(fs_path_sub)s/label/%(atlas)s-%(hemi)s --annotation aparc.a2009s'
        full_cmd = annot2label_cmd_pattern%{'fs_path':fs_path, 'sub_id':sub_id, 'hemi':hemi, 'fs_path_sub':fs_path_sub, 'atlas':atlas}
        # execute command
        run_subprocess(full_cmd)
    # read annotation labels to get indices
    ctab_path_pattern = '%(fs_path_sub)s/label/%(atlas)s.ctab'
    ctab_fname = ctab_path_pattern%{'fs_path_sub':fs_path_sub,'atlas':atlas}
    import pandas as pd
    ctab = pd.read_csv(ctab_fname, header=None, sep="\s", engine='python') # col 0=idx, 2=labels
    # process labels derived from annotations
    for annot in atlas_annotations:
        print('. creating mask from label (annotation): ' + annot)
        # get index (which also consitutues the label name) corresponding to current annotation
        annot_index = ctab[0][ctab[2]==annot].item()
        annot_index = "{0:0>3}".format(annot_index)
        
        annot = annot.replace("&","_and_")
        
        fill_label = hemispheres[0] + '_' + annot
        fname = fname_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'fill_label':fill_label}
        l_exist = os.path.isfile(fname)
        
        fill_label = hemispheres[1] + '_' + annot
        fname = fname_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'fill_label':fill_label}
        r_exist = os.path.isfile(fname)
        
        if l_exist and r_exist:
            print('Masks for ' + annot + ' exist. Skipped')
        else:
        
            for hemi in hemispheres:
                # create shell command for mri_label2vol for current subject & label dervied from annotation
                annot2vol_cmd_pattern = 'export SUBJECTS_DIR=%(fs_path)s; mri_label2vol --label %(fs_path_sub)s/label/%(atlas)s-%(hemi)s-%(annot_index)s.label --temp %(fs_path_sub)s/mri/orig.mgz --subject %(sub_id)s --hemi %(hemi)s --o %(mask_output_dir)s/%(sub_id)s_%(hemi)s_%(annot)s.nii.gz --proj frac 0 1 0.01 --fillthresh 0 --identity'
                full_cmd = annot2vol_cmd_pattern%{'fs_path':fs_path, 'fs_path_sub':fs_path_sub, 'atlas':atlas, 'hemi':hemi, 'annot_index':annot_index, 'sub_id':sub_id, 'mask_output_dir':mask_output_dir, 'annot':annot}
                # execute command
                run_subprocess(full_cmd)
                # fill holes in mask
                fill_label = hemi + '_' + annot
                fill_holes_and_dilate_masks(mask_output_dir, sub_id, fill_label)
            # merge left and right hemispheres
            if len(hemispheres) > 1:
                merge_hemispheres_for_bilateral_mask(mask_output_dir, sub_id, annot)

def merge_hemispheres_for_bilateral_mask(mask_output_dir, sub_id, label):
    """
    Creates bilateral (both hemispheres) volumetric ROI masks from left + right 
    hemisphere masks. LH and RH masks of the given label are assumed to exist
    in mask_output_dir when function is called.
    sub_id: subject ID.
    label: mask label or annotation label
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates bilateral ROI masks in mask_output_dir
    """
    print('. . merging masks to form bilateral mask for: ' + label)
    fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_rh_%(label)s.nii.gz -add %(mask_output_dir)s/%(sub_id)s_lh_%(label)s.nii.gz -bin %(mask_output_dir)s/%(sub_id)s_bh_%(label)s.nii.gz'
    full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'label':label}
    # execute command
    run_subprocess(full_cmd)

def make_overlapping_voxels_mask(mask_output_dir, sub_id, mask_list):
    """
    Creates a mask that contains only voxels that are shared/overlap between at 
    least two masks in the mask_list.
    sub_id: subject ID.
    mask_list: list of mask labels
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates overlap masks in mask_output_dir
    """
    print('. creating mask with overlapping voxels found in: ', mask_list)
    tmp_cmd = ''
    # add each mask in list together
    for label in mask_list:
        tmp_cmd = tmp_cmd + '%(mask_output_dir)s/%(sub_id)s_%(label)s.nii.gz -add '%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'label':label}
    # threshold such that only overlaps are retained
    fslmaths_cmd_pattern = 'fslmaths %(tmp_cmd)s -thr 2 -bin %(mask_output_dir)s/%(sub_id)s_tmp_mask_overlap.nii.gz'
    full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'tmp_cmd':tmp_cmd[0:-6]}
    # execute command to make overlap mask
    run_subprocess(full_cmd)

def remove_overlapping_voxels_from_masks(mask_output_dir, sub_id, mask_list):
    """
    Creates a version of each mask that contains no shared/overlapping voxels 
    that are part of any other mask in the mask_list by removing all 
    overlapping voxels from all masks (creates non assigned borders between 
    masks, which probably is not useful in most cases)
    sub_id: subject ID.
    mask_list: list of mask labels
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates versions of each masks in mask_list without 
    any overlap with other masks in the list
    """
    # remove overlap mask from all masks in list
    for label in mask_list:
        print('. removing any voxels with overlap from: ' + label)
        fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_%(label)s.nii.gz -sub %(mask_output_dir)s/%(sub_id)s_tmp_mask_overlap.nii.gz -thr 1 -bin %(mask_output_dir)s/%(sub_id)s_%(label)s.nii.gz'
        full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'label':label}
        # execute command to remove overlap voxels from mask
        run_subprocess(full_cmd)

def assign_overlapping_voxels_to_smaller_mask(mask_output_dir, sub_id, mask_list):
    """
    Creates a version of each mask that contains no shared/overlapping voxels 
    that are part of any other mask in the mask_list by assigning any 
    overlapping voxels to the smaller of two masks (makes use of all voxels)
    sub_id: subject ID.
    mask_list: list of mask labels
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates versions of each masks in mask_list without 
    any overlap with other masks in the list
    """
    from itertools import combinations
    print('. checking for overlapping voxels')
    # remove overlap mask from all masks in list
    for [label_A, label_B] in list(combinations(mask_list,2)):
        # get mask A & B
        mask_A = '%(mask_output_dir)s/%(sub_id)s_%(label)s.nii.gz'%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'label':label_A}
        mask_B = '%(mask_output_dir)s/%(sub_id)s_%(label)s.nii.gz'%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'label':label_B}
        # check if there is any overlap of mask and A & B
        size_overlap = int(load_mri(mask_A,mask_B).sum())
        # if there's no overlap masks can remain unchanged
        if size_overlap == 0:
            continue
        # if there is overlap, compare sizes of A & B and adjust masks
        size_A = int(load_mri(mask_A,mask_A).sum())
        size_B = int(load_mri(mask_B,mask_B).sum())
        # if A>=B remove voxels of B from A, keep B unchanged
        if size_A >= size_B:
            full_cmd = 'fslmaths ' + mask_A + ' -sub ' + mask_B + ' -thr 1 -bin ' + mask_A
            print('. . removing any voxels with overlap from: ' + label_A + ' while keeping the voxels in: ' + label_B)
        elif size_B > size_A:
            full_cmd = 'fslmaths ' + mask_B + ' -sub ' + mask_A + ' -thr 1 -bin ' + mask_B
            print('. . removing any voxels with overlap from: ' + label_B + ' while keeping the voxels in: ' + label_A)
        # execute command to remove overlap voxels from mask
        run_subprocess(full_cmd)

def create_evc_mask(fs_labels, hemispheres, sub_id, mask_output_dir):
    """
    Creates an early visual cortex (EVC) mask by combining existing V1+V2 
    exvivo masks from FS recon-all. Modifies fs_labels list to replace V1,V2 
    with new EVC mask label to avoid including of V1,V2 and EVC mask in later
    voxel overlap removal.
    sub_id: subject ID.
    hemispheres: list of hemisphere(s) to process (lh, rh)
    mask_output_dir: Path to mask output folder.
    Returns: mod_fs_labels updated list of fs_labels, including the new EVC 
    mask, but removing V1 and V2 labels. Also creates EVC mask in 
    mask_output_dir
    """
    # get hemisphere
    if len(hemispheres) > 1:
        hemi = 'bh_'
    else:
        hemi = hemispheres[0] + '_'
    # check that V1 and V2 labels exist in fs_labels, otherwise use only V1 or V2 (whatever is found)
    for label in fs_labels:
        # add all labels to new fs_label list that are not V1 or V2
        evc_labels = []
        if (label == 'V1_exvivo' or label == 'V2_exvivo'):
            evc_labels.append(hemi + label)
    # check whether V1 and/or V2 should be used to create EVC mask
    if len(evc_labels) == 1:
        print('. creating EVC mask by using only: ' + evc_labels[0])
        fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_%(v1label)s.nii.gz -bin %(mask_output_dir)s/%(sub_id)s_%(hemi)sEVC_exvivo.nii.gz'
        full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'v1label':evc_labels[0], 'hemi':hemi}
    elif len(evc_labels) == 2:
        print('. creating EVC mask by combining: ' + evc_labels[0] + ' + ' + evc_labels[1])
        fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_%(v1label)s.nii.gz -add %(mask_output_dir)s/%(sub_id)s_%(v2label)s.nii.gz -bin %(mask_output_dir)s/%(sub_id)s_%(hemi)sEVC_exvivo.nii.gz'
        full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'v1label':evc_labels[0], 'v2label':evc_labels[1], 'hemi':hemi}
    else:
        print('. NOT creating EVC mask - unexpected number of EVC labels. Use either V1 and/or V2.')
        return
    # execute command to remove overlap voxels from mask
    run_subprocess(full_cmd)
    # remove V1 and V2 from fs_labels list
    mod_fs_labels = ['EVC_exvivo']
    for label in fs_labels:
        # add all labels to new fs_label list that are not V1 or V2
        if not (label == 'V1_exvivo' or label == 'V2_exvivo'):
            mod_fs_labels.append(label)
    return mod_fs_labels

def fill_holes_and_dilate_masks(mask_output_dir, sub_id, fill_label):
    """
    Fills holes in mask and then dilates them (3x3x3 kernel) - overwrites input
    sub_id: subject ID.
    fill_label: label of mask to fill holes in 
    mask_output_dir: Path to mask output folder.
    Returns: nothing, but creates version of each masks in mask_list without 
    holes.
    """
    # remove holes in mask
    if fill_holes:
        print('. . filling holes in mask: ' + fill_label)
        fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_%(fill_label)s.nii.gz -fillh %(mask_output_dir)s/%(sub_id)s_%(fill_label)s.nii.gz'
        full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'fill_label':fill_label}
        # execute command to fill holes in mask
        run_subprocess(full_cmd)
    # dilate mask
    if dilate_mask:
        print('. . dilating mask: ' + fill_label)
        fslmaths_cmd_pattern = 'fslmaths %(mask_output_dir)s/%(sub_id)s_%(fill_label)s.nii.gz -kernel 3D -dilF %(mask_output_dir)s/%(sub_id)s_%(fill_label)s.nii.gz'
        full_cmd = fslmaths_cmd_pattern%{'mask_output_dir':mask_output_dir, 'sub_id':sub_id, 'fill_label':fill_label}
        # execute command to dilate masks
        run_subprocess(full_cmd)
        

# %% run
if __name__ == '__main__':
    """
    Create ROI masks for each subject.
    Gets subject list from bids_dir, then finds FS dir and anatomical scan's 
    path, before running bbregister for subject and creating volumetric ROI
    masks from list of fs_labels and from atlas annotations 
    (atlas_annotations). Dilates/fills holes in masks before merging left+right
    hemisphere masks into bilateral mask. Removes overlapping voxesl between 
    masks in list, if a remove_overlap_strategy is defined. Writes all masks to 
    mask_output_dir per subject.
    """
    
    # get subject list
    subjects = get_subject_list(bids_dir,subject_list_type)
    
    # run mask creation for each subject
    for sub_id in subjects:
        print('Creating ROI mask for: ' + sub_id)
        
        # get freesurfer path
        fs_path_sub = fs_subject_pattern%{'sub':sub_id}
        
        # find anat file
        for ses in session_labels:
            anat_fname = anat_file_pattern%{'sub':sub_id, 'ses':ses}
            if os.path.isfile(anat_fname):
                print('. using anat --> ' + anat_fname[anat_fname.find(sub_id)::])
                break
        
        # create FS register.dat if it does not exist yet
        create_fs_register_dat(fs_path_sub, sub_id, anat_fname)
        
        # make output dir for masks
        mask_output_dir = mask_output_pattern%{'sub':sub_id}
        if not os.path.isdir(mask_output_dir):
            os.makedirs(mask_output_dir)
        
        # process existing labels
        if not not fs_labels:
            create_masks_from_labels(fs_labels, hemispheres, fs_path, fs_path_sub, sub_id, mask_output_dir)
        
        # process annotations from atlas
        create_masks_from_annotations(atlas, atlas_annotations, hemispheres, fs_path, fs_path_sub, sub_id, mask_output_dir)
        
        # create EVC mask
        if create_EVC_mask:
            mod_fs_labels = create_evc_mask(fs_labels, hemispheres, sub_id, mask_output_dir)
        else:
            mod_fs_labels = fs_labels
        
        # remove overlapping/shared voxels from all masks; note if an EVC mask is created the separate V1 and V2 masks are not considered for overlap removal (as all voxels would be removed)
        mask_list = mod_fs_labels + atlas_annotations
        mask_list = [m.replace("&","_and_") for m in mask_list]
        
        if len(hemispheres) > 1:
            mask_list = ['bh_' + s for s in mask_list]
        else:
            mask_list = [hemispheres[0] + '_' + s for s in mask_list]
        if remove_overlap_strategy == 'Remove':
            # make overlap mask
            make_overlapping_voxels_mask(mask_output_dir, sub_id, mask_list)
            # remove overlap voxels from all masks
            remove_overlapping_voxels_from_masks(mask_output_dir, sub_id, mask_list)
        elif remove_overlap_strategy == 'Smaller':
            # remove overlapping voxels from any bigger of two overlapping masks
            assign_overlapping_voxels_to_smaller_mask(mask_output_dir, sub_id, mask_list)
        else:
            print('. keeping voxels in all masks / no (valid) overlap removal strategy selected')

