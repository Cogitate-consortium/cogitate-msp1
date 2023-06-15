# -*- coding: utf-8 -*-
"""
=================
S01. Forward model with MRI template
=================

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import argparse

import mne


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA101',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--space',
                    type=str,
                    default='surface',
                    help='source space ("surface" or "volume")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
                    help='Path to the BIDS root directory')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--coreg_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/coreg',
                    help='Path to the coreg (derivative) directory')
parser.add_argument('--out_fw',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward',
                    help='Path to the forward (derivative) directory')
opt=parser.parse_args()


# Set params
subject = "sub-"+opt.sub
visit = opt.visit
space = opt.space

subjects_dir = opt.fs_path
fname_coreg = op.join(opt.coreg_path, subject, "ses-"+visit, "meg")

fpath_fw = op.join(opt.out_fw, subject, "ses-"+visit, "meg")
if not op.exists(fpath_fw):
    os.makedirs(fpath_fw)

# fpath_fig = op.join(fpath_fw, "figures")
# if not op.exists(fpath_fig):
#     os.makedirs(fpath_fig)

def make_forward_model_from_template(task):
    
    # Set path to raw FIF
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-"+visit+"_task-"+task+"_run-01_meg.fif")

    # Set path to template files:
    subj = 'fsaverage'
    trans = 'fsaverage'
    if space == 'surface':
        src = op.join(subjects_dir, subj, 'bem', 'fsaverage-ico-5-src.fif')
        bem = op.join(subjects_dir, subj, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    
    # Load raw
    raw = mne.io.read_raw(fname_raw, preload=True)
    
    # # Check that the locations of sensors is correct with respect to MRI
    # mne.viz.plot_alignment(
    #     raw.info, src=src, trans=trans,
    #     subjects_dir=subjects_dir,
    #     show_axes=True, mri_fiducials=True, dig='fiducials')
    
    # Setup source space and compute forward
    fwd = mne.make_forward_solution(raw.info, 
                                    trans=trans, 
                                    src=src, 
                                    bem=bem,
                                    meg=True, eeg=False, 
                                    mindist=5.,
                                    verbose=True)
    
    # Save forward model
    fname_fwd = op.join(fpath_fw, subject+"_ses-"+visit+"_task-"+task+"_%s_fwd.fif" % space)
    mne.write_forward_solution(fname_fwd,
                               fwd,
                               overwrite=True)


# RUN
if __name__ == "__main__":
    if visit == 'V1':
        make_forward_model_from_template('dur')
    elif visit == 'V2':
        make_forward_model_from_template('vg')
        make_forward_model_from_template('replay')
