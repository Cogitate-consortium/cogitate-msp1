"""
=================
08. Forward model
=================

Compute the forward model

Step 1 - Freesurfer recontruction
Step 2 - MNE-python scalp surface reconstruction
Step 3 - Get Boundary Element Model (BEM)
Step 4 - Coregistration
Step 5 - Compute source space
Step 6 - Forward Model

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
# import numpy as np
import argparse

import mne


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA124',
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


# Step 1 - Compute source space
def make_source_space(space):
    '''
    Compute source space
    
    Surface-based source space is computed using:
    > mne.setup_source_space()
    Volumetric source space is computed using:
    > mne.setup_volume_source_space()
    
    '''
    if space == 'surface':
        # Get surface-based source space
        spacing='oct6'  # 4098 sources per hemisphere, 4.9 mm spacing
        src = mne.setup_source_space(subject,
                                     spacing=spacing,
                                     add_dist='patch',
                                     subjects_dir=subjects_dir)
        # Set filename
        fname_src = '%s-surface%s_src.fif' % (subject, spacing)
    elif space == 'volume':
        # Get volumetric source space (BEM required)
        surface = op.join(subjects_dir, subject, 
                          'bem', 'inner_skull.surf')
        src = mne.setup_volume_source_space(subject,
                                            subjects_dir=subjects_dir,
                                            surface=surface,
                                            mri='T1.mgz',
                                            verbose=True)
        # Set filename
        fname_src = '%s-volume_src.fif' % (subject)
    # Save source space
    mne.write_source_spaces(op.join(subjects_dir,subject,fname_src),
                            src,
                            overwrite=True)
    # Visualize source space and BEM
    mne.viz.plot_bem(subject=subject, 
                     subjects_dir=subjects_dir,
                     brain_surfaces='white', 
                     src=src, 
                     orientation='coronal')
    # # Visualize sources in 3d space
    # if space == 'surface':  #TODO: doesn't work with volume space
    #     fig = mne.viz.plot_alignment(subject=subject, 
    #                                   subjects_dir=subjects_dir,
    #                                   trans=trans,
    #                                   surfaces='white', 
    #                                   coord_frame='head',
    #                                   src=src)
    #     mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
    #                         distance=0.35, focalpoint=(-0.03, 0.01, 0.03))
    return src


# Step 2 - Forward Model
def make_forward_model(src, task):
    '''
    Forward Model
    
    '''
    
    # Set path to raw FIF
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-"+visit+"_task-"+task+"_run-01_meg.fif")

    
    # Set transformation matrix and bem pathes
    trans = op.join(fname_coreg, subject+"_ses-"+visit+"_trans.fif")
    bem = op.join(subjects_dir, subject, subject+"_ses-V1_bem-sol.fif") #BEM is shared between sessions
    
    # Calculate forward solution for MEG channels
    fwd = mne.make_forward_solution(fname_raw, 
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
    # Number of vertices
    print(f'\nNumber of vertices:  {fwd["src"]}')
    # Leadfield size
    leadfield = fwd['sol']['data']
    print("\nLeadfield size : %d sensors x %d dipoles" % leadfield.shape)
    return fwd


# RUN
if __name__ == "__main__":
    src = make_source_space(space)
    if visit == 'V1':
        make_forward_model(src, 'dur')
    elif visit == 'V2':
        make_forward_model(src, 'vg')
        make_forward_model(src, 'replay')
