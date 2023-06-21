"""
=================
S00. BEM (and coregistration)
=================

Perform the automated coregistration:

Step 1 - Visualize Freesurfer parcellation
Step 2 - MNE-python scalp surface reconstruction
Step 3 - Boundary Element Model (BEM) reconstruction
Step 4 - Get Boundary Element Model (BEM)
(Step 5 - Coregistration)

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
                    default='SA101',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
                    help='Path to the BIDS root directory')
parser.add_argument('--fs_path',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
opt=parser.parse_args()


# Set params
subject = "sub-"+opt.sub
visit = opt.visit
subjects_dir = opt.fs_path
if visit == "V1":
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-V1_task-dur_run-01_meg.fif")
elif visit == "V2":
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-V2_task-vg_run-01_meg.fif")  #TODO: to be tested
coreg_deriv_root = op.join(opt.bids_root, "derivatives", "coreg")
if not op.exists(coreg_deriv_root):
    os.makedirs(coreg_deriv_root)
coreg_figure_root =  op.join(coreg_deriv_root,
                            f"sub-{opt.sub}",f"ses-{visit}","meg",
                            "figures")
if not op.exists(coreg_figure_root):
    os.makedirs(coreg_figure_root)

# Step 1 - Freesurfer recontruction (only on Linux/MACos)
def viz_fs_recon():
    '''
    Freesurfer recontruction (only on Linux/MACos)
    
    Run the following command in a terminal:
    > recon-all -i SA101.nii -s SA101 -all
    For more info, go to https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all/
    
    To convert DICOM to NIFTI, use MRIcron
    
    '''
    # Visualize reconstruction:
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject, 
                  hemi='lh', 
                  surf='pial',
                  subjects_dir=subjects_dir, 
                  size=(800, 600))
    brain.add_annotation('aparc', borders=False)  #aparc.a2009s

    # Save figure
    fname_figure = op.join(subjects_dir, "fs_aparc.png")
    brain.save_image(fname_figure)
    
    
# Step 2 - Scalp surface reconstruction
def make_scalp_surf():
    '''
    Scalp surface reconstruction
    
    Either use this function ot run the following commands in a terminal:
    > mne make_scalp_surfaces --overwrite --subject SA101 --force
    
    
    '''
    mne.bem.make_scalp_surfaces(subject, 
                                subjects_dir=subjects_dir, 
                                force=True, 
                                overwrite=True, 
                                verbose=True)

    
# Step 3 - Boundary Element Model (BEM) reconstruction
def make_bem():
    '''
    Boundary Element Model (BEM)
    
    To create the BEM, either use this function or run the following command
    in a terminal (requires FreeSurfer):
    > mne watershed_bem --overwrite --subject ${file}
    
    '''
    mne.bem.make_watershed_bem(subject, 
                               subjects_dir=subjects_dir, 
                               overwrite=True, 
                               verbose=True)
    
    
# Step 4 - Get Boundary Element Model (BEM) solution
def get_bem():
    '''
    Make Boundary Element Model (BEM) solution
    
    Computing the BEM surfaces requires FreeSurfer and is done using the 
    following command:
    > mne watershed_bem --overwrite --subject SA101
    
    Once the BEM surfaces are read, create the BEM model
    
    '''
    # Create BEM model
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject,
                               ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    
    # Finally, the BEM solution is derived from the BEM model
    bem = mne.make_bem_solution(model)
    
    # Save data
    fname_bem = op.join(subjects_dir, subject, subject+"_ses-"+visit+"_bem-sol.fif")
    mne.write_bem_solution(fname_bem,
                           bem,
                           overwrite=True)
    # Visualize the BEM
    fig = mne.viz.plot_bem(subject=subject,
                           subjects_dir=subjects_dir,
                           #brain_surfaces='white',
                           orientation='coronal')
    
    # Save figure
    fname_figure = op.join(subjects_dir, subject, "bem-sol.png")
    fig.savefig(fname_figure)
    
    return bem


# # Step 5 - Coregistration
# def coreg():
#     '''
#     Coregistration
    
#     Tutorial: https://www.slideshare.net/mne-python/mnepython-coregistration
    
#     To get the path of MNE sample data, run:
#     > mne.datasets.sample.data_path()
    
#     Save fiducials as:
#         SA101_MRI-fiducials
    
#     At the end of the coregistration, save the transformation matrix and 
#     rename the file following the naming convention (see example below)
#         SA101-trans.fif
    
#     To open the coregistration GUI, run:
#     > mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
#     or run "mne coreg" from the terminal
    
#     '''
#     # Automated coregistration
#     info = mne.io.read_info(fname_raw)
#     fiducials = "estimated"  # get fiducials from fsaverage
#     coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    
#     # Fit using 3 fiducial points
#     coreg.fit_fiducials(verbose=True)
    
#     # Refine the transformation using the Iterative Closest Point (ICP) algorithm
#     coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True)
    
#     # Remove outlier points
#     coreg.omit_head_shape_points(distance=5. / 1000)
    
#     # Do a final coregistration fit
#     coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)
    
#     # Compute the distance error
#     dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
#     print(
#         f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
#         f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
#     )
    
#     # Save transformation matrix
#     fname_trans = op.join(coreg_deriv_root, subject+"_ses-"+visit+"_trans.fif")
#     mne.write_trans(fname_trans, coreg.trans)
    
#     # # Visualize the transformation   #TODO: 3d plots don't work on the HPC
#     # fig = mne.viz.plot_alignment(info, coreg.trans, subject=subject, dig=True,
#     #                        meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
#     #                        surfaces='head-dense')
    
#     # # Save figure
#     # fname_figure = op.join(coreg_figure_root, "coreg.png")
#     # fig.savefig(fname_figure)
    
#     return coreg.trans


if __name__ == "__main__":
    # viz_fs_recon()  #TODO: 3d plots don't work on the HPC
    make_scalp_surf()
    make_bem()
    bem = get_bem()
    # coreg()
    