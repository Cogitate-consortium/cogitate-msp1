"""
================
S03. Grand-average source localization
================

Grand-average of source localization.

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
    
import mne
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--method',
                    type=str,
                    default='dspm',
                    help='method used for the inverse solution ("lcmv", "dics", "dspm")')
parser.add_argument('--band',
                    type=str,
                    default='gamma',
                    help='frequency band of interest ("alpha", "beta", "gamma")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
inv_method = opt.method
fr_band = opt.band
visit_id = "V1"

debug = False


# Set participant list
phase = 3

if debug:
    sub_list = ["SA124", "SA126"]
else:
    # Read the .txt file
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")


def source_loc_ga():
    # Set directory paths
    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    
    stfr_deriv_root = op.join(bids_root, "derivatives", "source_loc")
    if not op.exists(stfr_deriv_root):
        os.makedirs(stfr_deriv_root)
    stfr_figure_root =  op.join(stfr_deriv_root,
                                f"sub-groupphase{phase}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(stfr_figure_root):
        os.makedirs(stfr_figure_root)
    
    # Set file name ending
    if inv_method in ["lcmv", "dics"]:
        fname_end = 'vl'
    elif inv_method == "dspm":
        fname_end = "lh"
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    # Load average source space
    if inv_method in ["lcmv", "dics"]:
        fname_fs_src = op.join(fs_deriv_root, 'fsaverage/bem/fsaverage-vol-5-src.fif')
    elif inv_method == "dspm":
        fname_fs_src = op.join(fs_deriv_root, 'fsaverage/bem/fsaverage-ico-5-src.fif')
    src_fs = mne.read_source_spaces(fname_fs_src)
    
    # Loop over frequency bands
    for fr_band in ['alpha', 'gamma']:
        # Loop over conditions
        stcs = {}
        for condition in range(1,3):
    
            # Pick condition
            if condition == 1:
                cond_name = "relevant non-target"
            elif condition == 2:
                cond_name = "irrelevant"
            else:
                raise ValueError("Condition %s does not exists" % condition)
            
            print(f"\Task {cond_name}")
            
            # Load data
            stcs_temp = []
            for sub in sub_list:
                print("participant:", sub)
                
                # Set path
                bids_path_sou = mne_bids.BIDSPath(
                    root=stfr_deriv_root,
                    subject=sub,  
                    datatype='meg',  
                    task=bids_task,
                    session=visit_id, 
                    suffix=f"stfr_beam-{inv_method}_band-{fr_band}_c-{cond_name}-{fname_end}",
                    extension=".stc",
                    check=False)
                
                # Load stc data
                stc = mne.read_source_estimate(bids_path_sou)
                
                # Read forward solution
                bids_path_fwd = bids_path_sou.copy().update(
                        root=fwd_deriv_root,
                        task=None,
                        suffix="surface_fwd",
                        extension='.fif',
                        check=False)
                
                fwd = mne.read_forward_solution(bids_path_fwd.fpath)
                
                # Morph to fsaverage
                if sub not in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
                    morph = mne.compute_source_morph(
                        fwd['src'], 
                        subject_from="sub-"+sub, 
                        subject_to='fsaverage', 
                        src_to=src_fs, 
                        subjects_dir=fs_deriv_root,
                        verbose=True)
                
                    stc = morph.apply(stc)
                
                # Append to temp stc list
                stcs_temp.append(stc)
            
            # Appenmd to full stcs list
            stcs[cond_name] = stcs_temp
            
            del stc, stcs_temp
            
            # Average stcs across participants
            stcs_data = [stc.data for stc in stcs[cond_name]]
            stc_ga = stcs[cond_name][0]
            stc_ga.data = np.mean(stcs_data, axis=0)
            
            # Save stc grandaverage
            bids_path_sou = bids_path_sou.update(
                subject=f"groupphase{phase}")
            stc_ga.save(bids_path_sou)


def plot_source_loc_ga(stc_path, desc=None, lims=None, hemi="lh", surface='pial',
                       size=(800, 600), colormap="RdYlBu_r", colorbar=False,
                       transparent=False, background="white", subject="fsaverage", 
                       subjects_dir=r'C:\Users\ferranto\Desktop\fs'):
    
    # Load stc data
    stc_ga = mne.read_source_estimate(stc_path)
    
    # Set view
    if desc:
        views = ['lateral', 'ventral', 'caudal', 'medial']
    else:
        views = ['lateral']
    
    # Set limits
    if lims == None:
        if desc == "alpha":
            lims=[0.8,1.,1.2]
        elif desc == "gamma":
            lims=[0.95,1.,1.05]
        else:
            lims=[0.8,1.,1.2]
    
    # Plot source estimates
    for view in views:
        fig = stc_ga.plot(
            hemi=hemi,
            views=view,
            surface=surface,
            size=size,
            colormap=colormap,
            colorbar=colorbar,
            transparent=transparent,
            background=background,
            subject=subject, 
            clim={"kind": "value", "lims": lims},
            subjects_dir=subjects_dir)
        
        # Save figure
        if desc:
            fname_fig = op.join(op.dirname(stc_path),
                                f"stfr_{desc}_{view}.png")
            fig.save_image(fname_fig)
            fig.close()
    
    # Plot colorbar separately
    if desc:
        # Create a figure and axes object
        fig, ax = plt.subplots(figsize=[3, 2])
        
        # Create a colorbar
        norm = matplotlib.colors.Normalize(vmin=lims[0], vmax=lims[-1])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
            aspect=10, ax=ax)
        
        # Get the axes object for the colorbar
        cb_ax = cb.ax
        
        # Remove the other axes objects from the figure
        for ax in fig.axes:
            if ax != cb_ax:
                ax.remove()
        
        # Save the figure
        fname_fig = os.path.join(os.path.dirname(stc_path),
                                  f"stfr_{desc}__colorbar.png")
        fig.savefig(fname_fig, dpi=300)
        fname_fig = os.path.join(os.path.dirname(stc_path),
                                  f"stfr_{desc}__colorbar.svg")
        fig.savefig(fname_fig, dpi=300)
        plt.close()
    
    return fig


if __name__ == '__main__':
    source_loc_ga()
