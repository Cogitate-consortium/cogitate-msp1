# -*- coding: utf-8 -*-
"""
===================================
Co00. GED spatial filter
===================================

Create prefrontal spatial filters through generalized eigendecomposition

@author: Oscar Ferrante oscfer88@gmail.com
"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as ss
import argparse
import json

import mne
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CA124',
                    help='site_id + subject_id (e.g. "CA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
# label_list = opt.labels
# label_name = ''.join(label_list)
# parc = opt.parc

debug = False


# Set derivatives paths
prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
fs_deriv_root = op.join(bids_root, "derivatives", "fs")

ged_deriv_root = op.join(bids_root, "derivatives", "ged")
if not op.exists(ged_deriv_root):
    os.makedirs(ged_deriv_root)
ged_figure_root =  op.join(ged_deriv_root,
                            f"sub-{subject_id}",f"ses-{visit_id}","meg",
                            "figures")
if not op.exists(ged_figure_root):
    os.makedirs(ged_figure_root)


# Set task
if visit_id == "V1":
    bids_task = 'dur'
elif visit_id == "V2":
    bids_task = 'vg'
# elif visit_id == "V2":  #find a better way to set the task in V2
#     bids_task = 'replay'
else:
    raise ValueError("Error: could not set the task")


# =============================================================================
# READ DATA
# =============================================================================

def read_cogitate_data(subject_id, visit_id):
    print("Processing subject: %s" % subject_id)

    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root,
        subject=subject_id,
        datatype='meg',
        task=bids_task,
        session=visit_id,
        suffix='epo',
        extension='.fif',
        check=False)

    epochs = mne.read_epochs(bids_path_epo.fpath,
                             preload=False)

    # Pick all non-target trials
    epochs = epochs['Task_relevance in ["Relevant non-target", "Irrelevant"] and Duration != "500ms"']
    if debug:
        epochs = epochs[0:100] # ONLY for DEBUG

    # Load data
    epochs.load_data()

    # Pick MEG sensors only #here I combine channel types
    epochs = epochs.pick('meg')

    return epochs


def select_time_window(epochs):
    # Select epochs
    epochs_acti = epochs.copy().crop(0., .5)
    epochs_base = epochs.copy().crop(-.501, -.001)

    return epochs_acti, epochs_base


# =============================================================================
# SOURCE MODELLING
# =============================================================================

def create_gnw_label():
    # Set path
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")

    # Read labels from FS parc
    if subject_id in ['CA102', 'CA104', 'CA110', 'CA111', 'CA152']:
        labels_atlas = mne.read_labels_from_annot(
            "fsaverage",
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)
    else:
        labels_atlas = mne.read_labels_from_annot(
            "sub-"+subject_id,
            parc='aparc.a2009s',
            subjects_dir=fs_deriv_root)

    # labels_atlas_names = [l.name for l in labels_atlas]

    # Read GNW and IIT ROI list
    f = open(op.join(rois_deriv_root,
                      'iit_gnw_rois.json'))
    gnw_iit_rois = json.load(f)

    # Create labels for selected ROIs
    labels = {}
    if subject_id in ['CA102', 'CA104', 'CA110', 'CA111', 'CA152']:
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            if (lab.find('&') != -1):
                lab = lab.replace('&','_and_')
            print(lab)
            labels["gnw_"+lab+"_lh"] = [l for l in labels_atlas if l.name == lab+"-lh"]
            labels["gnw_"+lab+"_rh"] = [l for l in labels_atlas if l.name == lab+"-rh"]
    else:
        for lab in gnw_iit_rois['surf_labels']['gnw']:
            print(lab)
            labels["gnw_"+lab+"_lh"] = [l for l in labels_atlas if l.name == lab+"-lh"][0]
            labels["gnw_"+lab+"_rh"] = [l for l in labels_atlas if l.name == lab+"-rh"][0]

    # # Show brain with label areas highlighted  #3D plots not working on the hpc
    # if os.getlogin() in ['oscfe', 'ferranto', 'FerrantO']:
    #     brain = mne.viz.Brain(
    #         "sub-"+subject_id,
    #         subjects_dir=fs_deriv_root)
    #     for n, l in labels.items():
    #         brain.add_label(l, color='g')

    #     # Save brain figure in different views
    #     #lateral
    #     brain.show_view('lateral')
    #     brain.save_image(op.join(ged_figure_root,
    #                               'label_gnw_pfc_lat.png'))
    #     #ventral
    #     brain.show_view('ventral')
    #     brain.save_image(op.join(ged_figure_root,
    #                               'label_gnw_pfc_ven.png'))
    #     #caudal
    #     brain.show_view('caudal')
    #     brain.save_image(op.join(ged_figure_root,
    #                               'label_gnw_pfc_cau.png'))
    #     brain.close()

    # Merge all labels in a single one separatelly for GNW and IIT
    label = np.sum([l for l_name, l in labels.items() if 'gnw' in l_name])

    return label


def create_inverse(epochs):
    # Apply baseline correction
    b_tmin = -.501
    b_tmax = -.001
    baseline = (b_tmin, b_tmax)
    epochs.apply_baseline(baseline=baseline)

    # Compute rank
    rank = mne.compute_rank(epochs,
                            tol=1e-6,
                            tol_kind='relative')

    # Read forward model
    bids_path_fwd = mne_bids.BIDSPath(
        root=fwd_deriv_root,
        subject=subject_id,
        datatype='meg',
        task=None,
        session=visit_id,
        suffix='surface_fwd',
        extension='.fif',
        check=False)
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)

    # Compute covariance matrices
    base_cov = mne.compute_covariance(epochs,
                                   tmin=b_tmin,
                                   tmax=b_tmax,
                                   method='empirical',
                                   rank=rank)
    active_cov = mne.compute_covariance(epochs,
                                 tmin=0.,
                                 tmax=None,
                                 method='empirical',
                                 rank=rank)
    common_cov = base_cov + active_cov

    # Compute inverse operator (filter)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info,
        fwd,
        common_cov,
        loose=.2,
        depth=.8,
        fixed=False,
        rank=rank,
        use_cps=True)

    src = inverse_operator['src']

    return inverse_operator ,src


def apply_inverse(epochs, inverse_operator, label, desc):
    # Apply dSPM inverse solution to individual epochs
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method,
        pick_ori="normal",
        label=label,
        )

    # Plot evoked averaged over vertices
    data = np.mean(np.array([stc.data for stc in stcs]), axis=1)
    times = stcs[0].times
    evk_m = np.mean(data, axis=0)
    evk_std = np.std(data,axis=0)

    plt.plot(times, evk_m)
    plt.fill_between(times, evk_m-evk_std, evk_m+evk_std, color='b', alpha=.1)

    # Save figure
    fname_fig = op.join(ged_figure_root,
                        f"ged_stc_evoked_{desc}_gnw.png")
    plt.savefig(fname_fig)
    plt.close()

    return stcs


# =============================================================================
# GED
# =============================================================================

def comp_cov_stcs(stcs):
    # Compute covariance matrices
    cov = []
    #loop over trials
    for stc in stcs:
        #get trial data
        data = stc.data
        #mean-center
        data = data - np.mean(data, axis=1, keepdims=True)
        #compute covariance
        cov_trial = data@data.T / (len(data[0]) - 1)
        #append results to list
        cov.append(cov_trial)

    return cov


def clean_and_average_cov(cov):
    # Clean covariance data from outliers and average trials

    # Average covariance over trials
    cov_m = np.mean(cov, axis=0)

    # Loop over trials
    dists = []
    for i in range(len(cov)):
        # Get data
        tcov = cov[i]
        # Compute euclidean distance
        dists.append(np.sqrt(np.sum((tcov.reshape(1,-1)-cov_m.reshape(1,-1))**2)))

    # Compute z-scored distance
    dists_Z = (dists-np.mean(dists)) / np.std(dists)

    # Average trial-covariances together, excluding outliers
    cov_avg = np.mean( np.asarray(cov)[dists_Z<3] ,axis=0)

    return cov_avg


def apply_reg(cov):
    # Apply regularization
    gamma = .01
    cov_r = cov*(1-gamma) + gamma * np.mean(scipy.linalg.eigh(cov)[0]) * np.eye(len(cov))

    return cov_r


def plot_cov(covSm, covRm):
    # Plot covariance matrices
    fig,axs = plt.subplots(1,3,figsize=(8,4))
    # A matrix
    axs[0].imshow(covSm,vmin=np.min(covSm),vmax=np.max(covSm),cmap='jet')
    axs[0].set_title('S matrix')
    # B matrix
    axs[1].imshow(covRm,vmin=np.min(covRm),vmax=np.max(covRm),cmap='jet')
    axs[1].set_title('R matrix')
    # R^{-1}S
    cov_sxinvr = np.linalg.inv(covRm)@covSm
    axs[2].imshow(cov_sxinvr,vmin=np.min(cov_sxinvr),vmax=np.max(cov_sxinvr),cmap='jet')
    axs[2].set_title('$R^{-1}S$ matrix')
    plt.tight_layout()

    # Save figure
    fname_fig = op.join(ged_figure_root,
                        "ged_covariace_matrices_gnw.png")
    fig.savefig(fname_fig)
    plt.close(fig)

    return fig


def comp_ged(covAm, covBm):
    # Run GED
    evals,evecs = scipy.linalg.eigh(covAm,covBm)

    # Sort eigenvalues/vectors
    sidx  = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:,sidx]

    # Save results
    bids_path_ged = mne_bids.BIDSPath(
        root=ged_deriv_root,
        subject=subject_id,
        datatype='meg',
        task=bids_task,
        session=visit_id,
        suffix='desc-gnw_evals',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, evals)

    bids_path_ged = bids_path_ged.copy().update(
        suffix='desc-gnw_evecs',)
    np.save(bids_path_ged.fpath, evecs)

    return evals, evecs


def plot_ged_evals(evals):
    # Plot the eigenspectrum
    fig = plt.figure()
    plt.plot(evals[0:20],'s-',markersize=15,markerfacecolor='k')
    plt.title('GED eigenvalues')
    plt.xlabel('Component number')
    plt.ylabel('Power ratio (norm-$\lambda$)')

    # Save figure
    fname_fig = op.join(ged_figure_root,
                        "ged_eigenvalues_sorted_gnw.png")
    fig.savefig(fname_fig)
    plt.close(fig)

    return fig


def create_ged_spatial_filter(evecs):
    # Filter forward model
    filt_topo = evecs[:,0]

    # Eigenvector sign
    se = np.argmax(np.abs( filt_topo ))
    filt_topo = filt_topo * np.sign(filt_topo[se])

    # Save results
    bids_path_ged = mne_bids.BIDSPath(
        root=ged_deriv_root,
        subject=subject_id,
        datatype='meg',
        task=bids_task,
        session=visit_id,
        suffix='desc-gnw_filttopo',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, filt_topo)

    return filt_topo


def get_ged_time_course(stcs, evecs):
    comp_ts = []
    # Loop over epochs
    for i in range(len(stcs)):

        # Get data
        data = stcs[i].data

        # Apply GED filter
        comp_ts.append(evecs[:,0].T @ data)

    # Save results
    bids_path_ged = mne_bids.BIDSPath(
        root=ged_deriv_root,
        subject=subject_id,
        datatype='meg',
        task=bids_task,
        session=visit_id,
        suffix='desc-gnw_compts',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, comp_ts)

    return comp_ts


def lowpass_filter(data, order=6, fs=1000.0, cutoff=30.0):
    # Low-pass filter the data
    b, a = ss.butter(order,
                        cutoff,
                        fs=fs,
                        btype='low',
                        analog=False)

    data_filt = ss.lfilter(b, a, data)
    return data_filt


def plot_ged_result(stcs, comp_ts):
    # Low-pass filter
    comp_ts = lowpass_filter(comp_ts)

    # Compute root mean square
    comp_ts_rms = np.sqrt((np.array(comp_ts)**2).mean(axis=0))

    # Baseline correction
    imin = (np.abs(stcs[0].times - -.1)).argmin()  #here I subtract a negative value
    imax = (np.abs(stcs[0].times - 0.)).argmin()

    mean_ts = np.mean(comp_ts_rms[..., imin:imax], axis=-1, keepdims=True)
    comp_ts_rms -= mean_ts
    comp_ts_rms /= mean_ts

    # Crop edges
    tmin = (np.abs(stcs[0].times - -.5)).argmin()  #here I subtract a negative value
    tmax = (np.abs(stcs[0].times - 2.)).argmin()

    comp_ts_rms = comp_ts_rms[tmin:tmax]
    times = stcs[0].times[tmin:tmax]

    # Plot filter time course
    fig = plt.figure()
    plt.plot(times, comp_ts_rms)
    plt.title("GED spatial filters' activity")
    plt.xlabel('time (sec)')
    plt.ylabel('RMS amplitude (a.u.)')

    # Save figure
    fname_fig = op.join(ged_figure_root,
                        "ged_filter_ts_gnw.png")
    fig.savefig(fname_fig)
    plt.close(fig)

    return fig


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    # Read epoched data
    epochs = read_cogitate_data(subject_id, visit_id)

    # Select conditions of interest  #try faces vs. objects
    epochs_acti, epochs_base = select_time_window(epochs)

    # Run source modeling (MNE-dSPM)

    # Create inverse solution
    inverse_operator, src = create_inverse(epochs)

    # Create label for interiortemporal cortex
    label = create_gnw_label()

    # Apply inverse solution
    stcs_acti = apply_inverse(epochs_acti,
                             inverse_operator,
                             label=label,
                             desc='acti')
    stcs_base = apply_inverse(epochs_base,
                               inverse_operator,
                               label=label,
                               desc='base')
    stcs_whole = apply_inverse(epochs,
                             inverse_operator,
                             label=label,
                             desc='whole')

    # Run GED

    # Compute covariance
    cov_acti = comp_cov_stcs(stcs_acti)
    cov_base = comp_cov_stcs(stcs_base)

    # Remove outliers and average
    cov_acti = clean_and_average_cov(cov_acti)
    cov_base = clean_and_average_cov(cov_base)

    # Apply regularization
    cov_base = apply_reg(cov_base)

    # Plot covariance
    plot_cov(cov_acti, cov_base)

    # Run GED
    evals_gnw_pfc, evecs_gnw_pfc = comp_ged(cov_acti, cov_base)

    # Plot GED eigenvalues
    plot_ged_evals(evals_gnw_pfc)

    # Create GED spatial filter
    filt_topo_gnw_pfc = create_ged_spatial_filter(evecs_gnw_pfc)

    # Get GED component time course
    comp_ts_gnw_pfc = get_ged_time_course(stcs_whole, evecs_gnw_pfc)

    # Plot GED spatial filter time course
    plot_ged_result(stcs_whole, comp_ts_gnw_pfc)
