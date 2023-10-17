# -*- coding: utf-8 -*-
"""
===================================
Co00. GED spatial filter
===================================

Create category-specific spatial filters through generalized eigendecomposition

@author: Oscar Ferrante oscfer88@gmail.com
"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as ss
import argparse

import mne
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA124',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--labels',
                    nargs='+',
                    default=['fusifor'],
                    help='name of the label to which contrain the spatial filter (e.g., "fusiform"')
parser.add_argument('--parc',
                    type=str,
                    default='aparc',
                    help='name of the parcellation atlas to use for contraining the spatial filter (e.g., "aparc", "aparc.a2009s")')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
label_list = opt.labels
label_name = ''.join(label_list)
parc = opt.parc

act_win_tmin = 0.
act_win_tmax = .5

debug = False

#aparc              aparc.a2009s
#fusifor            G_oc-temp_lat-fusifor
#inferiortemporal
#lateraloccipital   G&S_occipital_inf


# Set derivatives paths
prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
fs_deriv_root = op.join(bids_root, "derivatives", "fs")

if act_win_tmin == 0. and act_win_tmax == .5:
    ged_deriv_root = op.join(bids_root, "derivatives", "ged")
else:
    ged_deriv_root = op.join(bids_root, "derivatives", "ged", f"_{act_win_tmin}-{act_win_tmax}")
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


def select_cond(cond, epochs):
    # Select epochs
    cond_epochs = epochs['Category == "%s"' % cond]
    other_epochs = epochs['Category != "%s"' % cond]

    return cond_epochs, other_epochs


# =============================================================================
# SOURCE MODELLING
# =============================================================================

def create_inverse(epochs):    
    # Apply baseline correction
    b_tmin = -.5
    b_tmax = - 0.
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


def apply_inverse(epochs, inverse_operator, label):  
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
    
    return stcs


def select_act_win_source(stcs, tmin=0., tmax=.5):
    stcs_act = []
    # Loop over epochs
    for i in range(len(stcs)):
        # Select active time window
        stcs_act.append(stcs[i].copy().crop(tmin, tmax))
    
    return stcs_act


def create_label(label_list, parc):
    labels = []
    # Loop over labels
    for regexp in label_list:
        print("\nReading label "+regexp)
        
        # Create label for the given region
        if subject_id in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
            lab = mne.read_labels_from_annot(
                "fsaverage", 
                parc=parc,  #aparc  aparc.a2009s
                regexp=regexp, #'inferiortemporal'
                hemi='both',
                subjects_dir=fs_deriv_root)
        else:
            lab = mne.read_labels_from_annot(
                "sub-"+subject_id, 
                parc=parc,  #aparc  aparc.a2009s
                regexp=regexp, #'inferiortemporal'
                hemi='both',
                subjects_dir=fs_deriv_root)
            
        # Save label
        bids_path_ged = mne_bids.BIDSPath(
            root=ged_deriv_root, 
            subject=subject_id,  
            datatype='meg',  
            task=None,
            session=visit_id, 
            suffix=f"desc-{regexp}_label-lh",
            extension='.label',
            check=False)
        lab[0].save(bids_path_ged.fpath)
        
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f"desc-{regexp}_label-rh",)
        lab[1].save(bids_path_ged.fpath)
        
        # Append to labels
        labels.append(lab)
    
        # # Show brain with label areas highlighted  #3D plots not working on the hpc
        # if os.getlogin() in ['oscfe', 'ferranto', 'FerrantO']:
        #     if  regexp == label_list[0]:
        #         brain = mne.viz.Brain(
        #             "sub-"+subject_id,
        #             subjects_dir=fs_deriv_root)
        #     brain.add_label(lab[0])
        #     brain.add_label(lab[1])
            
    # # Save brain figure in different views
    # if os.getlogin() in ['oscfe', 'ferranto', 'FerrantO']:
    #     #lateral
    #     brain.show_view('lateral')
    #     brain.save_image(op.join(ged_figure_root,
    #                               f'label_{label_name}_lat.png'))
    #     #ventral
    #     brain.show_view('ventral')
    #     brain.save_image(op.join(ged_figure_root,
    #                               f'label_{label_name}_ven.png'))
    #     #caudal
    #     brain.show_view('caudal')
    #     brain.save_image(op.join(ged_figure_root,
    #                               f'label_{label_name}_cau.png'))
    #     brain.close()
        
    # Combine labels
    label = np.sum(labels)
    
    return label


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


def plot_cov(covSm, covRm, cond):
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
                        f"ged_covariace_matrices_{label_name}_{cond}.png")
    fig.savefig(fname_fig)
    plt.close(fig)
    
    return fig


def comp_ged(covAm, covBm, cond):
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
        suffix=f'desc-{label_name},{cond}_evals',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, evals)
    
    bids_path_ged = bids_path_ged.copy().update(
        suffix=f'desc-{label_name},{cond}_evecs',)
    np.save(bids_path_ged.fpath, evecs)
    
    return evals, evecs


def plot_ged_evals(evals, cond):
    # Plot the eigenspectrum
    fig = plt.figure()
    plt.plot(evals[0:20],'s-',markersize=15,markerfacecolor='k')
    plt.title('GED eigenvalues')
    plt.xlabel('Component number')
    plt.ylabel('Power ratio (norm-$\lambda$)')
    
    # Save figure
    fname_fig = op.join(ged_figure_root,
                        f"ged_eigenvalues_sorted_{label_name}_{cond}.png")
    fig.savefig(fname_fig)
    plt.close(fig)
    
    return fig


def create_ged_spatial_filter(evecs, cond):
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
        suffix=f'desc-{label_name},{cond}_filttopo',
        extension='.npy',
        check=False)
    np.save(bids_path_ged.fpath, filt_topo)
    
    return filt_topo


def get_ged_time_course(stcs, evecs, cond):
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
        suffix=f'desc-{label_name},{cond}_compts',
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


def plot_ged_result(stcs, comp_ts, comp_ts_other, cond):
    # Low-pass filter
    comp_ts = lowpass_filter(comp_ts)
    comp_ts_other = lowpass_filter(comp_ts_other)
    
    # Compute root mean square
    comp_ts_rms = np.sqrt((np.array(comp_ts)**2).mean(axis=0))
    comp_ts_other_rms = np.sqrt((np.array(comp_ts_other)**2).mean(axis=0))
    
    # Baseline correction
    imin = (np.abs(stcs[0].times - -.1)).argmin()  #here I subtract a negative value
    imax = (np.abs(stcs[0].times - 0.)).argmin()
    
    mean_ts = np.mean(comp_ts_rms[..., imin:imax], axis=-1, keepdims=True)
    comp_ts_rms -= mean_ts
    comp_ts_rms /= mean_ts
    
    mean_ts_other = np.mean(comp_ts_other_rms[..., imin:imax], axis=-1, keepdims=True)
    comp_ts_other_rms -= mean_ts_other
    comp_ts_other_rms /= mean_ts_other
    
    # Crop edges
    tmin = (np.abs(stcs[0].times - -.5)).argmin()  #here I subtract a negative value
    tmax = (np.abs(stcs[0].times - 2.)).argmin()
    
    comp_ts_rms = comp_ts_rms[tmin:tmax]
    comp_ts_other_rms = comp_ts_other_rms[tmin:tmax]
    times = stcs[0].times[tmin:tmax]
    
    # Set labels
    if cond == 'face':
        color_cond = 'blue'
        cond_other = 'object'
        color_other = 'orange'
    elif cond == 'object':
        color_cond = 'orange'
        cond_other = 'face'
        color_other = 'blue'
        
    # Plot filter time course
    fig = plt.figure()
    plt.plot(times, comp_ts_rms, 
             label=cond, color=color_cond)
    plt.plot(times, comp_ts_other_rms, 
             label=cond_other, color=color_other)
    plt.legend()
    plt.title("GED spatial filters' activity")
    plt.xlabel('time (sec)')
    plt.ylabel('RMS amplitude (a.u.)')
    
    # Save figure
    fname_fig = op.join(ged_figure_root,
                        f"ged_filter_ts_{label_name}_{cond}.png")
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
    fac_epochs, nofac_epochs = select_cond("face", epochs)
    obj_epochs, noobj_epochs = select_cond("object", epochs)
    
    # Run source modeling (MNE-dSPM)
    
    # Create inverse solution
    inverse_operator, src = create_inverse(epochs)
    
    # Create label for interiortemporal cortex
    label = create_label(label_list, 
                         parc=parc)
    
    # Apply inverse solution
    stcs_fac = apply_inverse(fac_epochs, 
                             inverse_operator, 
                             label=label)
    stcs_nofac = apply_inverse(nofac_epochs, 
                               inverse_operator, 
                               label=label)
    stcs_obj = apply_inverse(obj_epochs, 
                             inverse_operator, 
                             label=label)
    stcs_noobj = apply_inverse(noobj_epochs,
                               inverse_operator, 
                               label=label)
    
    # Run GED
    
    # Select activation (i.e., stimulus presentation) window
    stcs_fac_act = select_act_win_source(stcs_fac, 
                                         tmin=act_win_tmin, 
                                         tmax=act_win_tmax)
    stcs_nofac_act = select_act_win_source(stcs_nofac, 
                                           tmin=act_win_tmin, 
                                           tmax=act_win_tmax)
    stcs_obj_act = select_act_win_source(stcs_obj, 
                                         tmin=act_win_tmin, 
                                         tmax=act_win_tmax)
    stcs_noobj_act = select_act_win_source(stcs_noobj, 
                                           tmin=act_win_tmin, 
                                           tmax=act_win_tmax)
    
    # Compute covariance
    cov_fac = comp_cov_stcs(stcs_fac_act)
    cov_nofac = comp_cov_stcs(stcs_nofac_act)
    cov_obj = comp_cov_stcs(stcs_obj_act)
    cov_noobj = comp_cov_stcs(stcs_noobj_act)
    
    # Remove outliers and average
    cov_fac = clean_and_average_cov(cov_fac)
    cov_nofac = clean_and_average_cov(cov_nofac)
    cov_obj = clean_and_average_cov(cov_obj)
    cov_noobj = clean_and_average_cov(cov_noobj)
    
    # Apply regularization
    cov_nofac = apply_reg(cov_nofac)
    cov_noobj = apply_reg(cov_noobj)
    
    # Plot covariance
    plot_cov(cov_fac, cov_nofac, "face")
    plot_cov(cov_obj, cov_noobj, "object")
    
    # Run GED
    evals_fac, evecs_fac = comp_ged(cov_fac, cov_nofac, "face")
    evals_obj, evecs_obj = comp_ged(cov_obj, cov_noobj, "object")
    
    # Plot GED eigenvalues
    plot_ged_evals(evals_fac, "face")
    plot_ged_evals(evals_obj, "object")
    
    # Create GED spatial filter
    filt_topo_fac = create_ged_spatial_filter(evecs_fac, "face")
    filt_topo_obj = create_ged_spatial_filter(evecs_obj, "object")

    # Get GED component time course
    comp_ts_fac = get_ged_time_course(stcs_fac, evecs_fac, "facFilt_facCond")
    comp_ts_obj = get_ged_time_course(stcs_obj, evecs_obj, "objFilt_objCond")
    comp_ts_fac_on_obj = get_ged_time_course(stcs_obj, evecs_fac, "facFilt_objCond")
    comp_ts_obj_on_fac = get_ged_time_course(stcs_fac, evecs_obj, "objFilt_facCond")
    
    # Plot GED spatial filter time course
    plot_ged_result(stcs_fac, comp_ts_fac, comp_ts_fac_on_obj, "face")
    plot_ged_result(stcs_obj, comp_ts_obj, comp_ts_obj_on_fac, "object")
