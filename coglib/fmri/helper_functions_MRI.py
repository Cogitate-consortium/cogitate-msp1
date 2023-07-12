#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 12.10.2020

@author: David Richter (d.richter@donders.ru.nl), Yamil Vidal (hvidaldossantos@gmail.com)

Various helper functions for fMRI analyses.

"""


# %% Imports & parameters
import numpy as np
import os

# %% Helper functions for data handling; BIDS, fmriprep processed data, subject lists, running sub processes
def get_subject_list(bids_dir,list_type='all'):
    """
    Gets subject list from bids compliant participants tsv file.
    bidsDir: Bids directory
    list_type: which list should be retrieved. 
    By default all are returned
    Returns: subjects list containing subjects
    """
    import pandas as pd
    # get subject list (determines on which subjects scripts are run)
    if list_type == 'all_p':
        fname_suffix = 'participants.tsv'
    elif list_type == 'optimization_exp1':
        fname_suffix = 'optimization_exp1.tsv'
    elif list_type == 'replication_exp1':
        fname_suffix = 'replication_exp1.tsv'
    elif list_type == 'debug':
        fname_suffix = 'debug_participants.tsv'
    elif list_type == 'demo':
        fname_suffix = 'participants_demo.tsv'
    else:
        raise ValueError('Invalid subject list type specified. No subject list returned')
    subject_list = bids_dir + os.sep + fname_suffix
    subj_df = pd.read_csv(subject_list, sep="\t")
    # subjects = subj_df['participant_id'].values
    subjects = 'sub-'+subj_df['participant_id'].values
    print('Number of Subjects:', subjects.size)
    return subjects
	
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


# %% Helper functions for log file extraction
def saveEventFile(fname,event,fill_empty=False):
    """
    Save event file (.txt) to dir. Create dir if necessary.
    If event information is empty zeros can be added if fill_empty is True.
    This can be helpful to avoid FEAT crashing on empty event files.
    """
    if len(event)==0 and fill_empty:
        event = np.array([[0,0,0]])
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savetxt(fname, event, fmt='%f')
    
def saveErrorFlags(fname, errLogDf):
    """
    Save file with error flags (.csv) to dir. Create dir if necessary.
    """
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    errLogDf.to_csv(fname, sep=',', index=False, na_rep='Null')

	
# %% MRI data handling functions
def load_mri(func, mask):
    """returns functional data

    The data is converted into a 2D (n_voxel, n_tps) array.

    Parameters
    ----------
    func : string
        Path to fuctional imaging data (e.g. nifti) that contains the 3D + time
        information (n_tps).
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.

    Returns
    -------
    ts : ndarray, shape(n, n_tps)
        Timeseries information in a 2D array.
    """
    import nibabel as nib
    
    # load mask data
    m = nib.load(mask).get_fdata()

    # load func data
    d = nib.load(func).get_fdata()

    # mask the data
    return np.asarray(d[m != 0], dtype=np.float32)
    #return np.asarray(d, dtype=np.float32)
    


def save_mri(data, mask, fname=None):
    """saves MRI data

    Parameters
    ----------
    data : ndarray, shape(n,) **or** shape(n, n_tps)
       Data to save to disk.
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.
    fname : string
        Filename.
    """
    import nibabel as nib

    # load mask data for spatial information
    f = nib.load(mask)
    m = f.get_data()
    aff = f.get_affine()

    s = m.shape
    if len(data.shape) == 2:
        n_tps = data.shape[1]
    else:
        n_tps = 1
        data = data[:, np.newaxis]

    res = np.zeros((s[0], s[1], s[2], n_tps)) # + time
    res[m != 0] = data

    # save to disk
    if not fname is None:
        nib.save(nib.Nifti1Image(res, aff), fname)


def pval_to_zscore(data, two_sided=True, inv=False):
    """convert p-values to z-scores (and the inverse)
    Parameters
    -----------
    data : ndarray, shape(n, )
        Data values
    two_sided : boolean
        Values from two-sided test (default=True).
    inv : boolean
        Inverse transformation: convert z-scores to p-values (default=False).
    
    Returns
    -------
    v : ndarray, shape(n, )
        Transformed data values.
    """
    from scipy.stats import norm
    
    if two_sided:
        mul = 2.
    else:
        mul = 1.

    if inv:
        # zscore --> pval
        v = norm.cdf(-np.abs(data)) * mul

        # TODO better use survival function?
        # v  = norm.sf(data) * mul
    else:
        # pval --> zscore
        v = np.abs(norm.ppf(data / mul))

    return v
