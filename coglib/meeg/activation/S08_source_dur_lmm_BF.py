#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:48:12 2025

@author: oscar.ferrante
"""

import pandas as pd
import os.path as op
import json

from bayes_factor_fun import bic_to_bf10

from config import bids_root


data_path = "/hpc/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/source_dur/sub-groupphase3/ses-V1/meg"
rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")


def lmm_bf(results_file, h0s, h1s):
    """
    Compute Bayes Factors (BF) for linear mixed models (LMM) using BIC values from a results file.
    Parameters:
    -----------
    results_file : str
        Path to the CSV file containing BIC values for different models.
    h0s : list of str
        List of null hypothesis model names.
    h1s : list of str
        List of alternative hypothesis model names.
    Returns:
    --------
    bf_results : pd.DataFrame
        DataFrame containing the Bayes Factors for each channel and model comparison.
    """
    # Ensure inputs are lists
    if not isinstance(h0s, list):
        h0s = [h0s]
    if not isinstance(h1s, list):
        h1s = [h1s]
    # Load labels
    f = open(op.join(rois_deriv_root,
                      "iit_gnw_rois.json"))
    gnw_iit_rois = json.load(f)
    gnw_labels = ["gnw_"+lab for lab in gnw_iit_rois['surf_labels']['gnw']]
    iit_labels = ["iit_"+lab for lab in gnw_iit_rois['surf_labels']['iit_1']]
    labels = gnw_labels + iit_labels
    # Create results table
    bf_results = pd.DataFrame()
    # Loop through each H0 and H1 model pair
    for h0 in h0s:
        for h1 in h1s:
            # Process each unique label
            for label in labels:
                # Load the results
                label_results = pd.read_csv(results_file % label, sep='\t')
                # Extract BIC values for the models
                h1_bic = label_results.loc[label_results['model'] == h1, 'bic'].iloc[0]
                h0_bic = label_results.loc[label_results['model'] == h0, 'bic'].iloc[0]
                # Compute BF
                bf10 = bic_to_bf10(h1_bic, h0_bic)
                # Append results to the DataFrame
                bf_results = pd.concat([
                    bf_results,
                    pd.DataFrame({
                        'label': label,
                        'h1': h1,
                        'h0': h0,
                        'bf01': 1/bf10
                    }, index=[0])
                ], ignore_index=True)
    return bf_results

if __name__ == "__main__":
    # Loop over frequency bands
    for band in ['alpha', 'gamma']:
        # Compute Bayes Factors for the specified models and save results to a CSV file
        bf_results = lmm_bf(
            f'/hpc/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/source_dur/sub-groupphase3/ses-V1/meg/sub-groupphase3_ses-V1_task-dur_desc-{band},%s,[0.8, 1.0],Irr_lmm.tsv',
            ['null_model', 'time_win_dur'],
            ['time_win_dur_gnw', 'time_win_dur_cate_gnw']
        )
        bf_results.to_csv(op.join(data_path, f'BF_LMM_gnw_{band}.csv'), index=False)
        
        bf_results = lmm_bf(
            f'/hpc/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/source_dur/sub-groupphase3/ses-V1/meg/sub-groupphase3_ses-V1_task-dur_desc-{band},%s,[0.8, 1.0],Irr_lmm.tsv',
            ['null_model', 'time_win_dur'],
            ['time_win_dur_iit', 'time_win_dur_cate_iit']
        )
        bf_results.to_csv(op.join(data_path, f'BF_LMM_iit_{band}.csv'), index=False)