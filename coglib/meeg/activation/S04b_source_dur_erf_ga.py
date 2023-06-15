"""
================
S05. Grand-average source epochs
================

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import numpy as np
import argparse
import itertools
import pandas as pd

import mne_bids

from config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--method',
                    type=str,
                    default='dspm',
                    help='method used for the inverse solution')
opt=parser.parse_args()


# Set params
inv_method = opt.method
visit_id = "V1"

debug = False


factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant non-target','Irrelevant'],
              ['500ms', '1000ms', '1500ms']]


# Set participant list
phase = 3

if debug:
    sub_list = ["SA124", "SA124"]
else:
    # Read the .txt file
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")


def source_dur_ga():
    # Set directory paths
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur_ERF")
    if not op.exists(source_deriv_root):
        os.makedirs(source_deriv_root)
    source_figure_root =  op.join(source_deriv_root,
                                f"sub-groupphase{phase}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(source_figure_root):
        os.makedirs(source_figure_root)
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    # Find all combinations between variables' levels
    if len(factor) == 1:
        cond_combs = list(itertools.product(conditions[0]))
    if len(factor) == 2:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1]))
    if len(factor) == 3:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1],
                                            conditions[2]))
    
    # Read list with label names
    bids_path_source = mne_bids.BIDSPath(
                        root=source_deriv_root[:-10], 
                        subject=sub_list[0],  
                        datatype='meg',  
                        task=bids_task,
                        session=visit_id, 
                        suffix="desc-labels",
                        extension='.txt',
                        check=False)
    
    labels_names = open(bids_path_source.fpath, 'r').read()
    labels_names = labels_names[2:-2].split("', '")
    
    # Create empty dataframe
    all_data_df = pd.DataFrame()
    
    # Loop over conditions of interest
    for cond_comb in cond_combs:
        print("\n\nAnalyzing %s: %s" % (factor, cond_comb))
        
        # Select epochs
        if len(factor) == 1:
            fname = cond_comb[0]
        if len(factor) == 2:
            fname = cond_comb[0] + "_" + cond_comb[1]
        if len(factor) == 3:
            fname = cond_comb[0] + "_" + cond_comb[1] + "_" + cond_comb[2]
        
        # Loop over labels
        for label in labels_names:
            print('\nlabel:', label)
            
            # Create empty list
            label_data = []
            
            # Loop over participants
            for sub in sub_list:
                print('subject:', sub)
                
                # Read individual dataframe
                try:
                    bids_path_source = bids_path_source.update(
                        root=source_deriv_root, 
                        subject=sub,  
                        suffix=f"desc-{fname},ERF,{label}_datatable",
                        extension='.tsv',
                        check=False)
                    df = pd.read_csv(bids_path_source.fpath, sep="\t")
                except:
                    label_r = label.replace("&", "_and_")
                    bids_path_source = bids_path_source.update(
                        root=source_deriv_root, 
                        subject=sub,  
                        suffix=f"desc-{fname},ERF,{label_r}_datatable",
                        extension='.tsv',
                        check=False)
                    df = pd.read_csv(str(bids_path_source.fpath), sep="\t")
                
                # Append dataframe to list
                label_data.append(df['data'])
                    
            # Create table with the extracted label time course data
            label_data_df = pd.DataFrame(sub_list,columns=['sub'])
            label_data_df = pd.concat(
                [label_data_df,
                 pd.DataFrame(
                    np.array(label_data), 
                    columns=df['times'])],
                axis=1)
            
            # Add info to the table regarding the conditions
            if len(factor) == 1:
                label_data_df[factor[0]] = cond_comb[0]
            if len(factor) == 2:
                label_data_df[factor[0]] = cond_comb[0]
                label_data_df[factor[1]] = cond_comb[1]
            if len(factor) == 3:
                label_data_df[factor[0]] = cond_comb[0]
                label_data_df[factor[1]] = cond_comb[1]
                label_data_df[factor[2]] = cond_comb[2]
            
            label_data_df['band'] = "ERF"
            label_data_df['label'] = label
            
            # Append label table to data table
            all_data_df = all_data_df.append(label_data_df)
            
    # Save table as .tsv
    bids_path_source = bids_path_source.copy().update(
        root=source_deriv_root,
        subject=f"groupphase{phase}",
        suffix="datatable",
        check=False)
    all_data_df.to_csv(bids_path_source.fpath, 
                       sep="\t",
                       index=False)


if __name__ == '__main__':
    source_dur_ga()
