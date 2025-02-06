# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:34:23 2023

@author: Oscar Ferrante oscfer88@gmail.com
"""

import os.path as op
import json
import pandas as pd

from config import bids_root


data_path = "/hpc/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/%s/sub-groupphase%s/ses-V1/meg"
rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")

# Get label names
f = open(op.join(rois_deriv_root,
                  "iit_gnw_rois.json"))
gnw_iit_rois = json.load(f)
gnw_labels = ["gnw_"+lab for lab in gnw_iit_rois['surf_labels']['gnw']]
iit_labels = ["iit_"+lab for lab in gnw_iit_rois['surf_labels']['iit_1']]
labels = gnw_labels + iit_labels

# Loop over results and create a summary table
table = pd.DataFrame()
for phase in [2, 3]:
    print(f"phase{phase}")
    for task_rel in ['Irrelevant', 'Relevant non-target']:
        print(f"-{task_rel}")
        for band in ["alpha", "gamma", "ERF"]:
            if phase == 2 and band == "ERF":
                continue
            print(f"---{band}")
            if band == "ERF":
                source_path = "source_dur_ERF_oscar"
            else:
                source_path = "source_dur"
            for tbins in [[[0.8,1.0],[1.3,1.5],[1.8,2.0]], [[1.0,1.2],[1.5,1.7],[2.0,2.2]]]:
                print(f"--{tbins}")
                if tbins == [[1.0,1.2],[1.5,1.7],[2.0,2.2]] and band != "alpha":
                    continue
                for label in labels:
                    print(f"----{label}")
                    bids_path_source = op.join(data_path % (source_path, phase),
                                               f"sub-groupphase{phase}_ses-V1_task-dur_desc-{band},{label},{tbins[0]},{task_rel[:3]}_best_model.tsv")
                    lmm = pd.read_csv(bids_path_source,sep='\t')
                    anova = pd.read_csv(bids_path_source.replace("best_model", "anova"),sep='\t')
                    
                    table = pd.concat([table,
                                        pd.DataFrame([{
                                            "phase": phase,
                                            "task": task_rel,
                                            "band": band,
                                            "tbins": str(tbins),
                                            "label": label,
                                            "best_model": lmm["model"].iloc[-1],
                                            "coefficient-conditions": lmm["coefficient-conditions"].iloc[-1],
                                            "converged": lmm["converged"].iloc[-1],
                                            "conditions": anova.loc[anova["model"] == lmm["model"].iloc[-1], "conditions"].iloc[-1], 
                                            "F-stat": anova.loc[anova["model"] == lmm["model"].iloc[-1], "F-stat"].iloc[-1], 
                                            "NumDF": anova.loc[anova["model"] == lmm["model"].iloc[-1], "NumDF"].iloc[-1], 
                                            "DenomDF": anova.loc[anova["model"] == lmm["model"].iloc[-1], "DenomDF"].iloc[-1], 
                                            "p-value": anova.loc[anova["model"] == lmm["model"].iloc[-1], "p-value"].iloc[-1], 
                                            "bic": lmm["bic"].iloc[-1]
                                            }])])

# Save table
table_fname = op.join(data_path % ("source_dur", phase),
                      "MEG_act_table.tsv")
table.to_csv(table_fname, sep="\t", index=False)