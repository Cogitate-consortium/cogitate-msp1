import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import config
from general_utilities import load_fsaverage_coord

param = config.param
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "visual_responsiveness"
sub = "super"
ses = "V1"
data_type = "ieeg"
results_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "results")
fig_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "figure")


def pncc_handler(folder_prefix, categories, time_windows, category_selectivity_file, save_root=None):
    # Read the category selectivity results:
    cate_sel = pd.read_csv(category_selectivity_file)
    # Load each visual responsiveness files:
    results = []
    for cate in categories:
        print(cate)
        for time_win in time_windows:
            print(time_win)
            # Generate the folder name:
            folder = "_".join([folder_prefix, cate, time_win])
            results_path = Path(results_root, folder)
            # Loop through the subdirectories:
            subdirs = [x for x in results_path.iterdir() if x.is_dir()]
            for subdir in subdirs:
                results_files = []
                cond_res = pd.DataFrame()
                for file in glob.glob(str(Path(subdir, '*all_results.csv'))):
                    results_files.append(file)
                # Load this particular set of results:
                try:
                    time_win_cate_res = pd.read_csv(results_files[0])
                except IndexError:
                    print("Warning: the file {} is missing: ".format(folder))
                    continue
                # time_win_cate_res = time_win_cate_res.loc[time_win_cate_res["reject"] == True]
                # Extract channels showing significance to both task relevant and irrelevant trials:
                time_win_cate_res = \
                    time_win_cate_res.loc[time_win_cate_res["condition"] == "both"].reset_index(drop=True)
                if cate == "symbol":
                    cat = "false"
                else:
                    cat = cate
                # Loop through each channel:
                for ch in time_win_cate_res["channel"].to_list():
                    fsize_ti = time_win_cate_res.loc[time_win_cate_res["channel"] == ch,
                                                     "effect_strength-stimulus onset/Irrelevant/" + cat].item()
                    fsize_tr = time_win_cate_res.loc[time_win_cate_res["channel"] == ch,
                                                     "effect_strength-stimulus onset/Relevant non-target/" + cat].item()
                    if fsize_ti > 0 and fsize_tr > 0:
                        direction = "activation"
                    elif fsize_ti < 0 and fsize_tr < 0:
                        direction = "deactivation"
                    else:
                        continue
                    ch_res = time_win_cate_res.loc[time_win_cate_res["channel"] == ch]
                    # Extract the relevant information of the table:
                    cond_res = cond_res.append(pd.DataFrame({
                        "subject": ch_res["subject"].item(),
                        "category": cate,
                        "time_window": time_win,
                        "channel": ch,
                        "direction": direction,
                        "selectivity": cate_sel.loc[cate_sel["channel"] == ch, "condition"].item()
                    }, index=[0]), ignore_index=True)

                results.append(cond_res)

    # Concatenate the results:
    results = pd.concat(results)

    # Load the fsaverage coordinates:
    ch_coords = load_fsaverage_coord(bids_root, results["subject"].unique(), ses='V1', laplace_reloc=True)
    ch_coords = ch_coords.rename(columns={"name": "channel"})

    # Looping through each time window:
    for time_win in time_windows:
        print(time_win)
        # Extract the results from this time window:
        time_win_results = results.loc[results["time_window"] == time_win]
        time_win_coords = pd.DataFrame()
        time_win_colors = pd.DataFrame()
        # Loop through each channel
        for ch in list(cate_sel["channel"].unique()):
            if cate_sel.loc[cate_sel["channel"] == ch, "condition"].item() == \
                    cate_sel.loc[cate_sel["channel"] == ch, "condition"].item():
                color = param["colors"][cate_sel.loc[cate_sel["channel"] == ch, "condition"].item()]
            elif ch in time_win_results["channel"].to_list():
                if time_win_results.loc[time_win_results["channel"] == ch, "direction"].to_list()[0] == "activation":
                    color = [1, 0, 0]
                else:
                    color = [0, 0, 1]
            else:
                continue
            try:
                time_win_coords = time_win_coords.append(pd.DataFrame({
                    "channel": ch,
                    "x": ch_coords.loc[ch_coords["channel"] == ch, "x"].item(),
                    "y": ch_coords.loc[ch_coords["channel"] == ch, "y"].item(),
                    "z": ch_coords.loc[ch_coords["channel"] == ch, "z"].item(),
                    "size": 2
                }, index=[0]), ignore_index=True)
            except ValueError:
                print("A")
            time_win_colors = time_win_colors.append(pd.DataFrame({
                "channel": ch,
                "r": color[0],
                "g": color[1],
                "b": color[2]
            }, index=[0]), ignore_index=True)
        # Save the results:
        time_win_coords.to_csv(Path(save_root, "coords_{}.csv".format(time_win)))
        time_win_colors.to_csv(Path(save_root, "coords_colors_{}.csv".format(time_win)))

    return


if __name__ == "__main__":
    subjects_list = None
    folder_prefix = "high_gamma_wilcoxon_onset_two_tailed"
    categories = ["face", "object", "letter", "symbol"]
    time_windows = ["200_300", "300_400", "400_500"]
    category_selectivity_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/" \
                                "category_selectivity/sub-super/ses-V1/ieeg/results/high_gamma_dprime_test/" \
                                "desbadcharej_notfil_lapref/" \
                                "sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv"
    pncc_handler(folder_prefix,
                 categories,
                 time_windows,
                 category_selectivity_file,
                 save_root="/hpc/users/alexander.lepauvre/plotting_test/pncc")
