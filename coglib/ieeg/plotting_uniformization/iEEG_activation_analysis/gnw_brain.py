import pandas as pd
import os
from pathlib import Path
import theories_rois
import config
from general_utilities import load_fsaverage_coord, get_ch_rois

param = config.param
rois = theories_rois.rois
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"


def gnw_channels(vis_resp_file, cat_sel_file, lmm_file, save_root=""):
    """

    """
    vis_resp_results = pd.read_csv(vis_resp_file)
    cate_sel_results = pd.read_csv(cat_sel_file)
    lmm_results = pd.read_csv(lmm_file)

    # Extract only the GNW channels from the cate sel and vis resp:
    cate_sel_results = cate_sel_results.loc[cate_sel_results["channel"].isin(list(lmm_results["group"].unique()))]
    vis_resp_results = vis_resp_results.loc[vis_resp_results["channel"].isin(list(lmm_results["group"].unique()))]
    subjects_list = list(lmm_results["subject"].unique())
    # Get the loc of these channels:
    ch_coords = load_fsaverage_coord(bids_root, subjects_list, ses='V1', laplace_reloc=True)
    ch_coords = ch_coords.loc[ch_coords["name"].isin(list(lmm_results["group"].unique()))]
    ch_rois = get_ch_rois(bids_root, subjects_list, ses='V1', laplace_reloc=True)
    # Loop through each channel to get the color it should get:
    ch_coords_new = pd.DataFrame()
    ch_rois_new = pd.DataFrame()
    ch_colors = pd.DataFrame()
    ch_edge_color = pd.DataFrame()
    for ch in ch_coords["name"].to_list():
        cate = cate_sel_results.loc[cate_sel_results["channel"] == ch, "condition"].item()
        vis_resp = vis_resp_results.loc[vis_resp_results["channel"] == ch, "reject"].item()
        if ch == "SF104-G12":
            ch_coords_new = ch_coords_new.append(pd.DataFrame({
                "channel": ch,
                "x": ch_coords.loc[ch_coords["name"] == ch, "x"].item(),
                "y": ch_coords.loc[ch_coords["name"] == ch, "y"].item(),
                "z": ch_coords.loc[ch_coords["name"] == ch, "z"].item(),
                "radius": 4
            }, index=[0]))
            ch_colors = ch_colors.append(pd.DataFrame({
                "channel": ch,
                "r": 0.1,
                "g": 0.1,
                "b": 0.1
            }, index=[0]))
            ch_rois_new = ch_rois_new.append(pd.DataFrame({
                "channel": ch,
                "roi": ch_rois.loc[ch_rois["channel"] == ch, "region"].item()
            }, index=[0]))
            ch_edge_color = ch_edge_color.append(pd.DataFrame({
                "channel": ch,
                "r": 0,
                "g": 0,
                "b": 0
            }, index=[0]))

        elif vis_resp:
            ch_coords_new = ch_coords_new.append(pd.DataFrame({
                "channel": ch,
                "x": ch_coords.loc[ch_coords["name"] == ch, "x"].item(),
                "y": ch_coords.loc[ch_coords["name"] == ch, "y"].item(),
                "z": ch_coords.loc[ch_coords["name"] == ch, "z"].item(),
                "radius": 2
            }, index=[0]))
            ch_colors = ch_colors.append(pd.DataFrame({
                "channel": ch,
                "r": 0.8,
                "g": 0.8,
                "b": 0.8
            }, index=[0]))
            ch_rois_new = ch_rois_new.append(pd.DataFrame({
                "channel": ch,
                "roi": ch_rois.loc[ch_rois["channel"] == ch, "region"].item()
            }, index=[0]))
            # If this channel is also category selective, adding a ring to it:
            if cate == cate:
                if cate == "face":
                    c = [1, 0, 0]
                elif cate == "object":
                    c = [0, 1, 0]
                elif cate == "letter":
                    c = [0, 0, 1]
                elif cate == "false":
                    c = [1, 1, 0]
                ch_edge_color = ch_edge_color.append(pd.DataFrame({
                    "channel": ch,
                    "r": c[0],
                    "g": c[1],
                    "b": c[2]
                }, index=[0]))
            else:
                ch_edge_color = ch_edge_color.append(pd.DataFrame({
                    "channel": ch,
                    "r": 0,
                    "g": 0,
                    "b": 0
                }, index=[0]))

    ch_coords_new = ch_coords_new.reset_index(drop=True)
    ch_colors = ch_colors.reset_index(drop=True)
    ch_rois_new = ch_rois_new.reset_index(drop=True)
    ch_edge_color = ch_edge_color.reset_index(drop=True)
    # Save to csv:
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    ch_coords_new.to_csv(Path(save_root, "coords.csv"))
    ch_colors.to_csv(Path(save_root, "coords_colors.csv"))
    ch_rois_new.to_csv(Path(save_root, "coords_rois.csv"))
    ch_edge_color.to_csv(Path(save_root, "coords_edge_colors.csv"))
    # Save the ROIs as well:
    rois_colors = pd.DataFrame({roi.replace("ctx_rh_", "").replace("ctx_lh_", ""): param["colors"]["gnw"]
                                for roi in rois["gnw"]}, index=['r', 'g', 'b']).T
    rois_colors['roi'] = rois_colors.index
    rois_colors = rois_colors.reset_index(drop=True)
    rois_colors.to_csv(Path(save_root, "rois_dict.csv"))


if __name__ == "__main__":
    vis_resp = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/visual_responsiveness/sub-super/ses-V1/ieeg/results/high_gamma_wilcoxon_onset_two_tailed/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_analysis-vis_resp_all_results.csv"
    category_selectivity = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/category_selectivity/sub-super/ses-V1/ieeg/results/high_gamma_dprime_test/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv"
    lmm = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/high_gamma_gnw_ti/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"
    gnw_channels(vis_resp, category_selectivity, lmm,
                 save_root="/hpc/users/alexander.lepauvre/plotting_test/activation_analysis/gnw_brain")
