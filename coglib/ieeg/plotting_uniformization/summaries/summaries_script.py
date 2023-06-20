import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import config
import theories_rois
from pathlib import Path
import matplotlib.pyplot as plt
from general_utilities import load_fsaverage_coord
from matplotlib import font_manager
import os

selectivity_colors = {
    "face": [1, 1, 0],
    "object": [1, 0, 0],
    "letter": [0.3, 0.3, 0.3],
    "false": [0.93, 0.51, 0.93]
}

depth_roi = ["WM-hypointensities", "Thalamus-Proper", "VentralDC", "Unknown", "Putamen", "Lateral-Ventricle",
             "Inf-Lat-Vent", "Hippocampus", "ctx-rh-unknown", "ctx-lh-unknown", "choroid-plexus",
             "Cerebral-White-Matter", "Cerebellum-Cortex", "CC_Mid_Anterior", "Amygdala"]

rois = {
    "Occ": ["ctx-lh-lateraloccipital",
            "ctx-lh-cuneus",
            "ctx-lh-pericalcarine",
            "ctx-rh-lateraloccipital",
            "ctx-rh-cuneus",
            "ctx-rh-pericalcarine"],
    "Par": ["ctx-lh-isthmuscingulate",
            "ctx-lh-precuneus",
            "ctx-lh-inferiorparietal",
            "ctx-lh-superiorparietal",
            "ctx-lh-supramarginal",
            "ctx-rh-isthmuscingulate",
            "ctx-rh-precuneus",
            "ctx-rh-inferiorparietal",
            "ctx-rh-superiorparietal",
            "ctx-rh-supramarginal"],
    "VT": ["ctx-lh-inferiortemporal",
           "ctx-lh-lingual",
           "ctx-lh-fusiform",
           "ctx-lh-parahippocampal",
           "ctx-lh-entorhinal",
           "ctx-rh-inferiortemporal",
           "ctx-rh-lingual",
           "ctx-rh-fusiform",
           "ctx-rh-parahippocampal",
           "ctx-rh-entorhinal"],
    "LT": ["ctx-lh-middletemporal",
           "ctx-lh-bankssts",
           "ctx-lh-transversetemporal",
           "ctx-lh-superiortemporal",
           "ctx-lh-temporalpole",
           "ctx-rh-middletemporal",
           "ctx-rh-bankssts",
           "ctx-rh-transversetemporal",
           "ctx-rh-superiortemporal",
           "ctx-rh-temporalpole"],
    "PFC": ["ctx-lh-caudalmiddlefrontal",
            "ctx-lh-superiorfrontal",
            "ctx-lh-parsopercularis",
            "ctx-lh-rostralmiddlefrontal",
            "ctx-lh-parstriangularis",
            "ctx-lh-parsorbitalis",
            "ctx-lh-lateralorbitofrontal",
            "ctx-lh-medialorbitofrontal",
            "ctx-lh-orbitofrontal",
            "ctx-lh-frontalpole",
            "ctx-lh-medialorbitofrontal",
            "ctx-lh-rostralanteriorcingulate",
            "ctx-lh-caudalanteriorcingulate",
            "ctx-rh-caudalmiddlefrontal",
            "ctx-rh-superiorfrontal",
            "ctx-rh-parsopercularis",
            "ctx-rh-rostralmiddlefrontal",
            "ctx-rh-parstriangularis",
            "ctx-rh-parsorbitalis",
            "ctx-rh-lateralorbitofrontal",
            "ctx-rh-medialorbitofrontal",
            "ctx-rh-orbitofrontal",
            "ctx-rh-frontalpole",
            "ctx-rh-medialorbitofrontal",
            "ctx-rh-rostralanteriorcingulate",
            "ctx-rh-caudalanteriorcingulate"],
    "SM": [
        "ctx-rh-precentral",
        "ctx-rh-postcentral",
        "ctx-rh-paracentral",
        "ctx-lh-precentral",
        "ctx-lh-postcentral",
        "ctx-lh-paracentral"
    ]
}
roi_colors = {
    "Occ": [0.6313725490196078, 0.788235294117647, 0.9568627450980393],
    "Par": [1.0, 0.7058823529411765, 0.5098039215686274],
    "VT": [0.5529411764705883, 0.8980392156862745, 0.6313725490196078],
    "LT": [0.8156862745098039, 0.7333333333333333, 1.0],
    "PFC": [255 / 255, 233 / 255, 0],
    "SM": [0.8705882352941177, 0.7333333333333333, 0.6078431372549019]
}
model_colors = {
    "time_win_dur_iit": [0.52, 0.86, 1],
    "time_win_dur_gnw": [
        0,
        1,
        0
    ],
    "time_win_dur_cate_iit": [
        0,
        0,
        1
    ],
    "time_win_dur_cate_gnw": [0.42, 1, 0.86],
}
param = config.param
fig_size = param["figure_size_mm"]
def_cmap = param["colors"]["cmap"]
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": "none"
                 }
mpl.rcParams.update(new_rc_params)
# Set Helvetica as the default font:
font_path = os.path.join("/hpc/users/alexander.lepauvre/sw/github/plotting_uniformization", "Helvetica.ttf")
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()
plt.rc('font', size=param["font_size"])  # controls default text sizes
plt.rc('axes', titlesize=param["font_size"])  # fontsize of the axes title
plt.rc('axes', labelsize=param["font_size"])  # fontsize of the x and y labels
plt.rc('xtick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('ytick', labelsize=param["font_size"])  # fontsize of the tick labels
plt.rc('legend', fontsize=param["font_size"])  # legend fontsize
plt.rc('figure', titlesize=param["font_size"])  # fontsize of the fi


# Set Helvetica as the default font:


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# Set the color bars for each category:
activation_cmap = plt.get_cmap("RdYlBu_r")


def get_rois(bids_root, subject_list, channels_list, atlas="",
             folder="derivatives/preprocessing/sub-{}/ses-V1/ieeg/atlas_mapping/raw/desbadcharej_notfil_lapref"
                    "/sub-{}_ses-V1_task-Dur_desc-elecmapping_{}_ieeg.csv", ignore_hemi=True):
    channels_rois = []
    for sub in subject_list:
        # Get the channels of this subject:
        sub_ch = [ch.split("-")[1] for ch in channels_list if ch.split("-")[0] == sub]
        # Generate the path to the atlas mapping for this subject:
        try:
            atlas_map = pd.read_csv(Path(bids_root, folder.format(sub, sub, atlas)))
            atlas_map = atlas_map.loc[atlas_map["channel"].isin(sub_ch)]
        except FileNotFoundError:
            print("Warning: for subject {}, atlas file {} not found".format(sub, atlas))
            atlas_map = pd.DataFrame({
                "channel": sub_ch,
                "region": "Unknown"
            })
        if ignore_hemi:
            atlas_map["region"] = [lbl.replace("ctx_lh_", "").replace("ctx_rh_",
                                                                      "").replace("Right-", "").replace("Left-",
                                                                                                        "") if
                                   isinstance(lbl, str) else lbl
                                   for lbl in atlas_map["region"].to_list()]
        # Reformat the table output to remove the unknown and irrelevant labels:
        atlas_map_clean = pd.DataFrame()
        for ch in atlas_map["channel"].to_list():
            try:
                regions = atlas_map.loc[atlas_map["channel"] == ch, "region"].item().split("/")
            except AttributeError:
                print("No labels for ch-" + sub + "-" + ch)
                continue
            if len(regions) == 1:
                atlas_map_clean = atlas_map_clean.append(pd.DataFrame({
                    "channel": "-".join([sub, ch]),
                    "region": regions[0]
                }, index=[0]), ignore_index=True)
            else:
                reg = [region for region in regions if region != "Unknown" and "White-Matter" not in region]
                if len(reg) == 0:
                    atlas_map_clean = atlas_map_clean.append(pd.DataFrame({
                        "channel": "-".join([sub, ch]),
                        "region": regions[0]
                    }, index=[0]), ignore_index=True)
                else:
                    atlas_map_clean = atlas_map_clean.append(pd.DataFrame({
                        "channel": "-".join([sub, ch]),
                        "region": reg[0]
                    }, index=[0]), ignore_index=True)

        channels_rois.append(atlas_map_clean)

    return pd.concat(channels_rois, ignore_index=True)


# Set parameters:
categories = ["face", "object", "letter", "false"]
models = ["time_win_dur_iit", "time_win_dur_gnw", "time_win_dur_cate_iit", "time_win_dur_cate_gnw"]

# Set file names:
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
visual_responsiveness_file = "derivatives/visual_responsiveness/sub-super/ses-V1/ieeg/results/" \
                             "high_gamma_wilcoxon_onset_two_tailed/desbadcharej_notfil_lapref/" \
                             "sub-super_ses-V1_task-Dur_analysis-vis_resp_all_results.csv"
bayes_visual_responsiveness_file = "derivatives/visual_responsiveness/sub-super/ses-V1/ieeg/results/" \
                             "high_gamma_wilcoxon_onset_bayes_t_test/desbadcharej_notfil_lapref/" \
                             "sub-super_ses-V1_task-Dur_analysis-vis_resp_all_results.csv"
category_selectivity_ti_file = "derivatives/category_selectivity/sub-super/ses-V1/ieeg/results/" \
                               "high_gamma_dprime_test_ti/desbadcharej_notfil_lapref/" \
                               "sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv"
category_selectivity_tr_file = "derivatives/category_selectivity/sub-super/ses-V1/ieeg/results/" \
                               "high_gamma_dprime_test_tr/desbadcharej_notfil_lapref/" \
                               "sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv"
activation_analysis_iit_ti_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                  "high_gamma_iit_ti/desbadcharej_notfil_lapref/" \
                                  "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_iit_tr_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                  "high_gamma_iit_tr/desbadcharej_notfil_lapref/" \
                                  "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_gnw_ti_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                  "high_gamma_gnw_ti/desbadcharej_notfil_lapref/" \
                                  "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"
activation_analysis_gnw_tr_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                  "high_gamma_gnw_tr" \
                                  "/desbadcharej_notfil_lapref/" \
                                  "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"

activation_analysis_iit_ti_alpha_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                        "alpha_iit_ti/desbadcharej_notfil_lapref/" \
                                        "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_iit_tr_alpha_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                        "alpha_iit_tr/desbadcharej_notfil_lapref/" \
                                        "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_gnw_ti_alpha_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                        "alpha_gnw_ti/desbadcharej_notfil_lapref/" \
                                        "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"
activation_analysis_gnw_tr_alpha_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                        "alpha_gnw_tr" \
                                        "/desbadcharej_notfil_lapref/" \
                                        "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"

activation_analysis_iit_ti_erp_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                      "erp_iit_ti/desbadcharej_notfil_lapref/" \
                                      "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_iit_tr_erp_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                      "erp_iit_tr/desbadcharej_notfil_lapref/" \
                                      "sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_best_lmm_results.csv"
activation_analysis_gnw_ti_erp_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                      "erp_gnw_ti/desbadcharej_notfil_lapref/" \
                                      "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"
activation_analysis_gnw_tr_erp_file = "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                      "erp_gnw_tr" \
                                      "/desbadcharej_notfil_lapref/" \
                                      "sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_best_lmm_results.csv"

duration_decoding_face_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_face_ti_500ms/desbadcharej_notfil_lapref/sub" \
                                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_object_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_object_ti_500ms/desbadcharej_notfil_lapref/sub" \
                                   "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_letter_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_letter_ti_500ms/desbadcharej_notfil_lapref/sub" \
                                   "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_false_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                  "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_false_ti_500ms/desbadcharej_notfil_lapref/sub" \
                                  "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_face_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                 "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_face_tr_500ms/desbadcharej_notfil_lapref/sub" \
                                 "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_object_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_object_tr_500ms/desbadcharej_notfil_lapref/sub" \
                                   "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_letter_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_letter_tr_500ms/desbadcharej_notfil_lapref/sub" \
                                   "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"
duration_decoding_false_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                  "-V1/ieeg/results/duration_decoding_allbrain_high_gamma_false_tr_500ms/desbadcharej_notfil_lapref/sub" \
                                  "-super_ses-V1_task-Dur_ana-activation_analysis_allbrain_duration_decoding_accuracy_stats.csv"

duration_tracking_face_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                 "-V1/ieeg/results/duration_tracking_high_gamma_face_ti/desbadcharej_notfil_lapref" \
                                 "/sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_object_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_tracking_high_gamma_object_ti" \
                                   "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                                   "-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_letter_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_tracking_high_gamma_letter_ti" \
                                   "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                                   "-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_false_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                  "-V1/ieeg/results/duration_tracking_high_gamma_symbol_ti/desbadcharej_notfil_lapref" \
                                  "/sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_tracking_proportion_stats" \
                                  ".csv"
duration_tracking_face_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                 "-V1/ieeg/results/duration_tracking_high_gamma_face_tr/desbadcharej_notfil_lapref" \
                                 "/sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_object_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_tracking_high_gamma_object_tr" \
                                   "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                                   "-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_letter_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                   "-V1/ieeg/results/duration_tracking_high_gamma_letter_tr" \
                                   "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                                   "-activation_analysis_iit_tracking_proportion_stats.csv"
duration_tracking_false_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                                  "-V1/ieeg/results/duration_tracking_high_gamma_symbol_tr/desbadcharej_notfil_lapref" \
                                  "/sub-super_ses-V1_task-Dur_ana-activation_analysis_iit_tracking_proportion_stats" \
                                  ".csv"

onset_offset_face_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                            "-V1/ieeg/results/onset_offset_high_gamma_gnw_face_ti/desbadcharej_notfil_lapref" \
                            "/sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_object_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                              "-V1/ieeg/results/onset_offset_high_gamma_gnw_object_ti" \
                              "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                              "-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_letter_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                              "-V1/ieeg/results/onset_offset_high_gamma_gnw_letter_ti" \
                              "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                              "-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_false_ti_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                             "-V1/ieeg/results/onset_offset_high_gamma_gnw_false_ti/desbadcharej_notfil_lapref" \
                             "/sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_face_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                            "-V1/ieeg/results/onset_offset_high_gamma_gnw_face_tr/desbadcharej_notfil_lapref" \
                            "/sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_object_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                              "-V1/ieeg/results/onset_offset_high_gamma_gnw_object_tr" \
                              "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                              "-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_letter_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                              "-V1/ieeg/results/onset_offset_high_gamma_gnw_letter_tr" \
                              "/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_ana" \
                              "-activation_analysis_gnw_onset_offset_results.csv"
onset_offset_false_tr_file = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/activation_analysis/sub-super/ses" \
                             "-V1/ieeg/results/onset_offset_high_gamma_gnw_false_tr/desbadcharej_notfil_lapref" \
                             "/sub-super_ses-V1_task-Dur_ana-activation_analysis_gnw_onset_offset_results.csv"

# Load each:
vis_resp = pd.read_csv(Path(bids_root, visual_responsiveness_file))
bayes_vis_resp = pd.read_csv(Path(bids_root, bayes_visual_responsiveness_file))
cate_sel_ti = pd.read_csv(Path(bids_root, category_selectivity_ti_file))
cate_sel_tr = pd.read_csv(Path(bids_root, category_selectivity_tr_file))

activation_analysis_iit_ti = pd.read_csv(Path(bids_root, activation_analysis_iit_ti_file))
activation_analysis_iit_tr = pd.read_csv(Path(bids_root, activation_analysis_iit_tr_file))
activation_analysis_gnw_ti = pd.read_csv(Path(bids_root, activation_analysis_gnw_ti_file))
activation_analysis_gnw_tr = pd.read_csv(Path(bids_root, activation_analysis_gnw_tr_file))

activation_analysis_alpha_iit_ti = pd.read_csv(Path(bids_root, activation_analysis_iit_ti_alpha_file))
activation_analysis_alpha_iit_tr = pd.read_csv(Path(bids_root, activation_analysis_iit_tr_alpha_file))
activation_analysis_alpha_gnw_ti = pd.read_csv(Path(bids_root, activation_analysis_gnw_ti_alpha_file))
activation_analysis_alpha_gnw_tr = pd.read_csv(Path(bids_root, activation_analysis_gnw_tr_alpha_file))

activation_analysis_erp_iit_ti = pd.read_csv(Path(bids_root, activation_analysis_iit_ti_erp_file))
activation_analysis_erp_iit_tr = pd.read_csv(Path(bids_root, activation_analysis_iit_tr_erp_file))
activation_analysis_erp_gnw_ti = pd.read_csv(Path(bids_root, activation_analysis_gnw_ti_erp_file))
activation_analysis_erp_gnw_tr = pd.read_csv(Path(bids_root, activation_analysis_gnw_tr_erp_file))

duration_decoding_face_ti_results = pd.read_csv(duration_decoding_face_ti_file)
duration_decoding_object_ti_results = pd.read_csv(duration_decoding_object_ti_file)
duration_decoding_letter_ti_results = pd.read_csv(duration_decoding_letter_ti_file)
duration_decoding_false_ti_results = pd.read_csv(duration_decoding_false_ti_file)
duration_decoding_face_tr_results = pd.read_csv(duration_decoding_face_tr_file)
duration_decoding_object_tr_results = pd.read_csv(duration_decoding_object_tr_file)
duration_decoding_letter_tr_results = pd.read_csv(duration_decoding_letter_tr_file)
duration_decoding_false_tr_results = pd.read_csv(duration_decoding_false_tr_file)

duration_tracking_face_ti_results = pd.read_csv(duration_tracking_face_ti_file)
duration_tracking_object_ti_results = pd.read_csv(duration_tracking_object_ti_file)
duration_tracking_letter_ti_results = pd.read_csv(duration_tracking_letter_ti_file)
duration_tracking_false_ti_results = pd.read_csv(duration_tracking_false_ti_file)
duration_tracking_face_tr_results = pd.read_csv(duration_tracking_face_tr_file)
duration_tracking_object_tr_results = pd.read_csv(duration_tracking_object_tr_file)
duration_tracking_letter_tr_results = pd.read_csv(duration_tracking_letter_tr_file)
duration_tracking_false_tr_results = pd.read_csv(duration_tracking_false_tr_file)

# Onset offset analysis results:
onset_offset_face_ti_results = pd.read_csv(onset_offset_face_ti_file)
onset_offset_object_ti_results = pd.read_csv(onset_offset_object_ti_file)
onset_offset_letter_ti_results = pd.read_csv(onset_offset_letter_ti_file)
onset_offset_false_ti_results = pd.read_csv(onset_offset_false_ti_file)
onset_offset_face_tr_results = pd.read_csv(onset_offset_face_tr_file)
onset_offset_object_tr_results = pd.read_csv(onset_offset_object_tr_file)
onset_offset_letter_tr_results = pd.read_csv(onset_offset_letter_tr_file)
onset_offset_false_tr_results = pd.read_csv(onset_offset_false_tr_file)

# ====================================================================================================
# Get the name of each channel and their loc and ROIs:
channels_list = vis_resp["channel"].to_list()
# Extract the list of subjects:
subjects_list = list(set([ch.split("-")[0] for ch in channels_list]))
# Get both the destrieux and wang labels:
channels_rois_desikan = get_rois(bids_root, subjects_list, channels_list, atlas="aparc+aseg")
channels_rois_destrieux = get_rois(bids_root, subjects_list, channels_list, atlas="aparc.a2009s+aseg")
channels_rois_wang = get_rois(bids_root, subjects_list, channels_list, atlas="wang15_mplbl")
# Finally, load the channels MNI coordinates:
ch_coords = load_fsaverage_coord(bids_root, subjects_list, ses='V1', laplace_reloc=True)
ch_coords = ch_coords.loc[ch_coords["name"].isin(channels_list)]
ch_coords = ch_coords.rename(columns={"name": "channel"})

# ====================================================================================================
# Loop through each channel to extract all the relevant info:
channels_summary_table = pd.DataFrame()
for ch in channels_list:
    if ch == "SE107-O2PH5":
        print("!")
    # Get all the info from the different file:
    # Onset responsiveness:
    is_responsive = vis_resp.loc[vis_resp["channel"] == ch, "reject"].item()
    is_responsive_bayes = bayes_vis_resp.loc[bayes_vis_resp["channel"] == ch, "reject"].item()
    ti_onset_strength = vis_resp.loc[vis_resp["channel"] == ch,
                                     "effect_strength-stimulus onset/Irrelevant"].item()
    tr_onset_strength = vis_resp.loc[vis_resp["channel"] == ch,
                                     "effect_strength-stimulus onset/Relevant non-target"].item()
    if ti_onset_strength > 0 and tr_onset_strength > 0 and is_responsive:
        onset_type = "activated"
    elif ti_onset_strength < 0 and tr_onset_strength < 0 and is_responsive:
        onset_type = "deactivated"
    else:
        onset_type = None
        is_responsive = False
        ti_onset_strength = np.nan
        tr_onset_strength = np.nan
    if is_responsive:
        ti_latency = vis_resp.loc[vis_resp["channel"] == ch,
                                  "latency-stimulus onset/Irrelevant"].item()
        tr_latency = vis_resp.loc[vis_resp["channel"] == ch,
                                  "latency-stimulus onset/Relevant non-target"].item()
    else:
        ti_latency = np.nan
        tr_latency = np.nan

    # Category selectivity:
    selectivity_ti = cate_sel_ti.loc[cate_sel_ti["channel"] == ch, "condition"].item()
    selectivity_tr = cate_sel_tr.loc[cate_sel_tr["channel"] == ch, "condition"].item()
    dprime_ti = cate_sel_ti.loc[cate_sel_ti["channel"] == ch, "effect_strength"].item()
    dprime_tr = cate_sel_tr.loc[cate_sel_tr["channel"] == ch, "effect_strength"].item()
    if (selectivity_ti == selectivity_tr) and selectivity_ti is not None:
        selectivity = selectivity_ti
    else:
        selectivity = None

    # Activation analysis high gamma:
    if ch in activation_analysis_iit_ti["group"].to_list():
        theory_roi = "iit"
        ch_model_ti = list(activation_analysis_iit_ti.loc[activation_analysis_iit_ti["group"] == ch, "model"].unique())[
            0]
        if ch_model_ti not in models:
            ch_model_ti = "Theory agnostic"
        ch_model_tr = list(activation_analysis_iit_tr.loc[activation_analysis_iit_tr["group"] == ch, "model"].unique())[
            0]
        if ch_model_tr not in models:
            ch_model_tr = "Theory agnostic"
    elif ch in activation_analysis_gnw_ti["group"].to_list():
        theory_roi = "gnw"
        ch_model_ti = list(activation_analysis_gnw_ti.loc[activation_analysis_gnw_ti["group"] == ch, "model"].unique())[
            0]
        if ch_model_ti not in models:
            ch_model_ti = "Theory agnostic"
        ch_model_tr = list(activation_analysis_gnw_tr.loc[activation_analysis_gnw_tr["group"] == ch, "model"].unique())[
            0]
        if ch_model_tr not in models:
            ch_model_tr = "Theory agnostic"
    else:
        theory_roi = None
        ch_model_ti = None
        ch_model_tr = None

    # Activation analysis alpha
    if ch in activation_analysis_alpha_iit_ti["group"].to_list():
        ch_model_ti_alpha = list(activation_analysis_alpha_iit_ti.loc[activation_analysis_alpha_iit_ti["group"] == ch,
                                                                      "model"].unique())[0]
        if ch_model_ti_alpha not in models:
            ch_model_ti_alpha = "Theory agnostic"
        ch_model_tr_alpha = list(activation_analysis_alpha_iit_tr.loc[activation_analysis_alpha_iit_tr["group"] == ch,
                                                                      "model"].unique())[0]
        if ch_model_tr_alpha not in models:
            ch_model_tr_alpha = "Theory agnostic"
    elif ch in activation_analysis_alpha_gnw_ti["group"].to_list():
        ch_model_ti_alpha = \
            list(activation_analysis_alpha_gnw_ti.loc[
                     activation_analysis_alpha_gnw_ti["group"] == ch, "model"].unique())[
                0]
        if ch_model_ti_alpha not in models:
            ch_model_ti_alpha = "Theory agnostic"
        ch_model_tr_alpha = list(activation_analysis_alpha_gnw_tr.loc[activation_analysis_alpha_gnw_tr["group"] == ch,
                                                                      "model"].unique())[0]
        if ch_model_tr_alpha not in models:
            ch_model_tr_alpha = "Theory agnostic"
    else:
        ch_model_ti_alpha = None
        ch_model_tr_alpha = None

    # Activation analysis ERP:
    if ch in activation_analysis_erp_iit_ti["group"].to_list():
        ch_model_ti_erp = list(activation_analysis_erp_iit_ti.loc[activation_analysis_erp_iit_ti["group"] == ch,
                                                                  "model"].unique())[0]
        if ch_model_ti_erp not in models:
            ch_model_ti_erp = "Theory agnostic"
        ch_model_tr_erp = list(activation_analysis_erp_iit_tr.loc[activation_analysis_erp_iit_tr["group"] == ch,
                                                                  "model"].unique())[0]
        if ch_model_tr_erp not in models:
            ch_model_tr_erp = "Theory agnostic"
    elif ch in activation_analysis_erp_gnw_ti["group"].to_list():
        theory_roi = "gnw"
        ch_model_ti_erp = \
            list(activation_analysis_erp_gnw_ti.loc[activation_analysis_erp_gnw_ti["group"] == ch, "model"].unique())[
                0]
        if ch_model_ti_erp not in models:
            ch_model_ti_erp = "Theory agnostic"
        ch_model_tr_erp = list(activation_analysis_erp_gnw_tr.loc[activation_analysis_erp_gnw_tr["group"] == ch,
                                                                  "model"].unique())[0]
        if ch_model_tr_erp not in models:
            ch_model_tr_erp = "Theory agnostic"
    else:
        ch_model_ti_erp = None
        ch_model_tr_erp = None

    # Duration decoding:
    # Faces:
    if duration_decoding_face_ti_results.loc[
        duration_decoding_face_ti_results["channel"] == ch, "p-value"].item() < 0.05:
        face_duration_decoding_accuracy_ti = duration_decoding_face_ti_results.loc[
            duration_decoding_face_ti_results["channel"] == ch, "decoding_score"].item()
    else:
        face_duration_decoding_accuracy_ti = np.nan
        face_duration_decoding_accuracy_tr = np.nan
    if duration_decoding_face_ti_results.loc[
        duration_decoding_face_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
            duration_decoding_face_tr_results.loc[
                duration_decoding_face_tr_results["channel"] == ch, "p-value"].item() < 0.05:
        face_duration_decoding_accuracy_ti = duration_decoding_face_ti_results.loc[
            duration_decoding_face_ti_results["channel"] == ch, "decoding_score"].item()
        face_duration_decoding_accuracy_tr = duration_decoding_face_tr_results.loc[
            duration_decoding_face_tr_results["channel"] == ch, "decoding_score"].item()
    else:
        face_duration_decoding_accuracy_ti = np.nan
        face_duration_decoding_accuracy_tr = np.nan
    # Objects:
    if duration_decoding_object_ti_results.loc[
        duration_decoding_object_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
            duration_decoding_object_tr_results.loc[
                duration_decoding_object_tr_results["channel"] == ch, "p-value"].item() < 0.05:
        object_duration_decoding_accuracy_ti = duration_decoding_object_ti_results.loc[
            duration_decoding_object_ti_results["channel"] == ch, "decoding_score"].item()
        object_duration_decoding_accuracy_tr = duration_decoding_object_tr_results.loc[
            duration_decoding_object_tr_results["channel"] == ch, "decoding_score"].item()
    else:
        object_duration_decoding_accuracy_ti = np.nan
        object_duration_decoding_accuracy_tr = np.nan
    # Letter:
    if duration_decoding_letter_ti_results.loc[
        duration_decoding_letter_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
            duration_decoding_letter_tr_results.loc[
                duration_decoding_letter_tr_results["channel"] == ch, "p-value"].item() < 0.05:
        letter_duration_decoding_accuracy_ti = duration_decoding_letter_ti_results.loc[
            duration_decoding_letter_ti_results["channel"] == ch, "decoding_score"].item()
        letter_duration_decoding_accuracy_tr = duration_decoding_letter_tr_results.loc[
            duration_decoding_letter_tr_results["channel"] == ch, "decoding_score"].item()
    else:
        letter_duration_decoding_accuracy_ti = np.nan
        letter_duration_decoding_accuracy_tr = np.nan
    # False:
    if duration_decoding_false_ti_results.loc[
        duration_decoding_false_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
            duration_decoding_false_tr_results.loc[
                duration_decoding_false_tr_results["channel"] == ch, "p-value"].item() < 0.05:
        false_duration_decoding_accuracy_ti = duration_decoding_false_ti_results.loc[
            duration_decoding_false_ti_results["channel"] == ch, "decoding_score"].item()
        false_duration_decoding_accuracy_tr = duration_decoding_false_tr_results.loc[
            duration_decoding_false_tr_results["channel"] == ch, "decoding_score"].item()
    else:
        false_duration_decoding_accuracy_ti = np.nan
        false_duration_decoding_accuracy_tr = np.nan
    duration_decoding = not np.isnan([face_duration_decoding_accuracy_ti,
                                      object_duration_decoding_accuracy_ti,
                                      letter_duration_decoding_accuracy_ti,
                                      false_duration_decoding_accuracy_ti]).all()
    # Check without conjunction:
    # TI:
    decoding_ti_uncorr = \
        np.any([duration_decoding_face_ti_results.loc[duration_decoding_face_ti_results["channel"] == ch,
                                                      "p-value"].item() < 0.05,
                duration_decoding_object_ti_results.loc[duration_decoding_object_ti_results["channel"] == ch,
                                                        "p-value"].item() < 0.05,
                duration_decoding_letter_ti_results.loc[duration_decoding_letter_ti_results["channel"] == ch,
                                                        "p-value"].item() < 0.05,
                duration_decoding_false_ti_results.loc[duration_decoding_false_ti_results["channel"] == ch,
                                                       "p-value"].item() < 0.05
                ])
    # TR:
    decoding_tr_uncorr = \
        np.any([duration_decoding_face_tr_results.loc[duration_decoding_face_tr_results["channel"] == ch,
                                                      "p-value"].item() < 0.05,
                duration_decoding_object_tr_results.loc[duration_decoding_object_tr_results["channel"] == ch,
                                                        "p-value"].item() < 0.05,
                duration_decoding_letter_tr_results.loc[duration_decoding_letter_tr_results["channel"] == ch,
                                                        "p-value"].item() < 0.05,
                duration_decoding_false_tr_results.loc[duration_decoding_false_tr_results["channel"] == ch,
                                                       "p-value"].item() < 0.05
                ])

    # Duration tracking:
    if ch in duration_tracking_face_ti_results["channel"].to_list():
        # Faces:
        if duration_tracking_face_ti_results.loc[
            duration_tracking_face_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
                duration_tracking_face_tr_results.loc[
                    duration_tracking_face_tr_results["channel"] == ch, "p-value"].item() < 0.05:
            face_duration_tracking_accuracy_ti = duration_tracking_face_ti_results.loc[
                duration_tracking_face_ti_results["channel"] == ch, "tracking_accuracy"].item()
            face_duration_tracking_accuracy_tr = duration_tracking_face_tr_results.loc[
                duration_tracking_face_tr_results["channel"] == ch, "tracking_accuracy"].item()
        else:
            face_duration_tracking_accuracy_ti = np.nan
            face_duration_tracking_accuracy_tr = np.nan
        # Objects:
        if duration_tracking_object_ti_results.loc[
            duration_tracking_object_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
                duration_tracking_object_tr_results.loc[
                    duration_tracking_object_tr_results["channel"] == ch, "p-value"].item() < 0.05:
            object_duration_tracking_accuracy_ti = duration_tracking_object_ti_results.loc[
                duration_tracking_object_ti_results["channel"] == ch, "tracking_accuracy"].item()
            object_duration_tracking_accuracy_tr = duration_tracking_object_tr_results.loc[
                duration_tracking_object_tr_results["channel"] == ch, "tracking_accuracy"].item()
        else:
            object_duration_tracking_accuracy_ti = np.nan
            object_duration_tracking_accuracy_tr = np.nan
        # Letter:
        if duration_tracking_letter_ti_results.loc[
            duration_tracking_letter_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
                duration_tracking_letter_tr_results.loc[
                    duration_tracking_letter_tr_results["channel"] == ch, "p-value"].item() < 0.05:
            letter_duration_tracking_accuracy_ti = duration_tracking_letter_ti_results.loc[
                duration_tracking_letter_ti_results["channel"] == ch, "tracking_accuracy"].item()
            letter_duration_tracking_accuracy_tr = duration_tracking_letter_tr_results.loc[
                duration_tracking_letter_tr_results["channel"] == ch, "tracking_accuracy"].item()
        else:
            letter_duration_tracking_accuracy_ti = np.nan
            letter_duration_tracking_accuracy_tr = np.nan
        # False:
        if duration_tracking_false_ti_results.loc[
            duration_tracking_false_ti_results["channel"] == ch, "p-value"].item() < 0.05 and \
                duration_tracking_false_tr_results.loc[
                    duration_tracking_false_tr_results["channel"] == ch, "p-value"].item() < 0.05:
            false_duration_tracking_accuracy_ti = duration_tracking_false_ti_results.loc[
                duration_tracking_false_ti_results["channel"] == ch, "tracking_accuracy"].item()
            false_duration_tracking_accuracy_tr = duration_tracking_false_tr_results.loc[
                duration_tracking_false_tr_results["channel"] == ch, "tracking_accuracy"].item()
        else:
            false_duration_tracking_accuracy_ti = np.nan
            false_duration_tracking_accuracy_tr = np.nan
        duration_tracking = not np.isnan([face_duration_tracking_accuracy_ti,
                                          object_duration_tracking_accuracy_ti,
                                          letter_duration_tracking_accuracy_ti,
                                          false_duration_tracking_accuracy_ti]).all()

        # Check without conjunction:
        # TI:
        tracking_ti_uncorr = \
            np.any([duration_tracking_face_ti_results.loc[duration_tracking_face_ti_results["channel"] == ch,
                                                          "p-value"].item() < 0.05,
                    duration_tracking_object_ti_results.loc[duration_tracking_object_ti_results["channel"] == ch,
                                                            "p-value"].item() < 0.05,
                    duration_tracking_letter_ti_results.loc[duration_tracking_letter_ti_results["channel"] == ch,
                                                            "p-value"].item() < 0.05,
                    duration_tracking_false_ti_results.loc[duration_tracking_false_ti_results["channel"] == ch,
                                                           "p-value"].item() < 0.05
                    ])
        # TR:
        tracking_tr_uncorr = \
            np.any([duration_tracking_face_tr_results.loc[duration_tracking_face_tr_results["channel"] == ch,
                                                          "p-value"].item() < 0.05,
                    duration_tracking_object_tr_results.loc[duration_tracking_object_tr_results["channel"] == ch,
                                                            "p-value"].item() < 0.05,
                    duration_tracking_letter_tr_results.loc[duration_tracking_letter_tr_results["channel"] == ch,
                                                            "p-value"].item() < 0.05,
                    duration_tracking_false_tr_results.loc[duration_tracking_false_tr_results["channel"] == ch,
                                                           "p-value"].item() < 0.05
                    ])
    else:
        face_duration_tracking_accuracy_ti = np.nan
        face_duration_tracking_accuracy_tr = np.nan
        object_duration_tracking_accuracy_ti = np.nan
        object_duration_tracking_accuracy_tr = np.nan
        letter_duration_tracking_accuracy_ti = np.nan
        letter_duration_tracking_accuracy_tr = np.nan
        false_duration_tracking_accuracy_ti = np.nan
        false_duration_tracking_accuracy_tr = np.nan
        duration_tracking = None
        tracking_ti_uncorr = None
        tracking_tr_uncorr = None

    # Onset offset results:
    if ch in onset_offset_face_ti_results["channel"].to_list():
        # Extract the condition for each task and category:
        face_onset_offset_ti = onset_offset_face_ti_results.loc[onset_offset_face_ti_results["channel"] == ch,
                                                                "condition"].item()
        face_onset_offset_tr = onset_offset_face_tr_results.loc[onset_offset_face_tr_results["channel"] == ch,
                                                                "condition"].item()
        object_onset_offset_ti = onset_offset_object_ti_results.loc[onset_offset_object_ti_results["channel"] == ch,
                                                                    "condition"].item()
        object_onset_offset_tr = onset_offset_object_tr_results.loc[onset_offset_object_tr_results["channel"] == ch,
                                                                    "condition"].item()
        letter_onset_offset_ti = onset_offset_letter_ti_results.loc[onset_offset_letter_ti_results["channel"] == ch,
                                                                    "condition"].item()
        letter_onset_offset_tr = onset_offset_letter_tr_results.loc[onset_offset_letter_tr_results["channel"] == ch,
                                                                    "condition"].item()
        false_onset_offset_ti = onset_offset_false_ti_results.loc[onset_offset_false_ti_results["channel"] == ch,
                                                                  "condition"].item()
        false_onset_offset_tr = onset_offset_false_tr_results.loc[onset_offset_false_tr_results["channel"] == ch,
                                                                  "condition"].item()
        # Check:
        if face_onset_offset_ti == "both" and face_onset_offset_tr == "both":
            face_both = True
            face_onset = True
            face_offset = True
        elif 'stimulus onset' in face_onset_offset_ti and 'stimulus onset' in face_onset_offset_tr:
            face_both = False
            face_onset = True
            face_offset = False
        elif 'stimulus offset' in face_onset_offset_ti and 'stimulus offset' in face_onset_offset_tr:
            face_both = False
            face_onset = False
            face_offset = True
        else:
            face_both = False
            face_onset = False
            face_offset = False

        if object_onset_offset_ti == "both" and object_onset_offset_tr == "both":
            object_both = True
            object_onset = True
            object_offset = True
        elif 'stimulus onset' in object_onset_offset_ti and 'stimulus onset' in object_onset_offset_tr:
            object_both = False
            object_onset = True
            object_offset = False
        elif 'stimulus offset' in object_onset_offset_ti and 'stimulus offset' in object_onset_offset_tr:
            object_both = False
            object_onset = False
            object_offset = True
        else:
            object_both = False
            object_onset = False
            object_offset = False

        if letter_onset_offset_ti == "both" and letter_onset_offset_tr == "both":
            letter_both = True
            letter_onset = True
            letter_offset = True
        elif 'stimulus onset' in letter_onset_offset_ti and 'stimulus onset' in letter_onset_offset_tr:
            letter_both = False
            letter_onset = True
            letter_offset = False
        elif 'stimulus offset' in letter_onset_offset_ti and 'stimulus offset' in letter_onset_offset_tr:
            letter_both = False
            letter_onset = False
            letter_offset = True
        else:
            letter_both = False
            letter_onset = False
            letter_offset = False

        if false_onset_offset_ti == "both" and false_onset_offset_tr == "both":
            false_both = True
            false_onset = True
            false_offset = True
        elif 'stimulus onset' in false_onset_offset_ti and 'stimulus onset' in false_onset_offset_tr:
            false_both = False
            false_onset = True
            false_offset = False
        elif 'stimulus offset' in false_onset_offset_ti and 'stimulus offset' in false_onset_offset_tr:
            false_both = False
            false_onset = False
            false_offset = True
        else:
            false_both = False
            false_onset = False
            false_offset = False

        # Get general answers:
        if np.any([face_both, object_both, letter_both, false_both]):
            ch_onset_offset = "both"
        elif np.any([face_offset, object_offset, letter_offset, false_offset]):
            ch_onset_offset = "offset"
        elif np.any([face_onset, object_onset, letter_onset, false_onset]):
            ch_onset_offset = "onset"
        else:
            ch_onset_offset = None
    else:
        face_onset = None
        object_onset = None
        letter_onset = None
        false_onset = None
        face_offset = None
        object_offset = None
        letter_offset = None
        false_offset = None
        ch_onset_offset = None
    # Finally, extract the ROIs and coordinates for this particular channel:
    if ch in channels_rois_wang["channel"].to_list():
        ch_wang_roi = channels_rois_wang.loc[channels_rois_wang["channel"] == ch, "region"].item()
    else:
        ch_wang_roi = None
    if ch in channels_rois_destrieux["channel"].to_list():
        ch_destrieux_roi = channels_rois_destrieux.loc[channels_rois_destrieux["channel"] == ch, "region"].item()
    else:
        ch_destrieux_roi = None
    if ch in channels_rois_desikan["channel"].to_list():
        ch_desikan_roi = channels_rois_desikan.loc[channels_rois_desikan["channel"] == ch, "region"].item()
    else:
        ch_desikan_roi = None
    x = ch_coords.loc[ch_coords["channel"] == ch, "x"].item()
    y = ch_coords.loc[ch_coords["channel"] == ch, "y"].item()
    z = ch_coords.loc[ch_coords["channel"] == ch, "z"].item()
    # Put together in our results table:
    channels_summary_table = channels_summary_table.append(pd.DataFrame({
        "subject": ch.split("-")[0],
        "channel": ch,
        "x": x,
        "y": y,
        "z": z,
        "Destrieux label": ch_destrieux_roi,
        "Wang label": ch_wang_roi,
        "Desikan_label": ch_desikan_roi,
        "Theory ROI": theory_roi,
        "responsiveness": onset_type,
        "responsiveness_bayes": is_responsive_bayes,
        "resp dprime ti": ti_onset_strength,
        "resp dprime tr": tr_onset_strength,
        "latency ti": ti_latency,
        "latency tr": tr_latency,
        "selectivity": selectivity,
        "sel dprime ti": dprime_ti,
        "sel dprime tr": dprime_tr,
        "model ti HGP": ch_model_ti,
        "model tr HGP": ch_model_tr,
        "model ti alpha": ch_model_ti_alpha,
        "model tr alpha": ch_model_tr_alpha,
        "model ti ERP": ch_model_ti_erp,
        "model tr ERP": ch_model_tr_erp,
        "duration_decoding": duration_decoding,
        "duration_decoding_uncorr_ti": decoding_ti_uncorr,
        "duration_decoding_uncorr_tr": decoding_tr_uncorr,
        "face_dur_decoding_ti": face_duration_decoding_accuracy_ti,
        "face_dur_decoding_tr": face_duration_decoding_accuracy_tr,
        "object_dur_decoding_ti": object_duration_decoding_accuracy_ti,
        "object_dur_decoding_tr": object_duration_decoding_accuracy_tr,
        "letter_dur_decoding_ti": letter_duration_decoding_accuracy_ti,
        "letter_dur_decoding_tr": letter_duration_decoding_accuracy_tr,
        "false_dur_decoding_ti": false_duration_decoding_accuracy_ti,
        "false_dur_decoding_tr": false_duration_decoding_accuracy_tr,
        "duration_tracking": duration_tracking,
        "duration_tracking_uncorr_ti": tracking_ti_uncorr,
        "duration_tracking_uncorr_tr": tracking_tr_uncorr,
        "face_dur_tracking_ti": face_duration_tracking_accuracy_ti,
        "face_dur_tracking_tr": face_duration_tracking_accuracy_tr,
        "object_dur_tracking_ti": object_duration_tracking_accuracy_ti,
        "object_dur_tracking_tr": object_duration_tracking_accuracy_tr,
        "letter_dur_tracking_ti": letter_duration_tracking_accuracy_ti,
        "letter_dur_tracking_tr": letter_duration_tracking_accuracy_tr,
        "false_dur_tracking_ti": false_duration_tracking_accuracy_ti,
        "false_dur_tracking_tr": false_duration_tracking_accuracy_tr,
        "onset_offset": ch_onset_offset,
        "face_onset": face_onset,
        "face_offset": face_offset,
        "object_onset": object_onset,
        "object_offset": object_offset,
        "letter_onset": letter_onset,
        "letter_offset": letter_offset,
        "false_onset": false_onset,
        "false_offset": false_offset
    }, index=[0]), ignore_index=True)

# Save the table:
channels_summary_table.to_csv(Path(bids_root, "derivatives", "all_channels_info.csv"))

# =====================================================================================
# Generate summary tables:
# Make a summary by counting how many of each group we have in each ROIs:
destrieux_summaries = pd.DataFrame()
for lbl in list(channels_summary_table["Destrieux label"].unique()):
    # Extract only this label's channels:
    lbl_data = channels_summary_table.loc[channels_summary_table["Destrieux label"] == lbl]
    # Count how many responsive electrodes we have:
    n_activated = lbl_data.loc[lbl_data["responsiveness"] == "activated"].shape[0]
    n_deactivated = lbl_data.loc[lbl_data["responsiveness"] == "deactivated"].shape[0]
    mean_lat_ti = np.nanmean(lbl_data["latency ti"].to_numpy())
    mean_lat_tr = np.nanmean(lbl_data["latency tr"].to_numpy())
    std_lat_ti = np.nanstd(lbl_data["latency ti"].to_numpy())
    std_lat_tr = np.nanstd(lbl_data["latency tr"].to_numpy())
    n_face_selective = lbl_data.loc[lbl_data["selectivity"] == "face"].shape[0]
    n_object_selective = lbl_data.loc[lbl_data["selectivity"] == "object"].shape[0]
    n_letter_selective = lbl_data.loc[lbl_data["selectivity"] == "letter"].shape[0]
    n_false_selective = lbl_data.loc[lbl_data["selectivity"] == "false"].shape[0]
    n_models_ti = {}
    n_models_tr = {}
    for model in models:
        n_models_ti[model] = lbl_data.loc[lbl_data["model ti HGP"] == model].shape[0]
        n_models_tr[model] = lbl_data.loc[lbl_data["model tr HGP"] == model].shape[0]

    # Put everything together in the table:
    destrieux_summaries = destrieux_summaries.append(pd.DataFrame({
        "Destrieux label": lbl,
        "# Activated": n_activated,
        "# Deactivated": n_deactivated,
        "Mean latency TI": mean_lat_ti,
        "STD latency TI": std_lat_ti,
        "Mean latency TR": mean_lat_tr,
        "STD latency TR": std_lat_tr,
        "# Face selective": n_face_selective,
        "# Object selective": n_face_selective,
        "# Letter selective": n_letter_selective,
        "# False selective": n_false_selective,
        **n_models_ti,
        **n_models_tr,
        "# All": lbl_data.shape[0]
    }, index=[0]), ignore_index=True)

# Save to file:
destrieux_summaries.to_csv(Path(bids_root, "derivatives", "Destrieux_labels_summary.csv"))

# Same for the WANG
wang_summaries = pd.DataFrame()
for lbl in list(channels_summary_table["Wang label"].unique()):
    # Extract only this label's channels:
    lbl_data = channels_summary_table.loc[channels_summary_table["Wang label"] == lbl]
    # Count how many responsive electrodes we have:
    n_activated = lbl_data.loc[lbl_data["responsiveness"] == "activated"].shape[0]
    n_deactivated = lbl_data.loc[lbl_data["responsiveness"] == "deactivated"].shape[0]
    mean_lat_ti = np.nanmean(lbl_data["latency ti"].to_numpy())
    mean_lat_tr = np.nanmean(lbl_data["latency tr"].to_numpy())
    std_lat_ti = np.nanstd(lbl_data["latency ti"].to_numpy())
    std_lat_tr = np.nanstd(lbl_data["latency tr"].to_numpy())
    n_face_selective = lbl_data.loc[lbl_data["selectivity"] == "face"].shape[0]
    n_object_selective = lbl_data.loc[lbl_data["selectivity"] == "object"].shape[0]
    n_letter_selective = lbl_data.loc[lbl_data["selectivity"] == "letter"].shape[0]
    n_false_selective = lbl_data.loc[lbl_data["selectivity"] == "false"].shape[0]
    n_models_ti = {}
    n_models_tr = {}
    for model in models:
        n_models_ti[model] = lbl_data.loc[lbl_data["model ti HGP"] == model].shape[0]
        n_models_tr[model] = lbl_data.loc[lbl_data["model tr HGP"] == model].shape[0]

    # Put everything together in the table:
    wang_summaries = wang_summaries.append(pd.DataFrame({
        "Wang label": lbl,
        "# Activated": n_activated,
        "# Deactivated": n_deactivated,
        "Mean latency TI": mean_lat_ti,
        "STD latency TI": std_lat_ti,
        "Mean latency TR": mean_lat_tr,
        "STD latency TR": std_lat_tr,
        "# Face selective": n_face_selective,
        "# Object selective": n_face_selective,
        "# Letter selective": n_letter_selective,
        "# False selective": n_false_selective,
        **n_models_ti,
        **n_models_tr,
        "# All": lbl_data.shape[0]
    }, index=[0]), ignore_index=True)

# Save to file:
wang_summaries.to_csv(Path(bids_root, "derivatives", "Wang_labels_summary.csv"))

# =============================================================================================
# Single subject summary tables:
single_subject_count = pd.DataFrame()
for subject in list(channels_summary_table["subject"].unique()):
    # Extract the data for this subject:
    sub_data = channels_summary_table.loc[channels_summary_table["subject"] == subject]
    # Count the electrodes showing significance in each of the relevant tests:
    n_act_ch = sub_data.loc[sub_data["responsiveness"] == "activated"].shape[0]
    n_deact_ch = sub_data.loc[sub_data["responsiveness"] == "deactivated"].shape[0]
    n_face_sel_ch = sub_data.loc[sub_data["selectivity"] == "face"].shape[0]
    n_object_sel_ch = sub_data.loc[sub_data["selectivity"] == "object"].shape[0]
    n_letter_sel_ch = sub_data.loc[sub_data["selectivity"] == "letter"].shape[0]
    n_false_sel_ch = sub_data.loc[sub_data["selectivity"] == "false"].shape[0]
    n_iit = sub_data.loc[sub_data["Theory ROI"] == "iit"].shape[0]
    n_gnw = sub_data.loc[sub_data["Theory ROI"] == "gnw"].shape[0]
    n_all = sub_data.shape[0]
    single_subject_count = single_subject_count.append(pd.DataFrame({
        "subject": subject,
        "# Activated": n_act_ch,
        "# Deactivated": n_deact_ch,
        "# Face selective": n_face_sel_ch,
        "# Object selective": n_object_sel_ch,
        "# Letter selective": n_letter_sel_ch,
        "# False selective": n_false_sel_ch,
        "# IIT": n_iit,
        "# GNW": n_gnw,
        "# All": n_all
    }, index=[0]), ignore_index=True)
# Save the table:
single_subject_count.to_csv(Path(bids_root, "derivatives", "single_subject_counts.csv"))

# =============================================================================================
# Brain plots:
# Theories rois:
theory_roi_colors = pd.DataFrame()
for theory in theories_rois.rois:
    theory_color = param["colors"][theory]
    theory_roi_colors = theory_roi_colors.append(pd.DataFrame({
        "roi": [roi.replace("ctx_rh_", "").replace("ctx_lh_", "") for roi in theories_rois.rois[theory]],
        "r": theory_color[0],
        "g": theory_color[1],
        "b": theory_color[2]
    }), ignore_index=True)
# Save to file:
theory_roi_colors.to_csv("theory_rois_dict.csv")

# IIT ROI:
iit_roi_colors = pd.DataFrame()
theory_color = param["colors"]["iit"]
iit_roi_colors = iit_roi_colors.append(pd.DataFrame({
    "roi": [roi.replace("ctx_rh_", "").replace("ctx_lh_", "") for roi in theories_rois.rois["iit"]],
    "r": theory_color[0],
    "g": theory_color[1],
    "b": theory_color[2]
}), ignore_index=True)
# Save to file:
iit_roi_colors.to_csv("iit_rois_dict.csv")

anat_roi_colors = pd.DataFrame()
for roi in rois:
    theory_color = roi_colors[roi]
    anat_roi_colors = anat_roi_colors.append(pd.DataFrame({
        "roi": [roi.replace("ctx-rh-", "").replace("ctx-lh-", "") for roi in rois[roi]],
        "r": theory_color[0],
        "g": theory_color[1],
        "b": theory_color[2]
    }), ignore_index=True)
# Save to file:
anat_roi_colors.to_csv("anatomical_rois_dict.csv")

# ===============================================
# Visual responsiveness:
ch_coords = pd.DataFrame()
ch_colors_ti = pd.DataFrame()
ch_colors_tr = pd.DataFrame()
# Get the dprime ranges for the channels radius:
min_dprime = np.nanmin(np.concatenate([channels_summary_table["resp dprime ti"].to_numpy(),
                                       channels_summary_table["resp dprime tr"].to_numpy()], axis=0))
max_dprime = np.percentile(np.concatenate([channels_summary_table["resp dprime ti"].dropna().to_numpy(),
                                           channels_summary_table["resp dprime tr"].dropna().to_numpy()], axis=0), 90)
# Normalize the color bar:
norm = mpl.colors.TwoSlopeNorm(vmin=min_dprime, vcenter=0, vmax=max_dprime)
activation_scalar_map = cm.ScalarMappable(norm=norm, cmap=activation_cmap)

# Plot the activation color bar:
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=activation_cmap, norm=norm)
cb.ax.set_yscale('linear')  # To make sure that the spacing is correct despite normalization
plt.savefig("responsiveness_cbar.png", bbox_inches='tight', transparent=True)
plt.savefig("responsiveness_cbar.svg", bbox_inches='tight', transparent=True)
plt.close()
# Loop through each channel:
for ch in channels_summary_table["channel"].to_list():
    # Get the channel info:
    xyz = [channels_summary_table.loc[channels_summary_table["channel"] == ch, "x"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "y"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "z"].item()]
    # Responsiveness:
    ch_responsiveness = channels_summary_table.loc[channels_summary_table["channel"] == ch, "responsiveness"].item()
    resp_dprime_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "resp dprime ti"].item()
    resp_dprime_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "resp dprime tr"].item()

    # Parse that info:
    if ch_responsiveness not in ["activated", "deactivated"]:
        continue
    ch_coords = ch_coords.append(pd.DataFrame({
        "channel": ch,
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2],
        "radius": 4
    }, index=[0]), ignore_index=True)
    ch_colors_ti = ch_colors_ti.append(pd.DataFrame({
        "channel": ch,
        "r": activation_scalar_map.to_rgba(resp_dprime_ti)[0],
        "g": activation_scalar_map.to_rgba(resp_dprime_ti)[1],
        "b": activation_scalar_map.to_rgba(resp_dprime_ti)[2],
        "dprime": resp_dprime_ti
    }, index=[0]), ignore_index=True)
    ch_colors_tr = ch_colors_tr.append(pd.DataFrame({
        "channel": ch,
        "r": activation_scalar_map.to_rgba(resp_dprime_tr)[0],
        "g": activation_scalar_map.to_rgba(resp_dprime_tr)[1],
        "b": activation_scalar_map.to_rgba(resp_dprime_tr)[2],
        "dprime": resp_dprime_tr
    }, index=[0]), ignore_index=True)

# Save the coordinate tables to csvs:
ch_coords.to_csv("responsiveness_coords.csv")
ch_colors_ti.to_csv("responsiveness_coords_colors_ti.csv")
ch_colors_tr.to_csv("responsiveness_coords_colors_tr.csv")

# ===============================================
# Category selectivity:
ch_coords = pd.DataFrame()
ch_colors_ti = pd.DataFrame()
ch_colors_tr = pd.DataFrame()

# Loop through each channel:
for ch in channels_summary_table["channel"].to_list():
    # Get the channel info:
    xyz = [channels_summary_table.loc[channels_summary_table["channel"] == ch, "x"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "y"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "z"].item()]
    # Selectivity:
    ch_selectivity = channels_summary_table.loc[channels_summary_table["channel"] == ch, "selectivity"].item()
    sel_dprime_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "sel dprime ti"].item()
    sel_dprime_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "sel dprime tr"].item()

    # Parse that info:
    if ch_selectivity is None:
        continue
    ti_color = selectivity_colors[ch_selectivity]
    tr_color = selectivity_colors[ch_selectivity]
    ch_coords = ch_coords.append(pd.DataFrame({
        "channel": ch,
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2],
        "radius": 3
    }, index=[0]), ignore_index=True)
    ch_colors_ti = ch_colors_ti.append(pd.DataFrame({
        "channel": ch,
        "r": ti_color[0],
        "g": ti_color[1],
        "b": ti_color[2]
    }, index=[0]), ignore_index=True)
    ch_colors_tr = ch_colors_tr.append(pd.DataFrame({
        "channel": ch,
        "r": tr_color[0],
        "g": tr_color[1],
        "b": tr_color[2]
    }, index=[0]), ignore_index=True)

# Save the coordinate tables to csvs:
ch_coords.to_csv("category_selectivity_coords_ti.csv")
ch_colors_ti.to_csv("category_selectivity_coords_colors_ti.csv")
ch_colors_tr.to_csv("category_selectivity_coords_colors_tr.csv")

# ======================================================================================================================
# Duration decoding:
# Set up the colors:
accuracy_cmap = plt.get_cmap("Reds")
norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
accuracy_scalar_map = cm.ScalarMappable(norm=norm, cmap=accuracy_cmap)
ch_coords = pd.DataFrame()
ch_colors_ti = pd.DataFrame()
ch_colors_tr = pd.DataFrame()
ch_rois = pd.DataFrame()
# Loop through each channel:
for ch in channels_summary_table["channel"].to_list():
    # Locate the channel results in each table:
    face_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "face_dur_decoding_ti"].item()
    face_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "face_dur_decoding_tr"].item()
    object_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "object_dur_decoding_ti"].item()
    object_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "object_dur_decoding_tr"].item()
    letter_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "letter_dur_decoding_ti"].item()
    letter_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "letter_dur_decoding_tr"].item()
    false_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "false_dur_decoding_ti"].item()
    false_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "false_dur_decoding_tr"].item()

    # Make the conjunction:
    if not np.isnan(face_res_ti):
        face_decoding = True
        face_accuracy_ti = face_res_ti
        face_accuracy_tr = face_res_tr
    else:
        face_decoding = False
        face_accuracy_ti = 0
        face_accuracy_tr = 0
    if not np.isnan(object_res_ti):
        object_decoding = True
        object_accuracy_ti = object_res_ti
        object_accuracy_tr = object_res_tr
    else:
        object_decoding = False
        object_accuracy_ti = 0
        object_accuracy_tr = 0

    if not np.isnan(letter_res_ti):
        letter_decoding = True
        letter_accuracy_ti = letter_res_ti
        letter_accuracy_tr = letter_res_tr
    else:
        letter_decoding = False
        letter_accuracy_ti = 0
        letter_accuracy_tr = 0

    if not np.isnan(false_res_ti):
        false_decoding = True
        false_accuracy_ti = false_res_ti
        false_accuracy_tr = false_res_tr
    else:
        false_decoding = False
        false_accuracy_ti = 0
        false_accuracy_tr = 0

    # Check whether there are any conditions for which we have decoding
    if any([face_decoding, object_decoding, letter_decoding, false_decoding]):
        max_decoding_ti = max([face_accuracy_ti, object_accuracy_ti, letter_accuracy_ti,
                               false_accuracy_ti])
        max_decoding_tr = max([face_accuracy_tr, object_accuracy_tr, letter_accuracy_tr,
                               false_accuracy_tr])
    else:
        continue
    c_ti = accuracy_scalar_map.to_rgba(max_decoding_ti)
    c_tr = accuracy_scalar_map.to_rgba(max_decoding_tr)
    xyz = [channels_summary_table.loc[channels_summary_table["channel"] == ch, "x"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "y"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "z"].item()]
    ch_theory = channels_summary_table.loc[channels_summary_table["channel"] == ch, "Theory ROI"].item()
    if ch_theory not in ["iit", "gnw"]:
        continue
    # Append to the table:
    ch_coords = ch_coords.append(pd.DataFrame({
        "channel": ch,
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2],
        "radius": 3
    }, index=[0]), ignore_index=True)
    ch_colors_ti = ch_colors_ti.append(pd.DataFrame({
        "channel": ch,
        "r": c_ti[0],
        "g": c_ti[1],
        "b": c_ti[2],
    }, index=[0]), ignore_index=True)
    ch_colors_tr = ch_colors_tr.append(pd.DataFrame({
        "channel": ch,
        "r": c_tr[0],
        "g": c_tr[1],
        "b": c_tr[2],
    }, index=[0]), ignore_index=True)
    ch_rois = ch_rois.append(pd.DataFrame({
        "channel": ch,
        "roi": channels_summary_table.loc[channels_summary_table["channel"] == ch, "Destrieux label"].item()
    }, index=[0]), ignore_index=True)

# Save the data:
if len(ch_coords) > 0:
    ch_coords.to_csv("duration_decoding_coords.csv")
    ch_colors_ti.to_csv("duration_decoding_coords_colors_ti.csv")
    ch_colors_tr.to_csv("duration_decoding_coords_colors_tr.csv")
    ch_rois.to_csv("duration_decoding_ch_roi.csv")

# Plot the onset and offset color bars:
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=accuracy_cmap, norm=norm)
plt.savefig("duration_decoding_accuracy_cbar.png", bbox_inches='tight', transparent=True)
plt.savefig("duration_decoding_accuracy_cbar.svg", bbox_inches='tight', transparent=True)
plt.close()

# ======================================================================================================================
# Duration tracking:
# Get the max and min values:
min_track_prop = np.nanmin(np.abs(np.concatenate([channels_summary_table["face_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["face_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["object_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["object_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["letter_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["letter_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["false_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["false_dur_tracking_tr"].to_numpy()], axis=0)))
max_track_prop = np.nanmax(np.abs(np.concatenate([channels_summary_table["face_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["face_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["object_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["object_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["letter_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["letter_dur_tracking_tr"].to_numpy(),
                                                  channels_summary_table["false_dur_tracking_ti"].to_numpy(),
                                                  channels_summary_table["false_dur_tracking_tr"].to_numpy()],
                                                 axis=0)))
# Set up the colors:
accuracy_cmap = plt.get_cmap("Reds")
norm = mpl.colors.Normalize(vmin=min_track_prop, vmax=max_track_prop)
tracking_scalar_map = cm.ScalarMappable(norm=norm, cmap=accuracy_cmap)
ch_coords = pd.DataFrame()
ch_colors_ti = pd.DataFrame()
ch_colors_tr = pd.DataFrame()
# Loop through each channel:
for ch in channels_summary_table["channel"].to_list():
    # Locate the channel results in each table:
    face_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "face_dur_tracking_ti"].item()
    face_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "face_dur_tracking_tr"].item()
    object_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "object_dur_tracking_ti"].item()
    object_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "object_dur_tracking_tr"].item()
    letter_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "letter_dur_tracking_ti"].item()
    letter_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "letter_dur_tracking_tr"].item()
    false_res_ti = channels_summary_table.loc[channels_summary_table["channel"] == ch, "false_dur_tracking_ti"].item()
    false_res_tr = channels_summary_table.loc[channels_summary_table["channel"] == ch, "false_dur_tracking_tr"].item()

    # Make the conjunction:
    if not np.isnan(face_res_ti):
        face_decoding = True
        face_accuracy_ti = face_res_ti
        face_accuracy_tr = face_res_tr
    else:
        face_decoding = False
        face_accuracy_ti = 0
        face_accuracy_tr = 0
    if not np.isnan(object_res_ti):
        object_decoding = True
        object_accuracy_ti = object_res_ti
        object_accuracy_tr = object_res_tr
    else:
        object_decoding = False
        object_accuracy_ti = 0
        object_accuracy_tr = 0

    if not np.isnan(letter_res_ti):
        letter_decoding = True
        letter_accuracy_ti = letter_res_ti
        letter_accuracy_tr = letter_res_tr
    else:
        letter_decoding = False
        letter_accuracy_ti = 0
        letter_accuracy_tr = 0

    if not np.isnan(false_res_ti):
        false_decoding = True
        false_accuracy_ti = false_res_ti
        false_accuracy_tr = false_res_tr
    else:
        false_decoding = False
        false_accuracy_ti = 0
        false_accuracy_tr = 0

    # Check whether there are any conditions for which we have decoding
    if any([face_decoding, object_decoding, letter_decoding, false_decoding]):
        max_decoding_ti = max([face_accuracy_ti, object_accuracy_ti, letter_accuracy_ti,
                               false_accuracy_ti])
        max_decoding_tr = max([face_accuracy_tr, object_accuracy_tr, letter_accuracy_tr,
                               false_accuracy_tr])
    else:
        continue
    c_ti = tracking_scalar_map.to_rgba(max_decoding_ti)
    c_tr = tracking_scalar_map.to_rgba(max_decoding_tr)
    xyz = [channels_summary_table.loc[channels_summary_table["channel"] == ch, "x"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "y"].item(),
           channels_summary_table.loc[channels_summary_table["channel"] == ch, "z"].item()]
    # Append to the table:
    ch_coords = ch_coords.append(pd.DataFrame({
        "channel": ch,
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2],
        "radius": 3
    }, index=[0]), ignore_index=True)
    ch_colors_ti = ch_colors_ti.append(pd.DataFrame({
        "channel": ch,
        "r": c_ti[0],
        "g": c_ti[1],
        "b": c_ti[2],
    }, index=[0]), ignore_index=True)
    ch_colors_tr = ch_colors_tr.append(pd.DataFrame({
        "channel": ch,
        "r": c_tr[0],
        "g": c_tr[1],
        "b": c_tr[2],
    }, index=[0]), ignore_index=True)

# Save the data:
if len(ch_coords) > 0:
    ch_coords.to_csv("duration_tracking_coords.csv")
    ch_colors_ti.to_csv("duration_tracking_coords_colors_ti.csv")
    ch_colors_tr.to_csv("duration_tracking_coords_colors_tr.csv")

# Plot the onset and offset color bars:
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                               cmap=accuracy_cmap, norm=norm)
plt.savefig("duration_tracking_accuracy_cbar.png", bbox_inches='tight', transparent=True)
plt.savefig("duration_tracking_accuracy_cbar.svg", bbox_inches='tight', transparent=True)
plt.close()

# ======================================================================================================================
# Activation analysis:

# ==========================================================
# HGP:
for task in ["ti", "tr"]:
    for signal in ["HGP", "alpha", "ERP"]:
        # ============================
        # Task relevant:
        ch_coords = pd.DataFrame()
        ch_colors = pd.DataFrame()
        # Loop through each channel:
        for ch in channels_summary_table["channel"].to_list():
            if channels_summary_table.loc[channels_summary_table["channel"] == ch, "Theory ROI"].item() is None:
                continue
            # Get the model for that channel:
            ch_model = channels_summary_table.loc[channels_summary_table["channel"] == ch,
                                                  "model {} {}".format(task, signal)].item()
            if ch_model not in models:
                continue
            # Get the color for  that model:
            mdl_color = model_colors[ch_model]
            # Add to the table:
            xyz = [channels_summary_table.loc[channels_summary_table["channel"] == ch, "x"].item(),
                   channels_summary_table.loc[channels_summary_table["channel"] == ch, "y"].item(),
                   channels_summary_table.loc[channels_summary_table["channel"] == ch, "z"].item()]
            # Append to the table:
            ch_coords = ch_coords.append(pd.DataFrame({
                "channel": ch,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "radius": 3
            }, index=[0]), ignore_index=True)
            ch_colors = ch_colors.append(pd.DataFrame({
                "channel": ch,
                "r": mdl_color[0],
                "g": mdl_color[1],
                "b": mdl_color[2],
            }, index=[0]), ignore_index=True)

        ch_coords.to_csv("activation_{}_{}_coords.csv".format(signal, task))
        ch_colors.to_csv("activation_{}_{}_colors.csv".format(signal, task))
