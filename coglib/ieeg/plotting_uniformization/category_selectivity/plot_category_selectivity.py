import pandas as pd
from pathlib import Path
import config
import numpy as np
import os
import ptitprince as pt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from plotters import plot_time_series, mm2inch
import matplotlib.pyplot as plt
from general_utilities import epochs_loader, corrected_sem

param = config.param
bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
analysis_name = "category_selectivity"
preprocessing_folder = "epoching"
signal = "high_gamma"
preprocessing_steps = "desbadcharej_notfil_lapref"
sub = "super"
ses = "V1"
data_type = "ieeg"
conditions = ["stimulus onset/Relevant non-target", "stimulus onset/Irrelevant"]
results_root = Path(bids_root, "derivatives", analysis_name, "sub-" + sub, "ses-" + ses, data_type, "results")
category_order = ["face", "object", "letter", "false"]
duration_order = ["500ms", "1000ms", "1500ms"]
vlines = [0, 0.5, 1.0, 1.5]
ylim = None
crop_time = [-0.3, 2.0]
smooth_time_ms = 50
sfreq = 512
# Convert to samples:
smooth_samp = int(smooth_time_ms * (sfreq / 1000))
fig_size = param["figure_size_mm"]
ylabel = "HGP (norm.)"
colors = {
    "face": [1, 1, 0],
    "object": [1, 0, 0],
    "letter": [0.3, 0.3, 0.3],
    "false": [0.93, 0.51, 0.93]
}


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


def plot_category_selectivity(epochs, channels_selectivity, output_dir_dict, patches=None):
    """

    """
    group_level_results = pd.DataFrame()
    for subject in epochs:
        epo = epochs[subject]
        for ch in epo.ch_names:
            # Skip non selective channels
            if "-".join([subject, ch]) not in list(channels_selectivity.keys()):
                continue
            # Plot separately task relevant and irrelevant:
            for cond in conditions:
                evks = []
                data_sem = []
                bar_plot_data = pd.DataFrame()
                for cate in category_order:
                    evks.append(epo.copy()["/".join([cond, cate])].average().get_data(picks=ch))
                    data_sem.append(np.squeeze(epo.copy()["/".join([cond, cate])].get_data(picks=ch)))
                    avg_data = np.mean(np.squeeze(epo.copy()["/".join([cond, cate])].crop(tmin=0.05,
                                                                                          tmax=0.4).get_data(
                        picks=ch)),
                        axis=-1)
                    bar_plot_data = bar_plot_data.append(pd.DataFrame({
                        "category": [cate] * avg_data.shape[0],
                        "avg": avg_data
                    }), ignore_index=True)
                    group_level_results = group_level_results.append(pd.DataFrame({
                        "selectivity": channels_selectivity["-".join([subject, ch])],
                        "task": cond,
                        "category": cate,
                        "value": np.mean(avg_data)
                    }, index=[0]), ignore_index=True)
                # Compute the corrected SEM:
                errors = corrected_sem(data_sem, len(data_sem))
                c = [colors[cond] for cond in category_order]
                filename = Path(output_dir_dict[channels_selectivity["-".join([subject, ch])]],
                                "sub-{}_ch-{}_task-{}_category_selectivity.png".format(subject, ch, cond.split("/")[1]))
                # Plot:
                avgs = uniform_filter1d(np.array(evks), smooth_samp, axis=-1)
                errors = [uniform_filter1d(error, smooth_samp, axis=-1) for error in errors]
                plot_time_series(np.squeeze(avgs), epo.times[0], epo.times[-1], ax=None, err=errors, colors=c,
                                 vlines=vlines,
                                 ylim=None, xlabel="Time (s)", ylabel=ylabel, err_transparency=0.2,
                                 filename=filename, title=None, square_fig=False, conditions=category_order,
                                 do_legend=False, patches=patches, patch_color="r", patch_transparency=0.2)
                plt.close()

                # Plot half violin:
                fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0] / 2),
                                                mm2inch(fig_size[1])])
                pt.half_violinplot(x="category",
                                   y="avg", data=bar_plot_data, bw=.2, cut=0.,
                                   scale="area", width=.6, inner=None, ax=ax, palette=colors,
                                   order=category_order, alpha=.5)
                # Add the rain:
                ax = sns.stripplot(x="category", y="avg", data=bar_plot_data, palette=colors, edgecolor="white",
                                   size=3, jitter=1, zorder=0, order=category_order,
                                   orient="v")
                # Plot boxplot:
                sns.boxplot(x="category",
                            y="avg", data=bar_plot_data, ax=ax, width=0.2, palette=colors,
                            order=category_order, boxprops={'facecolor': 'none', "zorder": 10})
                ax.set_ylabel("HGP Mean")
                ax.set_xlabel("Category")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                # Save the figure:
                filename = Path(output_dir_dict[channels_selectivity["-".join([subject, ch])]],
                                "sub-{}_ch-{}_task-{}_box_plot_category_selectivity.png".format(subject, ch,
                                                                                                cond.split("/")[1]))
                plt.savefig(filename, transparent=True)
                filename = Path(output_dir_dict[channels_selectivity["-".join([subject, ch])]],
                                "sub-{}_ch-{}_task-{}_box_plot_category_selectivity.svg".format(subject, ch,
                                                                                                cond.split("/")[1]))
                plt.savefig(filename, transparent=True)
                plt.close()

    # Plot the group level results:
    for task in list(group_level_results["task"].unique()):
        task_data = group_level_results.loc[group_level_results["task"] == task]
        # Separately for each selectivities:
        for sel in list(task_data["selectivity"].unique()):
            data = task_data.loc[task_data["selectivity"] == sel]
            # Plot half violin:
            fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0] / 2),
                                            mm2inch(fig_size[1])])
            pt.half_violinplot(x="category",
                               y="value", data=data, bw=.2, cut=0.,
                               scale="area", width=.6, inner=None, ax=ax, palette=colors,
                               order=category_order, alpha=.5)
            # Add the rain:
            ax = sns.stripplot(x="category", y="value", data=data, palette=colors, edgecolor="white",
                               size=3, jitter=1, zorder=0, order=category_order,
                               orient="v")
            # Plot boxplot:
            sns.boxplot(x="category",
                        y="value", data=data, ax=ax, width=0.2, palette=colors,
                        order=category_order, boxprops={'facecolor': 'none', "zorder": 10})
            ax.set_ylabel("HGP Mean")
            ax.set_xlabel("Category")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            # Save the figure:
            filename = Path(output_dir_dict[sel],
                            "sub-{}_ch-{}_task-{}_box_plot_category_selectivity.png".format("group", "all",
                                                                                            task.split("/")[1]))
            plt.savefig(filename, transparent=True)
            filename = Path(output_dir_dict[sel],
                            "sub-{}_ch-{}_task-{}_box_plot_category_selectivity.svg".format("group", "all",
                                                                                            task.split("/")[1]))
            plt.savefig(filename, transparent=True)
            plt.close()


def category_selectivity_handler(result_tbls, subjects=None, save_root="", categories=None):
    if categories is None:
        categories = ["face", "object", "letter", "false"]

    # Read the results table table:
    ti_results_table = pd.read_csv(result_tbls[0])
    tr_results_table = pd.read_csv(result_tbls[1])

    # List the subjects:
    if subjects is None:
        subjects = list(ti_results_table["subject"].unique())
    # Extract only the channels that show selectivity:
    channels_list = []
    for cate in category_order:
        ti_sel = ti_results_table.loc[ti_results_table["condition"] == cate, "channel"].to_list()
        tr_sel = tr_results_table.loc[tr_results_table["condition"] == cate, "channel"].to_list()
        # Keep only the channels that are present in both:
        channels_list.append([ch for ch in ti_sel if ch in tr_sel])
    # Flatten the list:
    channels_list = [item for sublist in channels_list for item in sublist]
    # Dump the rest of the table:
    ti_results_table = ti_results_table.loc[ti_results_table["channel"].isin(channels_list)]
    tr_results_table = tr_results_table.loc[tr_results_table["channel"].isin(channels_list)]

    # Plot the d primes distribution:
    dprimes_tbl = pd.DataFrame()
    positions = {
        "face": 1,
        "object": 4,
        "letter": 7,
        "false": 10
    }
    for cate in ti_results_table["condition"].unique():
        # Get the dprimes for this category:
        ti_cate_dprimes = ti_results_table.loc[ti_results_table["condition"] == cate, "effect_strength"].to_list()
        tr_cate_dprimes = tr_results_table.loc[tr_results_table["condition"] == cate, "effect_strength"].to_list()
        dprimes_tbl = dprimes_tbl.append(pd.DataFrame({
            "category": [cate + " TI"] * len(ti_cate_dprimes),
            "position": positions[cate],
            "d'": ti_cate_dprimes
        }), ignore_index=True)
        dprimes_tbl = dprimes_tbl.append(pd.DataFrame({
            "category": [cate + " TR"] * len(ti_cate_dprimes),
            "position": positions[cate] + 1,
            "d'": tr_cate_dprimes
        }), ignore_index=True)

    # Plot half violin:
    fig, ax = plt.subplots(figsize=[mm2inch(fig_size[0]),
                                    mm2inch(fig_size[0])])
    palette = {1: [1, 1, 0],
                          2: [1, 1, 0],
                          3: [1, 0, 0],
                          4: [1, 0, 0],
                          5: [1, 0, 0],
                          6: [0.3, 0.3, 0.3],
                          7: [0.3, 0.3, 0.3],
                          8: [0.3, 0.3, 0.3],
                          9: [0.76, 0.09, 0.11],
                          10: [0.93, 0.51, 0.93],
                          11: [0.93, 0.51, 0.93]
                          }
    pt.RainCloud(x="position", y="d'", data=dprimes_tbl,
                 palette={1: [1, 1, 0],
                          2: [1, 1, 0],
                          3: [1, 0, 0],
                          4: [1, 0, 0],
                          5: [1, 0, 0],
                          6: [0.3, 0.3, 0.3],
                          7: [0.3, 0.3, 0.3],
                          8: [0.3, 0.3, 0.3],
                          9: [0.76, 0.09, 0.11],
                          10: [0.93, 0.51, 0.93],
                          11: [0.93, 0.51, 0.93]
                          },
                 bw=0.2, order=range(1, 12), cloud_alpha=0,
                 width_viol=0, rain_edgecolor="k", width_box=.5, ax=ax, orient="v", rain_linewidth=0.5)
    ax.set_ylabel("d'")
    ax.set_xlabel("")
    ax.set_xlim([-1, 11])
    ax.set_xticks([0, 1, 3, 4, 6, 7, 9, 10])
    ax.set_xticklabels(["TI", "TR", "TI", "TR", "TI", "TR", "TI", "TR"])
    plt.text(0.25, -0.65, 'Face')
    plt.text(3.25, -0.65, 'Object')
    plt.text(6.25, -0.65, 'Letter')
    plt.text(9.25, -0.65, 'False')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    # plt.tight_layout()
    # Save the figure:
    filename = Path(save_root, "dprime_distribution.png")
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    filename = Path(save_root, "dprime_distribution.svg")
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close()

    # Create a dictionary storing for each channel the selectivitiy:
    channels_selectivity = {ch: ti_results_table.loc[ti_results_table["channel"] == ch, "condition"].item()
                            for ch in ti_results_table["channel"].to_list()}
    # Load the epochs:
    epo_dir = str(Path(bids_root, "derivatives", "preprocessing", "sub-{}",
                       "ses-" + ses, "ieeg", preprocessing_folder,
                       signal, preprocessing_steps))
    epo_file = "sub-{}_ses-{}_task-Dur_desc-epoching_ieeg-epo.fif"
    epochs = epochs_loader(subjects, epo_dir, epo_file, channels_list, crop_time, ses,
                           conditions=conditions)

    # Create one directory for each category:
    cate_dirs = {cate: Path(save_root, cate) for cate in category_order}
    # Create the directory if it does not exist:
    for cate in category_order:
        if not os.path.exists(cate_dirs[cate]):
            os.makedirs(cate_dirs[cate])
    # Now plotting every single electrode we have left:
    plot_category_selectivity(epochs, channels_selectivity, cate_dirs)


if __name__ == "__main__":
    result_tbls = [
        "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/category_selectivity/sub-super/ses-V1/ieeg"
        "/results/high_gamma_dprime_test_ti/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv",
        "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/derivatives/category_selectivity/sub-super/ses-V1/ieeg"
        "/results/high_gamma_dprime_test_tr/desbadcharej_notfil_lapref/sub-super_ses-V1_task-Dur_analysis-category_selectivity_all_results.csv"
    ]
    category_selectivity_handler(result_tbls,
                                 save_root="/hpc/users/alexander.lepauvre/plotting_test/category_selectivity")
