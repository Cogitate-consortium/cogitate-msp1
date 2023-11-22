"""
Plot activation analysis results

@author: Oscar Ferrante oscfer88@gmail.com
"""

import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import os.path as op
from scipy import stats

import mne_bids

from plotters import plot_time_series, plot_brain
import config


phase = "3"                             # available options: "2" or "3"
task_rel = "Relevant non-target"        # available options: "Relevant non-target" or "Irrelevant"

freq_bands = ['alpha', 'gamma']
sub = f"groupphase{phase}"
ses = "V1"
task = "dur"
data_type = "meg"

bids_root = r"/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids"
analysis_name = "source_dur"
source_deriv_root = Path(bids_root, "derivatives", analysis_name)
save_root="/hpc/users/oscar.ferrante/plotting_activ"

param = config.param


def load_labels_names():
    # Read list with label names
    bids_path_source = mne_bids.BIDSPath(
                        root=source_deriv_root,
                        subject="CA124",  # any is good
                        datatype=data_type,
                        task=task,
                        session=ses,
                        suffix="desc-labels",
                        extension='.txt',
                        check=False)
    labels_names = open(bids_path_source.fpath, 'r').read()
    labels_names = labels_names[2:-2].split("', '")

    return labels_names


def load_spectral_act_df(root=source_deriv_root):
    # Read dataframe
    bids_path_source = mne_bids.BIDSPath(
        root=root,
        subject=sub,
        datatype=data_type,
        task=task,
        session=ses,
        suffix="datatable",
        extension='.tsv',
        check=False)
    df = pd.read_csv(bids_path_source.fpath, sep="\t")

    # Rename "False" as "false"
    df.loc[df['Category']==False, 'Category'] = 'false'

    # Get times
    times = np.array([float(t) for t in list(df.columns)[1:-5]])

    return df, times


def load_parcels_bestmodel_df(freq_band="alpha"):
    # Set file name
    if task_rel == "Irrelevant":
        filename = f"MEG_activation_table_p{phase}.tsv"
        t_name  = "irrelevant"
    elif task_rel == "Relevant non-target":
        filename = f"MEG_activation_table_p{phase}_rel.tsv"
        t_name = "relevant"
    else:
        raise ValueError("Error: incorrect values for task_rel")

    # Load .tsv
    df = pd.read_csv(op.join(save_root,filename),sep='\t')

    # Select frequency band and task relevance
    df = df.drop(df[(df.frequency_band != freq_band) | (df.task != t_name)].index)

    # Remove non-parcel results
    df = df.drop(df[(df.ROI == "gnw_all") | (df.ROI == "iit_all")].index)

    # Replace "&" with "_and_" in the ROI label names
    df.ROI = [l.replace("&", "_and_") for l in df.ROI]

    return df


def mean_ci_groupby(df, groupby=['band','label','Category','Duration'], zscore=False, design="within"):
    # Get number of participants
    df_groups = df.groupby(groupby).groups

    # Average across participants
    df_m = df.groupby(groupby).mean()

    # Z-score data by group
    if zscore:
        df_m = stats.zscore(df_m, axis=1)
        df_ci = pd.DataFrame(np.zeros(df_m.shape)) #fake values
    else:
        # Compute condidence intervals
        if design == "within": # within-subject design CIs from Cousineau (2005)
            df_new = df.groupby(["sub"]+groupby).mean().reset_index()
            for s in np.unique(df['sub']):
                sbj = np.array(df_new.query(f"sub == '{s}'"))[:,2:]
                sbj_m = np.mean(sbj, axis=0)
                grp_m = np.array(df_m)
                new_sbj = sbj - sbj_m + grp_m
                df_new.loc[df_new["sub"] == s, df_new.columns[2:]] = new_sbj
            df_ci = df_new.groupby(groupby).sem() * 1.96
        elif design == "between":
            n = len(np.unique(df['sub']))
            df_ci = df.groupby(groupby).sem() * stats.t.ppf((1 + .95) / 2., n - 1)

    return df_groups, df_m , df_ci


def plotting_brain_bestmodel_parcels(bands):
    print("\nPlotting brain with model-coded parcels")
    # Loop over freq bands
    for band in bands:
        print('\nfreq_band:', band)

        # Load best model result per parcel
        df_bm = load_parcels_bestmodel_df(freq_band=band)

        # Set model values
        model_values = {"null_model": 0.,
                        "time_win": 0.,
                        "duration": 0.,
                        "time_win_dur": 0.,
                        "time_win_dur_gnw": 1.,
                        "time_win_dur_iit": 2.,
                        "time_win_dur_cate_gnw": 3.,
                        "time_win_dur_cate_iit": 4.}

        # Create colormap
        model_colors = {"null_model/time_win/duration/time_win_dur": [1., 1., 1.],
                        # "time_win": [1., 1., 1.],
                        # "duration": [1., 1., 1.],
                        # "time_win_dur": [1., .6, .6],
                        "time_win_dur_gnw": [
                            0.00784313725490196,
                            0.6196078431372549,
                            0.45098039215686275],
                        "time_win_dur_iit": [
                            0.00392156862745098,
                            0.45098039215686275,
                            0.6980392156862745],
                        "time_win_dur_cate_gnw": [
                            0.792156862745098,
                            0.5686274509803921,
                            0.3803921568627451],
                        "time_win_dur_cate_iit": [
                            0.8352941176470589,
                            0.3686274509803922,
                            0.0]
                        }
        cmap = matplotlib.colors.ListedColormap(list(model_colors.values()))

        # Set best model per parcel
        sign_values = {}
        for l, label in enumerate(df_bm.ROI):
            best_model = df_bm.iloc[l]["best_model"]
            sign_values[label[4:]] = model_values[best_model]

        # Plotting brain surface
        folder = op.join(
            save_root, phase, task_rel, band)
        if not op.exists(folder):
            os.makedirs(folder)
        filename = op.join(
            folder,
            f"sourceparcel-sub-{sub}-desc-{band}-{task_rel}_brain.png")
        plot_brain(
            roi_map=sign_values, views=['lateral', 'caudal', 'medial'],
            cmap=cmap, colorbar=False, colorbar_title='Model', roi_map_edge_color=[0., 0., 0.],
            vmin=np.min(list(model_values.values())), vmax=np.max(list(model_values.values())),
            figsize=(18, 6), save_file=filename)


def plotting_spectral_time_series(labels, tmin = -.5, tmax = 2.2, vlines = [0, 0.5, 1.0, 1.5]):
    print("\n Plotting spectral time series")
    # Load data
    df, times = load_spectral_act_df(root=source_deriv_root)

    # Loop over freq bands and labels
    for band in freq_bands:
        print('freq_band:', band)

        for label in labels:
            print('label:', label)

            # Select data
            df_temp = df.query(f"band == '{band}' and label == '{label}' and Task_relevance == '{task_rel}'")

            # Average and compute CI across participants
            df_groups, df_m , df_ci = mean_ci_groupby(df_temp, groupby=["Duration"])

            # Crop data from tmin to tmax
            itmin = (np.abs(times - tmin)).argmin()
            itmax = (np.abs(times - tmax)).argmin()
            df_m = df_m.iloc[:,itmin:itmax]
            df_ci = df_ci.iloc[:,itmin:itmax]

            # Set plotting params
            if "iit" in label:
                if band == "alpha":
                    ylim=[0.55, 1.35]
                elif band == "gamma":
                    ylim=[0.98, 1.055]
            elif "gnw" in label:
                if band == "alpha":
                    ylim=[0.85, 1.15]
                elif band == "gamma":
                    ylim=[0.98, 1.055]

            conditions = list(df_groups.keys())
            t0 = float(df_m.keys()[0])
            tend = float(df_m.keys()[-1])

            folder = op.join(
            save_root, phase, task_rel, band)
            if not op.exists(folder):
                os.makedirs(folder)
            filename = op.join(
                folder,
                f"sourcedur-sub-{sub}-desc-{band}-{label}-{task_rel}_timecourse.png")

            if "iit" in label:
                colors = {
                    "500ms": [174/255, 226/255, 255/255],
                    "1000ms": [0 / 255, 159 / 255, 245 / 255],
                    "1500ms": [0 / 255, 53 / 255, 82 / 255],
                }
            elif "gnw" in label:
                colors = {
                    "500ms": [114/255, 254/255, 214/255],
                    "1000ms": [1 / 255, 244 / 255, 175 / 255],
                    "1500ms": [0 / 255, 81 / 255, 58 / 255],
                }
            colors = [colors[cond] for cond in conditions]

            # Plot
            plot_time_series(
                np.array(df_m), t0, tend, err=np.array(df_ci), colors=colors,
                vlines=vlines, filename=filename, conditions=conditions,
                xlabel="Time (s)", ylabel="Activation (a.u.)", ylim=ylim,
                err_transparency=0.2, title=None, square_fig=False, do_legend=True,
                patches=None, patch_color="r", patch_transparency=0.2)



def plot_apha_vs_gamma_in_label(labels, zscore=False, tmin = -.5, tmax = 2.2, vlines = [0, 0.5, 1.0, 1.5]):
    print("\nPlotting alpha vs. gamma time series")
    # Loop over labels
    for label in labels:
        print('label:', label)

        # Load data
        df, times = load_spectral_act_df(root=source_deriv_root)

        # Select data
        df_temp = df.query(f"label == '{label}' and Task_relevance == '{task_rel}'")

        # Average and compute CI across participants
        df_groups, df_m , df_ci = mean_ci_groupby(df_temp, groupby=["band"], zscore=zscore)

        # Crop data from tmin to tmax
        itmin = (np.abs(times - tmin)).argmin()
        itmax = (np.abs(times - tmax)).argmin()
        df_m = df_m.iloc[:,itmin:itmax]
        df_ci = df_ci.iloc[:,itmin:itmax]

        # Set plotting params
        conditions = list(df_groups.keys())
        t0 = float(df_m.keys()[0])
        tend = float(df_m.keys()[-1])

        folder = op.join(
            save_root, phase, task_rel, "alpha_vs_gamma")
        if not op.exists(folder):
            os.makedirs(folder)
        if zscore:
            filename = op.join(
                folder,
                f"sourceloc-sub-{sub}-desc-tag-{label}-{task_rel}_z-timecourse.png")
            unit = "z-score"
        else:
            filename = op.join(
                folder,
                f"sourceloc-sub-{sub}-desc-tag-{label}-{task_rel}_timecourse.png")
            unit = "a.u."

        colors = {
            "alpha": [150/255, 150/255, 150/255],
            "gamma": [0/255, 0/255, 0/255],
        }

        colors = [colors[cond] for cond in conditions]

        # Plot
        plot_time_series(
            np.array(df_m), t0, tend, err=np.array(df_ci), colors=colors,
            vlines=vlines, filename=filename, conditions=conditions,
            xlabel="Time (s)", ylabel=f"Activation ({unit})", ylim=None,
            err_transparency=0.4, title=None, square_fig=False, do_legend=True,
            patches=[0., .5], patch_color="y", patch_transparency=0.2)


def plotting_erf_time_series(labels, tmin = -.5, tmax = 2.2, vlines = [0, 0.5, 1.0, 1.5]):
    print("\nPlotting ERF time series")
    # Load data
    df, times = load_spectral_act_df(root=Path(bids_root, "derivatives", analysis_name+"_ERF_oscar"))

    # Loop over labels
    for label in labels:
        print('label:', label)

        # Select data
        df_temp = df.query(f"label == '{label}' and Task_relevance == '{task_rel}'")

        # Average and compute CI across participants
        df_groups, df_m , df_ci = mean_ci_groupby(df_temp, groupby=["Duration"])

        # Crop data from tmin to tmax
        itmin = (np.abs(times - tmin)).argmin()
        itmax = (np.abs(times - tmax)).argmin()
        df_m = df_m.iloc[:,itmin:itmax]
        df_ci = df_ci.iloc[:,itmin:itmax]

        # Set plotting params
        # if "iit" in label:
        #     ylim=[0.55, 1.35]
        # elif "gnw" in label:
        #     ylim=[0.85, 1.15]
        ylim=None

        conditions = list(df_groups.keys())
        t0 = float(df_m.keys()[0])
        tend = float(df_m.keys()[-1])

        folder = op.join(
        save_root, phase, task_rel, "erf")
        if not op.exists(folder):
            os.makedirs(folder)
        filename = op.join(
            folder,
            f"sourcedur-sub-{sub}-desc-erf-{label}-{task_rel}_timecourse.png")

        if "iit" in label:
            colors = {
                "500ms": [174/255, 226/255, 255/255],
                "1000ms": [0 / 255, 159 / 255, 245 / 255],
                "1500ms": [0 / 255, 53 / 255, 82 / 255],
            }
        elif "gnw" in label:
            colors = {
                "500ms": [114/255, 254/255, 214/255],
                "1000ms": [1 / 255, 244 / 255, 175 / 255],
                "1500ms": [0 / 255, 81 / 255, 58 / 255],
            }
        colors = [colors[cond] for cond in conditions]

        # Plot
        plot_time_series(
            np.array(df_m), t0, tend, err=np.array(df_ci), colors=colors,
            vlines=vlines, filename=filename, conditions=conditions,
            xlabel="Time (s)", ylabel="Activation (RMS)", ylim=ylim,
            err_transparency=0.2, title=None, square_fig=False, do_legend=True,
            patches=None, patch_color="r", patch_transparency=0.2)


if __name__ == "__main__":
    plotting_brain_bestmodel_parcels(bands=['alpha', 'gamma', 'late alpha', 'erf'])
    plotting_spectral_time_series(labels=["gnw_all", "gnw_G&S_cingul-Mid-Post",
                                          "iit_all", "iit_Pole_occipital"])
    plot_apha_vs_gamma_in_label(labels=["iit_S_calcarine"], zscore=False)
    plot_apha_vs_gamma_in_label(labels=["iit_S_calcarine"], zscore=True)
    plotting_erf_time_series(labels=["gnw_all", "gnw_G&S_cingul-Mid-Post",
                                     "iit_all", "iit_G&S_occipital_inf"])
