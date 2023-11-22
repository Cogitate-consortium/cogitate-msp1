"""
Plot PPC connectivity analysis results

@author: Oscar Ferrante oscfer88@gmail.com
"""

import numpy as np
from pathlib import Path
import os
import os.path as op
import pickle
from scipy import stats as stats
import scipy.signal as ss

import mne_bids
import mne_connectivity

from plotters import plot_pcolormesh, plot_time_series
import config


t_rel = "_irr"          # options: "", "_irr" or "_rel"
no_evo = "_no-evoked"             # options: "" or "_no-evoked"

bids_root = r"/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids"
analysis_name = "connectivity"
save_root = Path("/hpc/users/oscar.ferrante/plotting_conn")

phase = 3
sub = f"groupphase{phase}"
ses = "V1"
task = "dur"
data_type = "meg"

param = config.param


def load_ppc_data(sub, freq_range, cond_name):
    # Set path
    con_deriv_root = Path(bids_root, "derivatives", analysis_name+t_rel, no_evo)

    bids_path_con = mne_bids.BIDSPath(
        root=con_deriv_root,
        subject=sub,
        datatype="meg",
        task=task,
        session=ses,
        suffix=f"desc-gnw-pfc-ged,ppc,{freq_range},{cond_name}_con",
        extension=".nc",
        check=False)

    # Load con object
    conn = mne_connectivity.read_connectivity(bids_path_con.fpath)

    # Get data
    data = conn.get_data()

    # Get times, freqs and indices_comb
    times = [t - .5 for t in conn.times]
    freqs = conn.freqs
    indices_comb = [[i,j] for i,j in zip(conn.indices[0], conn.indices[1])]

    return data, times, freqs, indices_comb


def load_dfc_data(sub, cond_name):
    # Set path
    con_deriv_root = Path(bids_root, "derivatives", analysis_name+t_rel, "_dfc", no_evo)

    bids_path_con = mne_bids.BIDSPath(
        root=con_deriv_root,
        subject=sub,
        datatype="meg",
        task=task,
        session=ses,
        suffix=f"desc-dfc_{cond_name}_con",
        extension=".npy",
        check=False)

    # Load con data
    data = np.load(bids_path_con.fpath)

    # Load times and freqs
    bids_path_times = bids_path_con.copy().update(
        subject="CA102",
        suffix="desc-dfc_times")
    times = np.load(bids_path_times.fpath)

    bids_path_freqs = bids_path_times.copy().update(
        suffix="desc-dfc_freqs")
    freqs = np.load(bids_path_freqs.fpath)

    # Get indices combination
    n_labels = 2
    indices = (np.concatenate([range(0,n_labels),range(0,n_labels)]),
               np.array([n_labels]*len(range(0,n_labels)) + [n_labels+1]*len(range(0,n_labels))))
    indices_comb = [[i,j] for i,j in zip(indices[0], indices[1])]

    return data, times, freqs, indices_comb



def load_conn_mask(sub, desc, data, indices_comb, con_method="ppc"):
    # Set path
    if con_method == "ppc":
        mtd = ""
    elif con_method == "dfc":
        mtd = "_dfc"

    con_deriv_root = Path(bids_root, "derivatives", analysis_name+t_rel, mtd, no_evo)

    bids_path_con = mne_bids.BIDSPath(
        root=con_deriv_root,
        subject=sub,
        datatype="meg",
        task=task,
        session=ses,
        suffix =f"desc-{desc}_clusters",
        extension=".pkl",
        check=False)

    # Load significant clusters per node couple
    with open(bids_path_con.fpath, 'rb') as file:
        good_clusters_all = pickle.load(file)

    # Generate mask for mode couple
    masks = np.full(data.shape, False)
    for i in indices_comb:
        sign_mask = np.any(good_clusters_all[indices_comb.index(i)], axis=0)
        if sign_mask.any():
            masks[indices_comb.index(i),...] = np.any(good_clusters_all[indices_comb.index(i)], axis=0)

    # Convert list to array
    masks = np.array(masks)

    return masks


def plotting_ppc_matrix():
    print("\nPlotting PPC...")
    # Loop over frequencies
    data_1_all = []
    data_2_all = []
    masks_all = []
    freqs_all = []
    for freq_range in ['low', 'high']:
        print(f'Freq range: {freq_range}')

        # Loop over conditions to contrast
        for conds in [['face', 'object']]:
            print(f"Analysis: {conds[0]} vs {conds[1]}")

            # Load data
            data_1, times, freqs, indices_comb = load_ppc_data(
                sub, freq_range, conds[0])
            data_1_all.append(data_1)
            freqs_all += freqs

            data_2, _, _, _ = load_ppc_data(
                sub, freq_range, conds[1])
            data_2_all.append(data_2)

            # Load mask
            desc = f"gnw-pfc-ged,ppc,{freq_range},{conds}"
            masks = load_conn_mask(
                sub, desc,
                data_1, indices_comb,
                con_method="ppc")
            masks_all.append(masks)

    # Concatenate frequencies
    data_1_all = np.concatenate(data_1_all, axis=1)
    data_2_all = np.concatenate(data_2_all, axis=1)
    masks_all = np.concatenate(masks_all, axis=1)

    # Compute differce
    data_dif_all = data_1_all - data_2_all

    # Set plotting params
    labels = ['pfc','v1v2','face','object']

    # Loop over node couples
    for i in indices_comb:
        print(f'Plotting {labels[i[0]]}-{labels[i[1]]}...')

        # Plot each conditions separately
        vmin = 0.
        vmax = .15

        for j, data in enumerate([data_1_all, data_2_all]):
            # Set output path
            folder = op.join(
                save_root, "ppc", t_rel+no_evo)
            if not op.exists(folder):
                os.makedirs(folder)
            filename = op.join(
                folder,
                f"conn-sub-{sub}-desc-{labels[i[0]]}-{labels[i[1]]}_c-{conds[j]}_ppc{no_evo}.png")

            # Plot and save
            plot_pcolormesh(data[indices_comb.index(i),...],
                            times, freqs_all,  mask=None, transparency=1.,
                            vlim=[vmin, vmax], xlabel="Time (s)", ylabel="Frequency (Hz)",
                            cbar_label="PPC", filename=filename)

        # Plot difference between conditions
        vmin = -.075
        vmax = .075

        # Set output path
        folder = op.join(
            save_root, "ppc", t_rel+no_evo)
        if not op.exists(folder):
            os.makedirs(folder)
        filename = op.join(
            folder,
            f"conn-sub-{sub}-desc-{labels[i[0]]}-{labels[i[1]]}_{conds[0]}-{conds[1]}_ppc{no_evo}.png")

        # Plot and save
        plot_pcolormesh(data_dif_all[indices_comb.index(i),...],
                        times, freqs_all,
                        mask=masks_all[indices_comb.index(i),...], transparency=1.,
                        vlim=[vmin, vmax], xlabel="Time (s)", ylabel="Frequency (Hz)",
                        cbar_label="ΔDFC", filename=filename)




def load_ged_data(label_name="fusifor"):
    print(f"\nLoading {label_name} GED data")
    # Read participant list
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")

    # Set paths and load data
    ged_deriv_root = op.join(bids_root, "derivatives", "ged")

    ged_facFilt_facCond_ts = []
    ged_objFilt_objCond_ts = []
    ged_objFilt_facCond_ts = []
    ged_facFilt_objCond_ts = []
    for s in sub_list:
        print(f"subject: {s}")

        bids_path_ged = mne_bids.BIDSPath(
            root=ged_deriv_root,
            subject=s,
            datatype='meg',
            task=task,
            session=ses,
            suffix=f'desc-{label_name},facFilt_facCond_compts',
            extension='.npy',
            check=False)
        ged_facFilt_facCond_ts.append(np.load(bids_path_ged.fpath))

        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},facFilt_objCond_compts')
        ged_facFilt_objCond_ts.append(np.load(bids_path_ged.fpath))

        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},objFilt_objCond_compts')
        ged_objFilt_objCond_ts.append(np.load(bids_path_ged.fpath))

        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{label_name},objFilt_facCond_compts')
        ged_objFilt_facCond_ts.append(np.load(bids_path_ged.fpath))

    # Average trials within participants
    ged_facFilt_facCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_facFilt_facCond_ts]
    ged_facFilt_objCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_facFilt_objCond_ts]
    ged_objFilt_objCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_objFilt_objCond_ts]
    ged_objFilt_facCond_ts_ev = [np.mean(ged, axis=0) for ged in ged_objFilt_facCond_ts]

    # Merge data sets
    data = np.stack(
        [ged_facFilt_facCond_ts_ev,
        ged_facFilt_objCond_ts_ev,
        ged_objFilt_objCond_ts_ev,
        ged_objFilt_facCond_ts_ev])

    # Set times
    times = np.arange(-1, 2.501, .001)

    # Set indeces
    indeces = ["facFilt/facCond", "facFilt/objCond", "objFilt/objCond", "objFilt/facCond"]

    return data, times, indeces


def mean_ci_array(data, sbj_axis=1, zscore=False, design="within"):
    print("\nAveraging the data and computing CIs...")
    # Average across participants
    data_m = np.mean(data, axis=sbj_axis)

    # Z-score data by group
    if zscore:
        data_m = stats.zscore(data_m, axis=-1)
        data_ci = np.zeros(data_m.shape) #fake values
    else:
        # Compute condidence intervals
        if design == "within": # within-subject design CIs from Cousineau (2005)
            sbj_m = np.mean(data, axis=0, keepdims=True)
            grp_m = np.mean(data, axis=sbj_axis, keepdims=True)
            new_sbj = data - sbj_m + grp_m
            data_ci = stats.sem(new_sbj, axis=sbj_axis) * 1.96
        elif design == "between":
            n = data.shape[sbj_axis]
            data_ci = stats.sem(data, axis=sbj_axis) * stats.t.ppf((1 + .95) / 2., n - 1)

    return data_m , data_ci


def lowpass_filter(data, axis=2):
    print("\nLow-pass filtering the data...")
    # Low-pass filter the data
    order = 6
    fs = 1000.0  # sample rate (Hz)
    cutoff = 30.0
    b, a = ss.butter(order,
                        cutoff,
                        fs=fs,
                        btype='low',
                        analog=False)

    data = ss.lfilter(b, a, data, axis=axis)

    return data


def rms(data):
    print("\nComputing root mean square...")
    # Compute root mean square
    data = np.sqrt((np.array(data)**2))

    return data


def baseline_corr(data, times, axis=1, baseline_win=[-.5, 0]):
    print("\nBaseline-correcting the data...")
    # Set parameters correction
    imin = (np.abs(times - baseline_win[0])).argmin()
    imax = (np.abs(times - baseline_win[1])).argmin()

    # Apply correction
    mean = np.mean(data[..., imin:imax], axis=axis, keepdims=True)
    data -= mean

    return data


def plotting_ged_timeseries(vlines = [0, 0.5, 1.0, 1.5]):
    # Load data
    data, times, indeces = load_ged_data(label_name="fusifor")

    # Low-pass filter the data
    data = lowpass_filter(data, axis=2)

    # Compute root mean square
    data = rms(data)

    # Compute group mean and CIs
    data_m , data_ci = mean_ci_array(data, sbj_axis=1, zscore=False, design="within")

    # Aplly baseline correction
    data_m = baseline_corr(data_m, times)

    # Set plotting params
    colors_cond = {
        "facCond": [215/255, 27/255, 30/255],
        "objCond": [253/ 255, 175/255, 99/255],
    }
    conditions = [i[-7:] for i in indeces]
    folder = op.join(
        save_root, "ged", "fusifor")
    if not op.exists(folder):
        os.makedirs(folder)

    # Loop over filters
    for i, filt in enumerate(["facesel", "objsel"]):
        filename = op.join(folder,
                        f"ged-sub-{sub}-_desc_{filt}_timecourse.png")
        colors = [colors_cond[cond] for cond in conditions[0+i*2:2+i*2]]

        # Plot
        plot_time_series(
            data_m[0+i*2:2+i*2,...], t0=times[0], tend=times[-1], xlim=[-.5,2.], #ylim=[0.,1.6],
            err=data_ci[0+i*2:2+i*2,...], colors=colors, vlines=vlines, filename=filename,
            conditions=conditions[0+i*2:2+i*2], xlabel="Time (s)", ylabel="ERF (RMS)",
            err_transparency=0.2, title=None, square_fig=False, do_legend=False,
            patches=None, patch_color="r", patch_transparency=0.2)


def plotting_dfc_matrix():
    print("\nPlotting DFC...")
    # Loop over conditions to contrast
    for conds in [['face', 'object']]:
        print(f"Analysis: {conds[0]} vs {conds[1]}")

        # Load data
        data_1, times, freqs, indices_comb = load_dfc_data(
            sub, conds[0])

        data_2, _, _, _ = load_dfc_data(
            sub, conds[1])

        # Load mask
        desc = f"dfc_{conds}"
        masks = load_conn_mask(
            sub, desc,
            data_1, indices_comb,
            con_method="dfc")

    # Compute differce
    data_dif = data_1 - data_2

    # Set plotting params
    labels = ['pfc','v1v2','face','object']

    # Loop over node couples
    for i in indices_comb:
        print(f'Plotting {labels[i[0]]}-{labels[i[1]]}...')

        # Plot each conditions separately
        vmin = -4
        vmax = 4

        for j, data in enumerate([data_1, data_2]):
            # Zscore data
            data = stats.zscore(data[indices_comb.index(i),...], axis=1)

            # Set output path
            folder = op.join(
                save_root, "dfc", t_rel+no_evo)
            if not op.exists(folder):
                os.makedirs(folder)
            filename = op.join(
                folder,
                f"conn-sub-{sub}-desc-{labels[i[0]]}-{labels[i[1]]}_c-{conds[j]}_dfc{no_evo}.png")

            # Plot and save
            plot_pcolormesh(data,
                            times, freqs,
                            mask=None, transparency=1.,
                            vlim=[vmin, vmax], xlabel="Time (s)", ylabel="Frequency (Hz)",
                            cbar_label="DFC", filename=filename)

        # Plot difference between conditions
        vmin = -.1
        vmax = .1

        # Set output path
        folder = op.join(
            save_root, "dfc", t_rel+no_evo)
        if not op.exists(folder):
            os.makedirs(folder)
        filename = op.join(
            folder,
            f"conn-sub-{sub}-desc-{labels[i[0]]}-{labels[i[1]]}_{conds[0]}-{conds[1]}_dfc{no_evo}.png")

        # Plot and save
        plot_pcolormesh(data_dif[indices_comb.index(i),...],
                        times, freqs,
                        mask=masks[indices_comb.index(i),...], transparency=1.,
                        vlim=[vmin, vmax], xlabel="Time (s)", ylabel="Frequency (Hz)",
                        cbar_label="ΔDFC", filename=filename)


if __name__ == "__main__":
    plotting_ppc_matrix()
    plotting_dfc_matrix()
    plotting_ged_timeseries()
