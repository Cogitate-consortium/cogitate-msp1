""" This script generates data set with all the theories predictions baked in
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
    contributors: Simon Henin
    Simon.Henin@nyulangone.org
"""

import mne
import re
import os
import glob
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
from mne_bids import (write_raw_bids, BIDSPath)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nibabel.freesurfer.io import read_geometry, read_annot
from general_helper_functions.data_general_utilities import moving_average
from rsa.rsa_helper_functions import within_vs_between_cross_temp_rsa
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
colors = sns.color_palette("colorblind")
duration_mrk = {"500ms": ":", "1000ms": "-.", "1500ms": "--"}
cate_colors = {"face": colors[0], "object": colors[1]}


def plot_channel_groups(epochs, subjects_dir, subject="fsaverage", save_path=None):
    """

    :param epochs:
    :param subjects_dir:
    :param subject:
    :param save_path:
    :return:
    """

    # Get the different groups:
    channel_groups = list(set(re.sub(r'\d+', '', channel) for channel in epochs.ch_names))

    # Looping through each group:
    for group in channel_groups:
        group_channels = [channel for channel in epochs.ch_names if group in channel]
        group_epochs = epochs.copy().pick(group_channels)
        group_montage = group_epochs.get_montage()
        group_montage.add_estimated_fiducials("fsaverage", subjects_dir)
        trans = mne.channels.compute_native_head_t(group_montage)
        group_epochs.set_montage(group_montage)
        from mne.viz import plot_alignment, snapshot_brain_montage
        fig = plot_alignment(group_epochs.info, trans=trans,
                             subject='fsaverage', subjects_dir=subjects_dir,
                             surfaces=['pial'], coord_frame='head')
        mne.viz.set_3d_view(fig, azimuth=180, elevation=70)
        xy, im = snapshot_brain_montage(fig, group_epochs.info, hide_sensors=False)
        fig_2, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(im)
        ax.set_axis_off()
        plt.savefig("{} coverage.png".format(group))
        plt.close()
        mne.viz.close_3d_figure(fig)
    return None


def sine_wave_fun(amp, freq, phase, time):
    """
    This function generates a sine wave according to the parameters
    :param amp: (float) amplitude of the sine wave
    :param freq: (float) frequency of the sine wave
    :param phase: (float) phase of the sine wave
    :param time: (nd array) time over which to compute the sine wave
    :return:
    """
    return amp * np.sin(2 * np.pi * freq * time + phase)


def sim_channel_loc(roi, rois_dict, fs_path, ch_names):
    """
    This function generates random channels location within a given ROI
    :param roi: (string) name of the ROI corresponding to the ROI dict
    :param rois_dict: (dict) contain for  each rois a list of labels that corresponds to it
    :param fs_path: (string) path to the freesurfer subject to use to get the loc of the ROIs
    :param ch_names: (list of strings) list of the different channels for which to generate a random location
    :return:
    """
    # Get the labels of each vertex:
    annot = read_annot(Path(fs_path, "label", "lh.aparc.a2009s.annot"))
    # Get the coordinates of each vertex:
    vert_coords = read_geometry(Path(fs_path, "surf", "lh.pial"))[0]
    # Reformat the annot to be a little easier to handle:
    vertices_labels = np.array(["ctx_lh_" + annot[2][label_ind].decode("utf-8") for label_ind in annot[0]])
    annot_labels = ["ctx_lh_" + annot_label.decode("utf-8") for annot_label in annot[2]]
    # Fetch the labels for this ROI:
    roi_labels = rois_dict[roi]
    # Get indices of all vertices that are within the ROI:
    vertices_ind = [ind for ind, label in enumerate(vertices_labels) if label in roi_labels]
    # Randomly picking among these vertices indices:
    channels_coord = vert_coords[np.random.choice(vertices_ind, len(ch_names))]
    # Create the
    ch_loc_dict = {names: np.array([channels_coord[ind, 0] * 0.001, channels_coord[ind, 1] * 0.001,
                                    channels_coord[ind, 2] * 0.001])
                   for ind, names in enumerate(ch_names)}
    return ch_loc_dict


def sim_trial_data(pattern_vect, times, amp, freq, amp_offset, rise_time_samp, noise_loc, noise_factor,
                   rand_phase=False):
    """
    This function generates single trials activation according to the passed parameters
    :param pattern_vect: (nd array) for each time point, whether there should be anything (activation, MVP...)
    :param times: (nd array) times axis of the trial
    :param amp: (float) amplitude of the sine wave
    :param freq: (float) frequency of the sine wave
    :param amp_offset: (float) offset of the sine wave
    :param rise_time_samp: (float) how long it should last to transition from no patterns to pattern. To avoid transient
    :param noise_loc: (float) mean of the noise to add to the data
    :param noise_factor: (float) noise standard deviation
    :param rand_phase: (bool) whether or not to randomize the phase
    :return: np array: activation in the trial to simulate
    """
    # Check whether the patterns is supposed to be on or not:
    if rand_phase:
        phase = np.random.uniform(low=-5, high=5, size=1)
    else:
        phase = 0
    trial_data = np.multiply(sine_wave_fun(amp, freq, phase, times) + amp_offset, pattern_vect)
    # Looping through the patterns onset:
    patterns_onset = np.where(np.abs(np.diff(pattern_vect)) == 1)[0] + 1
    # Adding 0 and last:
    patterns_onset = np.insert(patterns_onset, 0, 0, axis=0)
    patterns_onset = np.append(patterns_onset, pattern_vect.shape[0] + 1)
    # Add a rise and drop at onset and offset:
    # Loop through each time point:
    for ind, onset in enumerate(patterns_onset):
        if ind == patterns_onset.shape[0] - 1:
            continue
        else:
            offset = patterns_onset[ind + 1] - 1
        if offset + 1 > trial_data.shape[0]:
            continue
        trial_data[offset - rise_time_samp + 1:offset + 1] = \
            np.linspace(trial_data[offset - rise_time_samp],
                        trial_data[offset + 1],
                        num=rise_time_samp,
                        endpoint=False)
    # Convert the noise scale to be in data units:
    noise_scale = noise_factor * np.std(trial_data)
    # Add noise:
    return np.add(trial_data, np.random.normal(loc=noise_loc,
                                               scale=noise_scale ** 2,
                                               size=trial_data.shape[0]))


def simulate_subjects(sub_list, plot_results=False, config_file="sim_config_highgamma.json",
                      trials_param="trials_param_category.csv"):
    # Load the simulation parameters files:
    simulation_parameters = pd.read_csv(Path(os.getcwd(), trials_param))
    # false are weirdly converted to FALSE, this puts it back into lower case as expected!
    simulation_parameters["Category"] = simulation_parameters["Category"].str.lower()
    simulation_parameters["identity"] = simulation_parameters["Category"].str.lower()
    config = Path(os.getcwd(), "configs", config_file)
    with open(config) as f:
        param = json.load(f)

    # Set up parameters for channels loc:
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = Path(sample_path, 'subjects')
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # Downloading the data if needed
    fs_dir = Path(subjects_dir, "fsaverage")

    # ===================================================================
    # Generate additional parameters:
    # Time
    n_samples = int((param["tmax"] - param["t0"]) * param["sfreq"])
    times = np.linspace(param["t0"], param["tmax"], n_samples, endpoint=True)
    # Convert the rise time to samples:
    rise_time_samp = int(param["rise_time_s"] * param["sfreq"])
    # Events and metadata:
    # Create unique condition pairs:
    cond_combinations = \
        list(zip(
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "event type"].to_list(),
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "Category"].to_list(),
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "Duration"].to_list(),
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "task relevance"].to_list(),
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "identity"].to_list(),
            simulation_parameters[
                ["event type", 'Category', 'Duration', "task relevance", "identity", "orientation"]].drop_duplicates()[
                "orientation"].to_list()))
    # Generate the metadata:
    metadata = pd.DataFrame(columns=param["metadata_col"])
    events_dict = {}
    time_ctr = 0
    for ind, pair in enumerate(cond_combinations):
        metadata = metadata.append(pd.DataFrame({
            param["metadata_col"][0]: [pair[0]] * param["n_trials"],
            param["metadata_col"][1]: [pair[1]] * param["n_trials"],
            param["metadata_col"][2]: [pair[2]] * param["n_trials"],
            param["metadata_col"][3]: [pair[3]] * param["n_trials"],
            param["metadata_col"][4]: [pair[4]] * param["n_trials"],
            param["metadata_col"][5]: [pair[5]] * param["n_trials"],
            param["metadata_col"][6]: [ind] * param["n_trials"],
            param["metadata_col"][7]: (np.linspace(time_ctr, time_ctr +
                                                   (param["tmax"] - param["t0"]) * param["n_trials"],
                                                   param["n_trials"], endpoint=False) + np.abs(param["t0"]))
        }))
        events_dict["/".join([pair[0], pair[1], pair[2], pair[3], pair[4], pair[5]])] = ind
        time_ctr = metadata[param["metadata_col"][7]].to_list()[-1] + (param["tmax"] - param["t0"])
    metadata = metadata.reset_index(drop=True)
    # Double the number of trials for center orientation:
    center_meta_data = metadata.loc[metadata["orientation"] == "Center"]
    center_meta_data["onset"] = center_meta_data["onset"] + metadata["onset"].to_list()[-1]
    metadata = pd.concat([metadata, center_meta_data]).reset_index(drop=True)
    # Generate the channels names:
    ch_names = []
    for group in list(simulation_parameters["Group"].unique()):
        ch_names.extend(["{}_ch-{}".format(group, i) for i in range(param["n_channels"])])

    info = mne.create_info(ch_names,
                           param["sfreq"], ch_types=["ecog"] * len(ch_names))
    info['description'] = 'Simulated data'
    events = np.column_stack(((metadata[param["metadata_col"][7]].to_numpy() * param["sfreq"]).astype(int),
                              np.zeros(metadata[param["metadata_col"][7]].to_numpy().shape[0], dtype=int),
                              metadata[param["metadata_col"][6]].to_numpy().astype(int)))

    # ======================================================================================================================
    # Simulating the data:
    for sub_id in sub_list:
        # Generate n sequences of random numbers for the groups that need it:
        n_sequences = len(list(simulation_parameters["Amplitude_sequence"].unique()))
        offset_seq = [np.random.normal(loc=param["activation_offset"],
                                       scale=param["patterns_std"],
                                       size=param["n_channels"]) for i in range(n_sequences)]
        data = np.zeros([len(metadata), len(ch_names), times.shape[0]])
        channels_loc = {}
        # Looping through each group:
        for group in list(simulation_parameters["Group"].unique()):
            print("=" * 40)
            print("Generating {} data".format(group))
            group_param = simulation_parameters.loc[simulation_parameters["Group"] == group]
            # Generate the name of the channels of this group:
            group_channels = ["{}_ch-{}".format(group, i) for i in range(param["n_channels"])]
            # Generating a dict of channels locs:
            ch_loc = sim_channel_loc(group_param["ROI"].to_list()[0], param["rois"], fs_dir, group_channels)
            channels_loc.update(ch_loc)

            # Looping  through each channel in this group:
            for ch_num in range(param["n_channels"]):
                channel_name = "{}_ch-{}".format(group, ch_num)
                channel_ind = np.where(np.array(ch_names) == channel_name)[0]
                # Looping through each trial:
                for trial_ind, trial_meta in metadata.iterrows():
                    # Extract the parameters for this specific trial and group:
                    ch_trial_param = group_param.loc[(group_param["event type"] == trial_meta["event type"]) &
                                                     (group_param["Category"] == trial_meta["category"]) &
                                                     (group_param["Duration"] == trial_meta["duration"]) &
                                                     (group_param["task relevance"] == trial_meta["task_relevance"]) &
                                                     (group_param["identity"] == trial_meta["identity"]) &
                                                     (group_param["orientation"] == trial_meta["orientation"])]
                    # Extract the relevant time window for this trial and parse it to convert to actual times:
                    time_wins = [[float(val) for val in time_pairs.split("-")]
                                 for time_pairs in ch_trial_param["MVP times"].item().replace("[", "").replace("]", "").
                                     split(";")]
                    # The config only specifies where the patterns are supposed to be on, not off. We need to infer
                    # all the off times to feel them with data:
                    pattern_vect = np.zeros(times.shape[0])
                    for time_win in time_wins:
                        onset = np.where(times >= time_win[0])[0][0]
                        try:
                            offset = np.where(times > time_win[1])[0][0]
                        except IndexError:
                            offset = times.shape[-1]
                        pattern_vect[onset:offset] = 1
                    # Parsing the config to get the different inputs to generate the different param:
                    rand_phase = True if ch_trial_param["Random phase"].item() == "Yes" else False
                    sine_amp = param["sine_amp_mean"] \
                        if ch_trial_param["Amplitude_sequence"].item() != "None" \
                        else param["sine_amp_mean"] * float(ch_trial_param["Activation"].item())
                    sine_offset = offset_seq[int(ch_trial_param["Amplitude_sequence"].item())][ch_num] \
                        if ch_trial_param["Amplitude_sequence"].item() != "None" \
                        else np.random.normal(loc=param["activation_offset"] *
                                                  float(ch_trial_param["Activation"].item()),
                                              scale=param["selectivity_std"], size=1)
                    # Generate the data:
                    trial_data = sim_trial_data(pattern_vect, times, sine_amp, param["sine_freq"], sine_offset,
                                                rise_time_samp, 0, 0,
                                                rand_phase=rand_phase)
                    data[trial_ind, channel_ind, :] = trial_data + param["baseline_offset"]
        # Add the noise to the data:
        noise_scale = param["noise_factor"] * np.std(data)
        data = np.add(data, np.random.normal(loc=param["noise_mean"],
                                             scale=noise_scale ** 2,
                                             size=data.shape))
        # Convert the data to an mne epochs objects:
        epochs = mne.EpochsArray(data, info, events=events, event_id=events_dict, tmin=param["t0"])
        epochs.metadata = metadata
        # Add the montage:
        montage = mne.channels.make_dig_montage(channels_loc, coord_frame='mri')
        epochs.set_montage(montage, on_missing='warn')
        # Plot the different electrodes groups:
        plot_channel_groups(epochs, subject="fsaverage", subjects_dir=subjects_dir, save_path=None)

        # Also, convert the data to a raw object:
        sim_raw = mne.io.RawArray(data.reshape([data.shape[1], data.shape[0] * data.shape[-1]]), info)
        # Creating a temporary directory
        temp_dir = tempfile.mkdtemp()
        raw_temp = Path(temp_dir, "_raw.fif")
        # Saving the data in that temp dir:
        sim_raw.save(raw_temp)
        # Deleting and reloading the raw:
        del sim_raw
        sim_raw = mne.io.read_raw(raw_temp, preload=False)
        # ======================================================================================================================
        # Save the data:
        print("=" * 40)
        print("Saving the results")
        save_root = Path(param["BIDS_root"], "derivatives", "preprocessing", "sub-" + sub_id, "ses-" + param["session"],
                         "ieeg", "epoching", param["signal"], param["preprocess_steps"])
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        fname = "sub-{}_ses-{}_task-{}_desc-epoching_ieeg-epo.fif".format(sub_id, param["session"], param["task_name"])
        epochs.save(Path(save_root, fname), overwrite=True)
        # Save the channels coordinates to a tsv:

        # Copy the fsaverage subject to the subject freesurfer dir:
        save_root = Path(param["BIDS_root"], "derivatives", "fs", "sub-" + sub_id)
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        # Copy fsaverage here:
        shutil.copytree(fs_dir, save_root, dirs_exist_ok=True)

        # Creating the BIDS path:
        bids_path = BIDSPath(subject=sub_id, session=param["session"],
                             task=param["task_name"], datatype="ieeg", root=param["BIDS_root"])
        try:
            write_raw_bids(sim_raw, bids_path,
                           overwrite=True, format='auto')
        except OSError:
            shutil.rmtree(Path(param["BIDS_root"], "sub-" + sub_id))
            write_raw_bids(sim_raw, bids_path,
                           overwrite=True, format='auto')

        # Cleaning up the temp buffer:
        files = glob.glob(temp_dir + os.sep + '*')
        for f in files:
            os.remove(f)
        os.rmdir(temp_dir)
        # Save the electrodes loc to file:
        channels_loc_df = pd.DataFrame.from_dict(channels_loc, orient="index").reset_index()
        # Rename the columns:
        channels_loc_df.columns = ["name", "x", "y", "z"]
        electodes_tsv_file = Path(bids_path.directory,
                                  "sub-{}_ses-{}_space-Other_electrodes.tsv".format(sub_id, param["session"]))
        channels_loc_df.to_csv(electodes_tsv_file, sep="\t", index=False)
        electodes_tsv_file = Path(bids_path.directory,
                                  "sub-{}_ses-{}_space-fsaverage_electrodes.tsv".format(sub_id, param["session"]))
        channels_loc_df.to_csv(electodes_tsv_file, sep="\t", index=False)

    # Loop through each electrode groups to plot them:
    if plot_results:
        print("=" * 40)
        print("Plotting the results")
        for ind, group in enumerate(list(simulation_parameters["Group"].unique())):
            fig, axs = plt.subplots(figsize=[14, 9])
            # Generate the channels name for this group:
            ch_names = ["{}_ch-{}".format(group, i) for i in range(param["n_channels"])]
            # Set the title of the plot:
            axs.set_title("{} channels, n={}, freq={}Hz".format(group, param["n_channels"], param["sine_freq"]))
            axs.set_ylabel("Gain (a.u.)")
            axs.set_xlabel("Time (s)")
            # Looping through the conditions:
            for cond_ind, cond in enumerate(["face/1500ms", "object/1500ms"]):
                # Get the data
                cond_epochs = epochs.copy()[cond]
                evk = cond_epochs.average(ch_names)
                # Split the cond to generate the line styles and colors:
                cond_split = cond.split("/")
                # Add the legend:
                axs.plot(epochs.times[0], evk.data.T[0, 0], label=cond, c=cate_colors[cond_split[0]],
                         linestyle=duration_mrk[cond_split[1]])
                axs.plot(epochs.times, evk.data.T, c=cate_colors[cond_split[0]],
                         linestyle=duration_mrk[cond_split[1]])
                axs.set_xlim([epochs.times[0], epochs.times[-1]])
            axs.legend()
            plt.savefig("noise free {} channels.png".format(group))
            plt.close()
            # Try the cross temporal generalization RSA:
            for cond_ind, cond in enumerate(list(metadata["duration"].unique())):
                fig, axs = plt.subplots(figsize=[14, 9])
                # Generate the channels name for this group:
                cond_epochs = epochs.copy()[cond]
                data = cond_epochs.get_data(picks=ch_names)
                n_samples = int(np.floor(20 * param["sfreq"] / 1000))
                data = moving_average(data, n_samples, axis=-1, overlapping=False)
                labels = cond_epochs.metadata["category"].to_numpy()
                cross_temporal_mat, sample_rdm = within_vs_between_cross_temp_rsa(data,
                                                                                  labels,
                                                                                  metric="correlation",
                                                                                  zscore=False,
                                                                                  groups=None,
                                                                                  regress_groups=False,
                                                                                  between_within_group=False,
                                                                                  onset_offset=[epochs.times[0],
                                                                                                epochs.times[-1]],
                                                                                  sample_rdm_times=[0.3, 0.5],
                                                                                  n_features=None,
                                                                                  n_folds=None,
                                                                                  shuffle_labels=False,
                                                                                  verbose=False)

                im = axs.imshow(cross_temporal_mat, origin="lower",
                                extent=[epochs.times[0], epochs.times[-1], epochs.times[0], epochs.times[-1]],
                                aspect='equal', vmin=-2, vmax=2, cmap="RdYlBu_r")
                axs.set_title("{} RSA, n={}, freq={}Hz".format(group, param["n_channels"], param["sine_freq"]))
                plt.colorbar(im)
                plt.savefig("RSA_{}_group{}_channels.png".format("_".join(cond.split("/")), group))
                plt.close()
                fig, axs = plt.subplots(figsize=[14, 9])
                im = axs.imshow(sample_rdm, origin="lower",
                                aspect='equal', vmin=0, vmax=2, cmap="RdYlBu_r")
                axs.set_title("{} RDM, n={}, freq={}Hz".format(group, param["n_channels"], param["sine_freq"]))
                axs.set_ylabel("First presentation")
                axs.set_xlabel("Second presentation")
                plt.colorbar(im)
                plt.savefig("RDM_{}_group{}_channels.png".format("_".join(cond.split("/")), group))
                plt.close()


if __name__ == "__main__":
    simulate_subjects(["sim1", "sim2", "sim3"], plot_results=True, config_file="sim_config_highgamma.json",
                      trials_param="trials_param_category.csv")
