"""Utils for trigger alignment."""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import mne


def detect_triggers(data_preparation_parameters, subject_info, raw):
    """
    Loading the edf files:
    :param data_preparation_parameters: object containing the analysis parameters
    :param subject_info: object containing info specific to this participant
    :param raw: (mne raw object) raw data object
    :return pd_onsets: (dict of np arrays) contain the photodiode events timestamps in seconds and in sample number
    :return pd_onsets_clean: (dict of np arrays) contain the photodiode events timestamps in seconds and in sample
    number but cleaned of unnecessary input, ready for log files alignment
    :return pd_signal: (dict of np arrays or lists) contain the photodiode amplitude and times
    :return srate: (int or float) signal sampling rate
    :return raw: (mne raw object) raw data object
    """

    # Extract photodiode signal
    # If a reference channel was used:
    if subject_info.TRIGGER_REF_CHANNEL != '':
        # Take the photodiode time and the amplitude of the signal by subtracting the amplitude in the reference channel
        # to the trigger channel
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
                   - np.array([raw.get_data(picks=subject_info.TRIGGER_REF_CHANNEL)[0]])
        })
    else:  # Otherwise, just extract the photodiode channel:
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
        })

    # Clean trigger signal from calibration or other trigger noise
    pd_signal_no_noise = clean_pd_signal_from_noise(pd_signal, subject_info)
    pd_signal = pd_signal_no_noise

    # Get sampling rate
    srate = raw.info['sfreq']

    # Getting the PD onsets:
    pd_onsets = find_onset_pd(pd_signal, subject_info,
                              data_preparation_parameters, srate)

    # Cleaning the photodiode signal:
    pd_onsets_clean = clean_pd_onsets(pd_onsets, data_preparation_parameters)

    return pd_onsets, pd_onsets_clean, pd_signal, srate, raw


def clean_pd_signal_from_noise(pd_signal, subject_info):
    """
    Cleaning the pd signal from the calibration screen light and other signal noise.
    When using Eyelink tracker, during the calibration, the background turns gray, and therefore the PD signal needs
    to be adjusted,
    :param: pd_signal: (Dictionary of np arrays) The raw photodiode signal and timestamps.
    :param: subject_info: (class subjectInfo) custom object containing information about the subject
    :return pd_signal: (Dictionary of np arrays) photodiode signal with removed noisy chunks
    """

    # If intervals are already given
    if len(subject_info.start_inds_trigger_noise) > 0:

        for ind_start_noise, ind_end_noise in zip(subject_info.start_inds_trigger_noise,
                                                  subject_info.end_inds_trigger_noise):
            # Set the signal to "zero":
            pd_signal['Amp'][0][ind_start_noise:ind_end_noise] = [(pd_signal['Amp'][0]).min()] * (
                    ind_end_noise - ind_start_noise)

    # Otherwise request from user:
    choose_intervals = input("\nThe trigger signal may be cleaned by selecting sections of the signal for removal. "
                             "\nIf your experiment was restarted, it is a good idea to remove the following triggers"
                             "\n which are send in the restarting process: "
                             "\n 1. two experiment start/end triggers (4 consecutive triggers each)"
                             "\n 2. block start (4 consecutive triggers)"
                             "\n\n Do you wish to clean the trigger signal from noise? [Yes or No]")

    if choose_intervals == 'Yes' or choose_intervals == 'yes':
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        plt.plot(pd_signal['Amp'][0], 'b')
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')

        while choose_intervals == 'Yes' or choose_intervals == 'yes':

            plt.close()

            fig, ax = plt.subplots()
            fig.set_figwidth(15)
            fig.set_figheight(10)
            plt.plot(pd_signal['Amp'][0], 'b')
            plt.xlabel('Sample nr')
            plt.ylabel('Amplitude')
            plt.title('Raw photodiode signal. '
                      + '\n 1. Zoom in on the interval you want to zero.'
                      + '\n 2. Deselect all buttons from the lower left button bar.'
                      + '\n 3. You can now click and drag a rectangle around the x axis interval enclosing the specific'
                        ' interval'
                      + '\n 4. When you are happy with the interval, close this plot to choose more intervals or '
                        'proceed.')

            ind_start_noise, ind_end_noise = select_start_and_end(
                fig, ax, 'box')
            print('You have chosen the start index to be ', ind_start_noise)
            print('and the end index to be ', ind_end_noise)
            print('All values in this interval will be set to zero.')
            print(
                'If you are happy with these values, close the figure and you can enter the next interval.')

            if ind_start_noise < 0:
                ind_start_noise = 0
            if ind_end_noise > len(pd_signal['Amp'][0]):
                ind_end_noise = len(pd_signal['Amp'][0])

            subject_info.start_inds_trigger_noise.append(ind_start_noise)
            subject_info.end_inds_trigger_noise.append(ind_end_noise)

            # Set the signal to "zero"

            pd_signal['Amp'][0][ind_start_noise:ind_end_noise] = [(pd_signal['Amp'][0]).min()] * (
                    ind_end_noise - ind_start_noise)

            print('Done. Please inspect.')
            fig = plt.figure()
            sig = pd_signal['Amp'][0]
            plt.plot(range(len(sig)), sig)
            plt.show()
            choose_intervals = input('More to remove? [Yes or No]')

        # Finally, updating the json with that info:
        print("Updating the subject info json file")
        subject_info.update_json()

    return pd_signal


def toggle_selector(event):
    print('Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def select_start_and_end(fig, ax, draw_type='line'):
    """
    Shows the plot in interactive mode.
    The user can click and draw to show where the file should start and end.
    The file end (start) will be taken as the x coordinate of the end point
    of the (next to) last line that was drawn by the user.
    When the user is happy, she should close the plot to proceed.
    :param: fig
    :param: ax
    :param: draw_type (str): 'line' for choosing start and end point separately (can zoom out in between)
            so best if far from each other or 'box' for choosing them close to one another.
    """

    x_ends = []

    # Define the interaction with the plot
    if draw_type == 'line':
        def onselect(eclick, erelease):
            "eclick and erelease are matplotlib events at press and release."
            print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
            print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
            x_ends.append(erelease.xdata)

        # toggle_selector.RS = plt.RectangleSelector(ax, onselect, drawtype='line')
        toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='line')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()

    elif draw_type == 'box':
        def onselect(eclick, erelease):
            "eclick and erelease are matplotlib events at press and release."
            print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
            print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
            x_ends.append(eclick.xdata)
            x_ends.append(erelease.xdata)

        # toggle_selector.RS = plt.RectangleSelector(ax, onselect, drawtype='line')
        toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()

    return int(x_ends[-2]), int(x_ends[-1])


def find_onset_pd(pd_raw, subject_info, data_preparation_parameters, srate):
    """
    This function detects the onset of the photodiode triggers
    :param pd_raw: dictionary of np array: time: np arraylen(datapoints) of PD timestamp, Amp: len(datapoints)
    np array of PD amplitude
    :param subject_info: object containing information specific to a subject
    :param data_preparation_parameters: (DataPreparation object) custom made object containing info about data
    preparation
    :param srate: (int) sampling rate of the signal
    :return: PD_onsets: pandas series containing the timestamps of the detect photodiode onsets
    """

    # Binarizing the photodiode signal:
    binary_pd = (pd_raw["Amp"] > subject_info.PD_THRESHOLD).astype(
        int)  # np array dim: len(datapoints)

    # Finding the onset of the photodiode peaks:
    pd_onsets_ind = np.where(np.diff(binary_pd) == 1)[1]

    # Make sure that you don't get false positives from noisy signal so that the same signal triggers the threshold
    # twice one photodiode signal is 3 ref rates ms and there are srate samples per second
    pd_onsets_ind_no_fp = pd_onsets_ind
    minimum_ind_diff = int(
        (data_preparation_parameters.ref_rate_ms * 4.5 * srate) / 1000)
    indices_to_remove = []
    for i in range(1, len(pd_onsets_ind)):
        onset_ind = pd_onsets_ind[i]
        prev_onset_ind = pd_onsets_ind[i - 1]
        if (onset_ind - prev_onset_ind) < minimum_ind_diff:
            indices_to_remove.append(i)

    pd_onsets_ind_no_fp = np.delete(pd_onsets_ind_no_fp, indices_to_remove)

    # Using the pd_onsets_ind, getting the time of the onsets of photodiode from pd_raw.time
    pd_onsets_time = pd_raw["time"][0, pd_onsets_ind_no_fp]

    # Creating a dictionary of np array: "sample" is the sample corresponding to the onset, pd_onsets_time is the time
    pd_onsets = {"Sample_num": pd_onsets_ind_no_fp, "Time": pd_onsets_time}

    # Computing the size of the bars for the plot:
    min_amp = subject_info.PD_THRESHOLD - 0.5 * subject_info.PD_THRESHOLD
    max_amp = subject_info.PD_THRESHOLD + 0.5 * subject_info.PD_THRESHOLD

    # Get the path where to save data
    save_path = Path(subject_info.participant_save_root, "info", "figure")

    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    # Plot raw signal with threshold and onsets (zoom in and check if they make sense)
    plt.figure(figsize=(8, 6))
    plt.plot(pd_raw["Amp"][0], 'g')
    plt.vlines(pd_onsets["Sample_num"], min_amp,
               max_amp, 'r', zorder=10, linewidth=2)
    plt.title('Raw photodiode signal with thresholds and onset.')
    plt.xlabel('Sample nr')
    plt.ylabel('Amplitude')

    if data_preparation_parameters.show_check_plots:
        plt.show()
        manual_remove_or_add_trigger_detections = \
            input(
                'Do you want to add or remove trigger onset detections? [Yes or No]')
        if manual_remove_or_add_trigger_detections == 'Yes' or manual_remove_or_add_trigger_detections == 'yes':
            pd_onsets = manually_add_or_remove_trigger_detections(
                pd_raw, pd_onsets, srate, subject_info)

        # Plot it again so that the correct one is saved.
        plt.plot(pd_raw["Amp"][0], 'g')
        plt.vlines(pd_onsets["Sample_num"], min_amp,
                   max_amp, 'r', zorder=10, linewidth=2)
        plt.title('Raw photodiode signal with thresholds and onset.')
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')

    plt.savefig(os.path.join(save_path, "Raw_photodiode_signal.png"))
    plt.close()
    # Dumping the subject info and the analysis parameters with the figures:
    subject_info.save(save_path)
    data_preparation_parameters.save(save_path, subject_info.files_prefix)

    return pd_onsets


def clean_pd_onsets(pd_onsets, data_preparation_parameters):
    """
    This function removes everything that is not needed from the PD signal: block onsets, block offsets...
    :param pd_onsets: dictionary of np array: Sample: arraylen(datapoints) of PD onset sample, Time: np len(datapoints)
    PD onsets timestamp
    :param data_preparation_parameters: custom made object containing information about the analysis
    :return: pd_onsets_clean: photodiode signal cleaned by removing the entry that are not needed for further analysis
    """

    # Converting the screen refresh rate from ms to sec:
    ref_rate_sec = data_preparation_parameters.ref_rate_ms * 0.001

    # First, computing the diff on the photodiode timestamps:
    pd_onsets_diff = np.diff(pd_onsets["Time"])

    # Find triggers that are successive and categorize as block onsets/offsets (4 consecutive)
    # or experiment onset and offsets (3 consecutives)
    block_onsets_sample_nrs = []
    block_offsets_sample_nrs = []

    experiment_start_or_end_sample_nrs = []

    # i is the sample nr
    i = 0
    while i < len(pd_onsets_diff) - 2:
        if (pd_onsets_diff[i] <= ref_rate_sec * 3 * 2 + ref_rate_sec * 2) & (
                pd_onsets_diff[i] >= ref_rate_sec * 3 - 0.5 * ref_rate_sec):
            # Now we have found two successive triggers. See if we have 2 or 3 more
            if (pd_onsets_diff[i + 1] <= ref_rate_sec * 3 * 2 + ref_rate_sec * 2) & (
                    pd_onsets_diff[i + 1] >= ref_rate_sec * 3 - 0.5 * ref_rate_sec):
                # Now we have found three. Let's see if we have one more. But first, check if we are at the end
                if (pd_onsets_diff[i + 2] <= ref_rate_sec * 3 * 2 + ref_rate_sec * 2) & (
                        pd_onsets_diff[i + 2] >= ref_rate_sec * 3 - 0.5 * ref_rate_sec):
                    # Yes we have 4 consecutive. Add them to block onset
                    for j in range(i, i + 4):
                        block_onsets_sample_nrs.append(j)
                    i += 3
                else:
                    # No. Then we have 3 and are at experiment start or end
                    for j in range(i, i + 3):
                        experiment_start_or_end_sample_nrs.append(j)
                    i += 2
            else:
                # Then it is a block offset:
                for j in range(i, i + 2):
                    block_offsets_sample_nrs.append(j)
                i += 1
        i += 1

    # Did we find the experiment end yet?
    if len(experiment_start_or_end_sample_nrs) < 6:
        # Also evaluate the last 2 triggers which were not treated in the loop above
        if (pd_onsets_diff[-1] <= ref_rate_sec * 3 * 2 + ref_rate_sec * 2) & (
                pd_onsets_diff[-1] >= ref_rate_sec * 3 - 0.5 * ref_rate_sec) \
                & (pd_onsets_diff[-2] <= ref_rate_sec * 3 * 2 + ref_rate_sec * 2) & (
                pd_onsets_diff[-2] >= ref_rate_sec * 3 - 0.5 * ref_rate_sec):
            for j in [-3, -2, -1]:
                experiment_start_or_end_sample_nrs.append(j)

    # Plotting the results:
    plt.figure(figsize=(8, 6))
    plt.vlines(pd_onsets["Sample_num"], -1, 1)
    plt.vlines(pd_onsets["Sample_num"]
               [block_onsets_sample_nrs], -0.25, 0.75, colors="yellow")
    plt.vlines(pd_onsets["Sample_num"]
               [block_offsets_sample_nrs], -0.5, 0.5, colors="red")
    plt.vlines(pd_onsets["Sample_num"]
               [experiment_start_or_end_sample_nrs], -0.75, 0.25, colors="green")
    plt.title(
        'Blocking: green: experiment start, yellow: block start, red: block end')
    plt.grid()
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()

    # Then, removing everything before the first trig begin, because they are spurious:
    pd_onsets_clean = ({
        "Sample_num": np.delete(pd_onsets["Sample_num"], block_onsets_sample_nrs + block_offsets_sample_nrs
                                + experiment_start_or_end_sample_nrs),
        "Time": np.delete(pd_onsets["Time"],
                          block_onsets_sample_nrs + block_offsets_sample_nrs + experiment_start_or_end_sample_nrs)})

    # Getting the first block onset in samples and time. Using pd_onset here because in the clean one,
    # stuffs were removed and so the indices don't match anymore
    block_onset_sample = pd_onsets["Sample_num"][block_onsets_sample_nrs[0]]
    block_onset_time = pd_onsets["Time"][block_onsets_sample_nrs[0]]
    try:
        experiment_end_sample = pd_onsets["Sample_num"][experiment_start_or_end_sample_nrs[-1]]
        experiment_end_time = pd_onsets["Time"][experiment_start_or_end_sample_nrs[-1]]
        # Then, removing all photodiode triggers that occured before the first level start:
        pd_onsets_clean = {
            "Sample_num": [onsets for onsets in pd_onsets_clean["Sample_num"] if
                           block_onset_sample < onsets < experiment_end_sample],
            "Time": [onsets for onsets in pd_onsets_clean["Time"] if block_onset_time < onsets < experiment_end_time]
        }
    except IndexError:
        print("No experiment_start_or_end_sample_nrs found")

    # Plotting the results:
    # No triggers should appear where the colored lines are
    plt.figure(figsize=(8, 6))
    plt.vlines(pd_onsets_clean["Sample_num"], -1, 1, colors="black")
    plt.vlines(pd_onsets["Sample_num"]
               [block_onsets_sample_nrs], -0.25, 0.75, colors="yellow")
    plt.vlines(pd_onsets["Sample_num"]
               [block_offsets_sample_nrs], -0.5, 0.5, colors="red")

    plt.vlines(pd_onsets["Sample_num"]
               [experiment_start_or_end_sample_nrs], -0.75, 0.25, colors="green")

    plt.title(
        'Blocking after signal has been cleaned: No triggers should be visible at the places of the \n for the colored '
        'bars \n green: experiment start, yellow: block start, red: block end')
    plt.grid()
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    return pd_onsets_clean


def manually_add_or_remove_trigger_detections(pd_signal, pd_onsets_clean, srate, subject_info):
    """
    Manually add or remove trigger detections which were not or falsely detected by the automatic
    algorithm.
    :param: pd_signal: (dict of np arrays) the amplitude of the raw photodiode signal
    :param: pd_onsets_clean: (dict of np arrays) the detected onsets of triggers
    :param: srate: (int) sampling rate
    :param: subject_info: (SubjectInfo class object) custom object containing info about the subject
    :return: updated pd_onsets_clean
    """
    # Let the user add/remove triggers
    pd = pd_signal['Amp'][0]

    # Plot raw signal with threshold and onsets
    # Computing the size of the bars for the plot:
    min_amp = subject_info.PD_THRESHOLD - 0.5 * subject_info.PD_THRESHOLD
    max_amp = subject_info.PD_THRESHOLD + 0.5 * subject_info.PD_THRESHOLD

    user_input = input("Do you wish to ADD trigger detections? [Yes or No]")

    while user_input == 'yes' or user_input == 'Yes':
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        plt.plot(pd, 'g')
        plt.vlines(pd_onsets_clean["Sample_num"],
                   min_amp, max_amp, 'r', zorder=10, linewidth=2)
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')
        plt.title('Raw photodiode signal with thresholds and onset. '
                  + '\n 1. Zoom in on the trigger you want to add an onset.'
                  + '\n 2. Deselect all buttons from the lower left button bar.'
                  + '\n 3. You can now click and drag a rectacle around the x axis interval enclosing the specific '
                    'trigger, but only that one.'
                  + '\n 4. When you are happy with the interval, close this plot to choose more intervals or proceed.')
        start, end = select_start_and_end(fig, ax, 'box')
        pd_onsets_clean = add_trigger_onset(int(start), int(
            end), pd, pd_onsets_clean, srate, subject_info)
        user_input = input('More to add? [Yes or No]')

    user_input = input("Do you wish to REMOVE trigger detections? [Yes or No]")

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    plt.plot(pd, 'g')
    plt.vlines(pd_onsets_clean["Sample_num"], min_amp,
               max_amp, 'r', zorder=10, linewidth=2)
    plt.xlabel('Sample nr')
    plt.ylabel('Amplitude')

    while user_input == 'yes' or user_input == 'Yes':
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        plt.plot(pd, 'g')
        plt.vlines(pd_onsets_clean["Sample_num"],
                   min_amp, max_amp, 'r', zorder=10, linewidth=2)
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')
        plt.title('Raw photodiode signal with thresholds and onset. '
                  + '\n 1. Zoom in on the trigger you want to remove.'
                  + '\n 2. Deselect all buttons from the lower left button bar.'
                  + '\n 3. You can now click and drag a rectacle around the x axis interval enclosing the specicic '
                    'trigger onset.'
                  + '\n 4. When you are happy with the interval, close this plot to choose more intervals or proceed.')
        start, end = select_start_and_end(fig, ax, 'box')
        pd_onsets_clean = remove_trigger_onset(
            int(start), int(end), pd_onsets_clean, srate)
        user_input = input('More to remove? [Yes or No]')

    print('Done. Please inspect.')

    plt.figure(figsize=(8, 6))
    plt.plot(pd, 'g')
    plt.vlines(pd_onsets_clean["Sample_num"], min_amp,
               max_amp, 'r', zorder=10, linewidth=2)
    plt.title('Raw photodiode signal with thresholds and onset')
    plt.xlabel('Sample nr')
    plt.ylabel('Amplitude')
    plt.show()

    return pd_onsets_clean


def remove_trigger_onset(start_sample_point, end_sample_point, pd_onsets_clean, sr):
    """
    Remove a trigger onset detected within the given interval
    :param: start_sample_point (int): first sample point of interval
    :param: start_sample_point (int): last sample point of interval
    :param: pd (np.array) the raw photo diode signal
    :param: pd_onsets_clean (dict of np arrays) the detected onset indices and times
    :param: sr (float) sampling rate of signal
    :param: subject_info (subjectInfo class object)
    """

    # Check if there is an onset in the given interval.
    pd_onsets_sample_points = pd_onsets_clean['Sample_num']
    pd_onsets_times = pd_onsets_clean['Time']

    pd_onsets_sample_points_new = pd_onsets_sample_points[
        np.logical_or(pd_onsets_sample_points > end_sample_point, pd_onsets_sample_points < start_sample_point)]
    pd_onsets_times_new = pd_onsets_times[
        np.logical_or(pd_onsets_times > end_sample_point / sr, pd_onsets_times < start_sample_point / sr)]

    pd_onsets_clean['Sample_num'] = pd_onsets_sample_points_new
    pd_onsets_clean['Time'] = pd_onsets_times_new

    return pd_onsets_clean


def add_trigger_onset(start_sample_point, end_sample_point, pd, pd_onsets_clean, sr, subject_info):
    """
    Add a trigger onset detected within the given interval
    :param: start_sample_point: (int) first sample point of interval
    :param: start_sample_point: (int) last sample point of interval
    :param: pd: (np.array) the raw photo diode signal
    :param: pd_onsets_clean: (dict of np arrays) the detected onset indices and times
    :param: sr: (int) sampling rate of signal
    :param: subject_info: (subjectInfo class object)
    """

    # Select only the range we are interested in, the rest is set to zero.
    pd_selected_range_only = [pd[i] if i <= end_sample_point and i >= start_sample_point else 0 for i in
                              range(pd.shape[0])]

    # Binarizing this photodiode signal:
    binary_pd_selected_range_only = (np.array(pd_selected_range_only) > subject_info.PD_THRESHOLD).astype(
        int)  # np array dim: len(datapoints)

    # Finding the onset of the photodiode peaks:
    pd_onset_ind_selected_range_only = np.where(
        np.diff(binary_pd_selected_range_only) == 1)[0]

    # Add the onset to pd_onsets_clean:
    pd_onsets_sample_nr_new = np.sort(
        np.append(pd_onsets_clean['Sample_num'], pd_onset_ind_selected_range_only))
    pd_onsets_time_new = np.sort(
        np.append(pd_onsets_clean['Time'], pd_onset_ind_selected_range_only / sr))
    pd_onsets_clean['Sample_num'] = pd_onsets_sample_nr_new
    pd_onsets_clean['Time'] = pd_onsets_time_new

    return pd_onsets_clean


def manually_reconstruct_triggers_from_log_and_remove_triggers_by_index(full_logs_clean, pd_onsets_clean, sr):
    """
    This function plots the signal and asks for user input. This function assumes that the log file contains all the
    information you need. If something doesn't align, this must therefore be because of missing triggers in your signal
    In this function, we therefore plot the signal. The user can then identify where things misalign and feed in the
    index of the triggers where the misalignment starts. The function will retrofit it into the timeline of the
    photodiode using the diff in the log time line. This will be repeated up until alignment is achieved!
    :param full_logs_clean: (pd.DataFrame) full log files with removed unecessary entries and duplicates
    :param pd_onsets_clean: (dict of np arrays)
    :param sr: (int) sampling rate of the signal
    :return:
    """

    def query_input():
        does_it_looks_good = \
            input(
                "Do you wish to manually insert triggers [Yes] or proceed keep the signal as is [No]?")
        index_to_add = None
        index_to_remove = None

        if does_it_looks_good != 'No' and does_it_looks_good != 'no':
            continue_looping = True
            # Asking for user input:
            index_to_add = input("Give in the index of a trigger you would like to add (or leave empty if you wish to "
                                 "remove one)?")
            if index_to_add == '':
                index_to_remove = input(
                    "Give in the index of a trigger you would like to remove (or leave empty if you "
                    "wish to remove one)?")
        else:
            continue_looping = False

        return continue_looping, index_to_add, index_to_remove

    # First, setting a few flags, because we will loop until the user says otherwise:
    keep_looping = True

    # Looping until we are told to stop
    while keep_looping:
        # computing the interval of the triggers in the file and in the photodiode:
        interval_pd = np.diff(pd_onsets_clean['Time'])
        interval_log = np.diff(full_logs_clean['time'])

        # We can now plot those on top of another:
        fig, ax = plt.subplots()
        ax.plot(interval_pd, 'r', label='Photodiode')
        ax.plot(interval_log, 'b', label='Log file')
        plt.legend()
        plt.show()

        keep_looping, add_index, remove_index = query_input()

        if keep_looping:
            if add_index != '':
                add_index = int(add_index)
                try:
                    new_pd_ts = pd_onsets_clean['Time'][add_index] + \
                                (full_logs_clean['time'][add_index + 1]
                                 - full_logs_clean['time'][add_index])
                    new_pd_sample = pd_onsets_clean['Sample_num'][add_index] + \
                                    (int(
                                        full_logs_clean['time'][add_index + 1] - full_logs_clean['time'][
                                            add_index]) * sr)
                    pd_onsets_clean = {
                        'Sample_num': np.insert(pd_onsets_clean['Sample_num'], add_index + 1, new_pd_sample),
                        'Time': np.insert(pd_onsets_clean['Time'], add_index + 1, new_pd_ts)
                    }
                except KeyError:
                    print(
                        "The index you set was outside the bounds of the array. Choose another index")
                except ValueError:
                    print("Are you sure you entered a number? Try again")
            elif remove_index != '':
                try:
                    remove_index = int(remove_index)
                    # If asked to remove a specific event from the photodiode timeline, simply removing it!
                    pd_onsets_clean = {
                        'Sample_num': np.delete(pd_onsets_clean['Sample_num'], remove_index + 1),
                        'Time': np.delete(pd_onsets_clean['Time'], remove_index + 1)
                    }
                except KeyError:
                    print(
                        "The index you set was outside the bounds of the array. Choose another index")
                except ValueError:
                    print("Are you sure you entered a number? Try again")
            else:
                print(
                    "You must pass an index to be able to adjust the alignment. If you are happy as is, reply correctly"
                    "to the input")

    return pd_onsets_clean


def remove_duplicates(full_logs_no_resp, full_logs_with_resp):
    """
    If there were any restarting of blocks or miniblocks, we would have duplicates of data, which we don't want.
    Therefore, these needs to be removed. These needs to be removed from the log files, but also from the photodiode
    timestamps. This function is removing the duplicates based on the duplicated function called when loading the log
    files
    :param full_logs_with_resp:  full logs with the responses
    :param full_logs_no_resp: full logs files without the responses
    :return:
    """
    return full_logs_no_resp[full_logs_no_resp['duplicate'] != 1].reset_index(drop=True), \
           full_logs_with_resp[full_logs_with_resp['duplicate']
                               != 1].reset_index(drop=True)


def introduce_response(full_logs_clean, full_logs_with_resp):
    """
    This function reintroduces the response timestamps in the photodiode timeline, by taking the interval between
    previous event and and response in the log files with the response. That way, the responses are in the photodiode
    timeline as well
    :param full_logs_clean: pandas data frame of the full log of the experiment where the responses were removed
    :param full_logs_with_resp: pandas data frame of the full log of the experiment with the responses
    :return:
    """
    # Looping through the log file with the response
    for row_ind, row in full_logs_with_resp.iterrows():
        # If in the log file with the response, the current row is a response, the we add a row to the clean log:
        if row.eventType == 'Response':
            # Computing the delta t of the response between the previous event and the response
            resp_delay = row["time"] - full_logs_with_resp.time[row_ind - 1]
            # The response delay is added to the previous  photodiode timestamp. This generates the time stamps in the
            # photodiode timeline for the response:
            row["time_PD"] = full_logs_clean.time_PD[row_ind - 1] + resp_delay
            # Now, the new row must be inserted
            full_logs_clean = full_logs_clean.append(row)
            # Finally, it needs to be reordered by the photodiode timestamp:
            full_logs_clean = full_logs_clean.sort_values(
                by=["time_PD"]).reset_index(drop=True)

    # If things went well, the interval between events should be the same in the full logs with the response, and the
    # full logs in which we just introduced the response. But if for whatever reason that is no the case, raise an
    # error
    if not all([diff == 0 for diff in full_logs_clean.time - full_logs_with_resp.time]):
        raise Exception(
            "The introduction of responses in the clean logs caused an error")

    return full_logs_clean


def check_alignment(pd_onsets_clean, full_logs, data_preparation_parameters, subject_info, sr):
    """
    This function checks the alignment between the logs files and the photodiode trigger to be able to create the
    correct events down the line
    :param subject_info: object containing specific info for this subject
    :param data_preparation_parameters: custom made object containing specific info for the analysis
    :param pd_onsets_clean: pandas series containing the timestamps of the detected photodiode triggers, with block
    onset and end removed
    :param sr: sampling rate of the signal, necessary in case triggers get messed up
    :param full_logs: Loaded full logs of the experiment
    :return: PD_onsets_clean: photodiode with extra triggers in the end removed
    :return: full_logs_clean: full logs with extra entries (save, responses...) removed
    """

    # Plot log to see that it looks reasonable
    interval_logs = np.diff(full_logs.time)
    plt.plot(range(len(full_logs.time)), full_logs.time)
    plt.title('Log file time stamps. Check that it is monotonically growing')
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    # Removing log entries that do not have any matching photodiode entries
    full_logs_clean = full_logs.loc[(
            full_logs["eventType"] != "Save")].reset_index(drop=True)
    # Removing these (only there has been a restart)
    full_logs_clean = full_logs_clean.loc[(
            full_logs_clean["eventType"] != "TargetScreenOnset")].reset_index(drop=True)
    full_logs_clean = full_logs_clean.loc[(
            full_logs_clean["eventType"] != "Pause")].reset_index(drop=True)
    full_logs_clean = full_logs_clean.loc[(
            full_logs_clean["eventType"] != "Abortion")].reset_index(drop=True)
    full_logs_clean = full_logs_clean.loc[(
            full_logs_clean["eventType"] != "Interruption")].reset_index(drop=True)
    # Removing the first jitter and fixation from each block, because there are not corresponding photodiode triggers:
    indices_to_drop = []
    mini_block_nr = 0
    for i in range(len(full_logs_clean)):
        mini_block_nr_now = full_logs_clean["miniBlock"][i]
        # Now we are at the beginning of a new block:
        if mini_block_nr_now != mini_block_nr:
            indices_to_drop.append(i)
            indices_to_drop.append(i + 1)
        mini_block_nr = mini_block_nr_now

    full_logs_clean.drop(full_logs_clean.index[indices_to_drop], inplace=True)

    # Resetting the index and saving a version with responses in there for later:
    full_logs_with_resp = full_logs_clean.reset_index(drop=True)
    full_logs_clean = full_logs_clean.loc[(
            full_logs_clean["eventType"] != "Response")].reset_index(drop=True)

    # Then, computing intervals:
    # Between successive timestamps in the logs:
    interval_logs = np.diff(full_logs_clean.time)
    # And in the photodiode signal:
    interval_pd = np.diff(pd_onsets_clean["Time"])

    # Making sure there are as many triggers as there are log entries
    print('Now trying to plot the full log entries vs the pd entries. ')
    print('If you get an error here, it means your triggers don\'t match the number of log entries')
    if len(interval_logs) != len(interval_pd):
        print("The number of triggers was not equivalent between the logs and the photodiode. "
              + "\n Nr of detected triggers: " + str(len(interval_pd) + 1)
              + "\n Nr of log entries: " + str(len(interval_logs) + 1))

        print("You will have to manually intervene somehow. "
              "\n If you want to check that all onsets are there, restart the data preparation script, make sure to"
              "\n preprocessing in show_check_plots = 1 mode (as set in the analysisParameter config file).")

        manual_extrapolation_from_log_file = input(
            'But if you want to reconstruct missing triggers by extrapolating log file entries '
            '\n or remove triggers by index, you can do that now [Yes or No].')
        if manual_extrapolation_from_log_file:
            # Calling the function to manually realign the signal:
            pd_onsets_clean = manually_reconstruct_triggers_from_log_and_remove_triggers_by_index(full_logs_clean,
                                                                                                  pd_onsets_clean, sr)
        # Recomputing the photodiode intervals in case it was changed:
        interval_pd = np.diff(pd_onsets_clean["Time"])
    else:
        print(
            "SUCCESS! There was the same number of photodiode and log files events and we will go straight to plotting")

    # Get the path where to save data
    save_path = Path(subject_info.participant_save_root, "info", "figure")

    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    # Finally, plotting the alignment:
    plt.figure(figsize=(8, 6))
    plt.plot(interval_logs, color="red")
    plt.plot(interval_pd, color="blue")
    plt.legend(["Logs", "Photodiode"])
    plt.title("Alignment photodiode vs logs \n difference between consecutive events (stimulus, fixation, jitter)"
              "\n # detected triggers: " + str(len(interval_logs) + 1) + ", # log entries: " + str(
        len(interval_pd) + 1))
    plt.xlabel("Event nr")
    plt.ylabel('Difference in time [s]')
    plt.savefig(os.path.join(save_path, "Photodiode_log_alignment.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    # Dumping the subject info and the analysis parameters with the figures:
    subject_info.save(save_path)
    data_preparation_parameters.save(save_path, subject_info.files_prefix)
    try:
        # Following confirmation that the alignment is fine, the photodiode time stamps are copied to the clean logs:
        full_logs_clean["time_PD"] = pd_onsets_clean["Time"]
    except ValueError:
        raise ValueError("The number of photodiode events does not match the number of log file events! "
                         "You must sort this out for this function to preprocessing!")

    if (full_logs_clean['duplicate'] == 1).any():
        remove_aborted_data = \
            input('It seems you have restarted the experiment. For the trials that were preprocessing twice,'
                  ' \n would you like to keep only the more recent ones (afer restarting) [Yes, No]?')

        if remove_aborted_data == 'yes' or remove_aborted_data == 'Yes':
            # Now, if there was a restart, we need to remove:
            full_logs_clean, full_logs_with_resp = remove_duplicates(
                full_logs_clean, full_logs_with_resp)

            # Then, computing intervals:
            # Between successive timestamps in the logs:
            interval_logs = np.diff(full_logs_clean.time)
            # And in the photodiode signal:
            interval_pd = np.diff(full_logs_clean.time_PD)

            print('Now trying to plot the full log entries vs the pd entries again. ')
            print(
                'If you get an error here, it means your triggers don\'t match the number of log entries')
            print('len(interval_logs): ', len(interval_logs))
            print('len(interval_pd): ', len(interval_pd))

            # Make new plots
            # Plotting the alignment:
            plt.figure(figsize=(8, 6))
            plt.plot(interval_logs, color="red")
            plt.plot(interval_pd, color="blue")
            plt.legend(["Logs", "Photodiode"])
            plt.title(
                "Alignment photodiode vs logs difference between consecutive events \n (stimulus, fixation, jitter) "
                "after aborted data has been removed, \n # detected triggers: "
                + str(len(interval_logs) + 1)
                + ", # log entries: " + str(len(interval_pd) + 1))
            plt.xlabel("Event nr")
            plt.ylabel('Difference in time [s]')
            plt.savefig(os.path.join(
                save_path, "Photodiode_log_alignment_no_aborted_data.png"))
            if data_preparation_parameters.show_check_plots:
                plt.show()
            plt.close()
    full_logs_clean = introduce_response(full_logs_clean, full_logs_with_resp)

    return full_logs_clean


def check_stim_duration(logs, data_preparation_parameters, subject_info):
    """
    This function checks the events duration of the stimuli against what they should have been
    :param subject_info:
    :param data_preparation_parameters:
    :param logs:
    :return:
    """

    # Getting the index of the stim onsets:
    stim_log = logs.loc[logs["eventType"] == "Stimulus"]
    fixation_log = logs.loc[logs["eventType"] == "Fixation"]
    jitter_log = logs.loc[logs["eventType"] == "Jitter"]

    # Resetting the index to make the indices match across the three vectors
    stim_log = stim_log.reset_index(drop=True)
    fixation_log = fixation_log.reset_index(drop=True)
    jitter_log = jitter_log.reset_index(drop=True)

    # Equating the size of the arrays. It is possible that interruption occured in the middle of a trial, meaning there
    # could be non equivalence of the size of the arrays, because for example the jitter didn't occur. In this scenario,
    # the duration computations will fail. Therefore, equating their size in advance. NOTE THAT THERE IS A RISK THAT
    # THIS LEADS TO MISALIGNMENT, ALWAYS CHECK THE OUTPUT FIGs:
    min_len = min([len(stim_log), len(fixation_log), len(jitter_log)])
    # Equating the size if needed:
    stim_log = stim_log.iloc[0:min_len]
    fixation_log = fixation_log.iloc[0:min_len]
    jitter_log = jitter_log.iloc[0:min_len]

    # Compute the stim duration inaccuracies:
    stim_dur_pd = [fixation_log["time_PD"] - stim_log["time_PD"]]
    stim_dur_inaccuracies_ms = (
                                       stim_dur_pd[0] - stim_log["plndStimulusDur"]) * 1000

    # Compute the fixation inaccuracies
    fixation_dur_pd = np.array(
        jitter_log["time_PD"]) - np.array(fixation_log["time_PD"])
    fixation_dur_inaccuracies_ms = (
                                           fixation_dur_pd - (2.0 - np.array(stim_log["plndStimulusDur"]))) * 1000

    # Compute the jitter inaccuracies. These are a little bit more complicated since we
    # don't have a stimulus onset reference for the ones at the end of a mb.
    # We also want to adjust the comparison so that we compare to stimuli onsets
    # corresponding to the next index.
    # We do this mb per mb:
    jitter_dur_inaccuracies_ms = list()
    stim_onsets_pd = np.array(stim_log["time_PD"])
    jitter_onsets_pd = np.array(jitter_log["time_PD"])
    planned_jitter_log = np.array(jitter_log["plndJitterDur"])
    targets = np.array(stim_log["targ1"])
    target = targets[0]
    for trial in range(len(stim_onsets_pd) - 1):
        next_stim_onset_pd = stim_onsets_pd[trial + 1]
        jitter_onset_pd = jitter_onsets_pd[trial]
        planned_jitter_dur_log = planned_jitter_log[trial]
        target_next_trial = targets[trial + 1]
        # Are we not entering a new miniblock? If we are we have to skip this entry
        if not target != target_next_trial:
            jitter_dur_inaccuracy = next_stim_onset_pd - \
                                    jitter_onset_pd - planned_jitter_dur_log
            jitter_dur_inaccuracies_ms.append(jitter_dur_inaccuracy * 1000)
        else:
            target = target_next_trial

    # Where to save the figures
    # Get the path where to save data
    save_path = Path(subject_info.participant_save_root, "info", "figure")

    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    # Plot stimulus duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.plot(stim_dur_inaccuracies_ms, '.k')
    plt.hlines(16.6, 0, len(stim_dur_inaccuracies_ms))
    plt.hlines(-16.6, 0, len(stim_dur_inaccuracies_ms))
    plt.xlabel('Sample nr')
    plt.ylabel('Duration inaccuracy [ms] ')
    plt.title(
        'Difference between measured and planned duration of stimulus presentation')
    # Percentage missed frames:
    stim_dur_inaccuracies_ms_np = np.array(stim_dur_inaccuracies_ms)
    missed_frames = stim_dur_inaccuracies_ms_np[np.absolute(
        stim_dur_inaccuracies_ms_np) > 14.]
    ratio_missed_frames = round(
        np.size(missed_frames) / np.size(stim_dur_inaccuracies_ms_np), 2)
    text = 'Missed frames: ' + str(ratio_missed_frames * 100) + '%'
    plt.text(0.5 * np.size(stim_dur_inaccuracies_ms_np), 13, text)
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_stimuli.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()

    # Plot fixation duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.plot(fixation_dur_inaccuracies_ms, '.k')
    plt.hlines(16.6, 0, len(fixation_dur_inaccuracies_ms))
    plt.hlines(-16.6, 0, len(fixation_dur_inaccuracies_ms))
    plt.xlabel('Sample nr')
    plt.ylabel('Duration inaccuracy [ms] ')
    plt.title('Difference between measured and planned duration of fixations')
    # Percentage missed frames:
    fixation_dur_inaccuracies_ms_np = np.array(fixation_dur_inaccuracies_ms)
    missed_frames = fixation_dur_inaccuracies_ms_np[np.absolute(
        fixation_dur_inaccuracies_ms_np) > 14.]
    ratio_missed_frames = round(
        np.size(missed_frames) / np.size(fixation_dur_inaccuracies_ms_np), 2)
    text = 'Missed frames: ' + str(ratio_missed_frames * 100) + '%'
    plt.text(0.5 * np.size(fixation_dur_inaccuracies_ms_np), 13, text)
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_fixations.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()

    # Plot stimulus duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.plot(jitter_dur_inaccuracies_ms, '.k')
    plt.hlines(16.6, 0, len(jitter_dur_inaccuracies_ms))
    plt.hlines(-16.6, 0, len(jitter_dur_inaccuracies_ms))
    plt.xlabel('Sample nr')
    plt.ylabel('Duration inaccuracy [ms] ')
    plt.title('Difference between measured and planned duration of jitters')
    # Percentage missed frames:
    jitter_dur_inaccuracies_ms_np = np.array(jitter_dur_inaccuracies_ms)
    missed_frames = jitter_dur_inaccuracies_ms_np[np.absolute(
        jitter_dur_inaccuracies_ms_np) > 14.]
    ratio_missed_frames = round(
        np.size(missed_frames) / np.size(jitter_dur_inaccuracies_ms_np), 2)
    text = 'Missed frames: ' + str(ratio_missed_frames * 100) + '%'
    plt.text(0.5 * np.size(jitter_dur_inaccuracies_ms_np), 13, text)
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_jitters.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()

    # Histogram with stimuli, jitter and blanks, separately

    # Stimulus duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.hist(stim_dur_inaccuracies_ms, bins=100, color="black")
    stim_dur_avg = (sum(stim_dur_inaccuracies_ms)
                    / len(stim_dur_inaccuracies_ms))
    stim_dur_std = np.std(np.array(stim_dur_inaccuracies_ms))
    axes = plt.gca()
    ylims = axes.get_ylim()
    plt.vlines(stim_dur_avg, ylims[0], ylims[1],
               colors="green", linestyles="solid", label='Mean')
    plt.vlines((stim_dur_avg + stim_dur_std),
               ylims[0], ylims[1], colors="r", linestyles="dotted", label="std")
    plt.vlines((stim_dur_avg - stim_dur_std),
               ylims[0], ylims[1], colors="r", linestyles="dotted")
    plt.xlabel('Duration inaccuracy [ms] ')
    plt.legend()
    plt.title('Difference between measured and planned duration of stimuli')
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_hist_all_events.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    # Fixation duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.hist(fixation_dur_inaccuracies_ms, bins=100, color="black")
    fixation_dur_avg = np.mean(fixation_dur_inaccuracies_ms)
    fixation_dur_std = np.std(fixation_dur_inaccuracies_ms)
    axes = plt.gca()
    ylims = axes.get_ylim()
    plt.vlines(fixation_dur_avg, ylims[0], ylims[1],
               colors="green", linestyles="solid", label='Mean')
    plt.vlines((fixation_dur_avg + fixation_dur_std), ylims[0], ylims[1], colors="r", linestyles="dotted",
               label="std")
    plt.vlines((fixation_dur_avg - fixation_dur_std),
               ylims[0], ylims[1], colors="r", linestyles="dotted")
    plt.xlabel('Duration inaccuracy [ms] ')
    plt.legend()
    plt.title('Difference between measured and planned duration of fixations')
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_hist_fixations.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    # Jitter duration inaccuracies
    plt.figure(figsize=(8, 6))
    plt.hist(jitter_dur_inaccuracies_ms, bins=100, color="black")
    jitter_dur_avg = np.mean(jitter_dur_inaccuracies_ms)
    jitter_dur_std = np.std(jitter_dur_inaccuracies_ms)
    axes = plt.gca()
    ylims = axes.get_ylim()
    plt.vlines(jitter_dur_avg, ylims[0], ylims[1],
               colors="green", linestyles="solid", label='Mean')
    plt.vlines((jitter_dur_avg + jitter_dur_std), ylims[0], ylims[1], colors="r", linestyles="dotted",
               label="std")
    plt.vlines((jitter_dur_avg - jitter_dur_std),
               ylims[0], ylims[1], colors="r", linestyles="dotted")
    plt.xlabel('Duration inaccuracy [ms] ')
    plt.legend()
    plt.title('Difference between measured and planned duration of jitter')
    plt.savefig(os.path.join(save_path,
                             "Duration_inaccuracy_hist_jitter.png"))
    if data_preparation_parameters.show_check_plots:
        plt.show()
    plt.close()
    # Dumping the subject info and the analysis parameters with the figures:
    subject_info.save(save_path)
    data_preparation_parameters.save(save_path, subject_info.files_prefix)


def check_if_response_in_this_trial(event_type, i):
    """
    Check of this trial includes a response.
    :param: eventType (list) al the log entries events (Stimulus, Fixation, Jitter, Response)
    :param: i (int) the index of the 'Stimulus' event log entry corresponding to the trial to be checked
    :return: response_present (bool) whether there is a response in this trial or not
    """

    response_present = False

    # The response can come after the fixation and even after the jitter(?). If no response is there
    # full_logs.eventType[i + 3] will be the next "Stimulus".
    # To not reach the end of the file, let's first check how far we can go:
    # Can fix this to be nicer if we keep this implementation

    remaining_entries = len(event_type) - i - 1

    if remaining_entries >= 3:
        if event_type[i + 1] == "Response" or event_type[i + 2] == "Response" or event_type[i + 3] == "Response":
            response_present = True
    elif remaining_entries >= 2:
        if event_type[i + 1] == "Response" or event_type[i + 2] == "Response":
            response_present = True
    elif remaining_entries >= 1:
        if event_type[i + 1] == "Response":
            response_present = True

    return response_present


def create_annotations(full_logs, raw_sig):
    """
    This function creates the events from scratch, based on the photodiode for the onsets and on the log files for the
    info
    :param full_logs: log files of experiment 1 (pd.DataFrame)
    :param raw_sig: mne raw object containing the ECoG signal
    :return: raw_Sig: raw signal with added events
    """

    # First, creating np array with time of the Photodiode triggers onsets:
    onsets = full_logs["time_PD"]

    # To get the duration of each events, looping through the full logs:
    duration = []
    description = []
    for i in range(len(full_logs.index)):
        # For each event, adding block and miniblock number:
        block = "block_" + str(full_logs.block[i])
        miniblock = "miniblock_" + str(full_logs.miniBlock[i])

        # If the current entry is a stimulus:
        if full_logs.eventType[i] == "Stimulus":
            event_type = "stimulus onset"
            hit = False
            miss = False
            FA = False
            # If the stimulus row is not the last entry of the table (as it can
            if i + 1 < len(full_logs.index):
                # happen in case of interruptions)
                duration.append(
                    full_logs.time_PD[i + 1] - full_logs.time_PD[i])
                # Now getting the rest of the info:
                # Getting the stimulus category:
                if str(full_logs.event[i])[0] == "1":
                    stim_category = "face"
                elif str(full_logs.event[i])[0] == "2":
                    stim_category = "object"
                elif str(full_logs.event[i])[0] == "3":
                    stim_category = "letter"
                elif str(full_logs.event[i])[0] == "4":
                    stim_category = "false"

                # Now getting the stimulus orientation:
                if str(full_logs.event[i])[1] == "1":
                    stim_or = "Center"
                elif str(full_logs.event[i])[1] == "2":
                    stim_or = "Left"
                elif str(full_logs.event[i])[1] == "3":
                    stim_or = "Right"

                # Now getting the stimulus ID, which is the number + the stim type, as to have a single identifier
                # for each identity:
                stim_id = stim_category + "_" + \
                          str(full_logs.event[i])[2] + str(full_logs.event[i])[3]

                # Converting the duration into words for ease of access (need to round a bit because the duration
                # depends on the refresh rate:
                if full_logs.plndStimulusDur[i] < 0.6:
                    stim_dur = "500ms"
                elif 0.6 < full_logs.plndStimulusDur[i] < 1.1:
                    stim_dur = "1000ms"
                elif full_logs.plndStimulusDur[i] > 1.1:
                    stim_dur = "1500ms"

                # Finally, getting the task relevance:
                tar1 = full_logs.targ1[i]
                tar2 = full_logs.targ2[i]

                response_present = check_if_response_in_this_trial(
                    full_logs.eventType, i)

                # Check relevance
                if (str(full_logs.event[i])[0] == str(tar1)[0] and str(full_logs.event[i])[2:4] == str(tar1)[2:4]) or \
                        (str(full_logs.event[i])[0] == str(tar2)[0] and str(full_logs.event[i])[2:4] == str(tar2)[2:4]):
                    # This is a target:
                    stim_tr = "Relevant target"

                    # Check for hits and misses
                    if response_present:
                        hit = True
                    else:
                        miss = True
                else:  # not target
                    # Check for false alarms
                    if response_present:
                        FA = True
                    if (str(full_logs.event[i])[0] == str(tar1)[0]) or (str(full_logs.event[i])[0] == str(tar2)[0]):
                        # This is a task relevant non target:
                        stim_tr = "Relevant non-target"
                    else:
                        stim_tr = "Irrelevant"

                # Now that we have everything we need, the description can be added:
                if hit:
                    description.append(
                        "/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                  stim_dur, stim_tr, "Hit"]))
                elif miss:
                    description.append(
                        "/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                  stim_dur, stim_tr, "Miss"]))
                elif FA:
                    description.append(
                        "/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                  stim_dur, stim_tr, "FA"]))
                else:
                    description.append(
                        "/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                  stim_dur, stim_tr, "CorrRej"]))
            else:  # But if the stimulus is the last event of the table, then it is something that should be discarded,
                # because the trial wasn't full-filled
                onsets = onsets.drop(i)

        elif full_logs.eventType[i] == "Response":
            duration.append(0)
            event_type = "response"
            # Fetch the previous stimulus onset information:
            for ii in range(len(description), 0, -1):
                if "stimulus onset" in description[ii-1]:
                    ind = ii - 1
                    break
            # Get all the info for this response:
            block, miniblock, stim_category, stim_id, stim_or, stim_dur, stim_tr = description[ind].split("/")[1:-1]

            # If the stimulus before was a target, then this is a hit
            if FA:
                description.append("/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                             stim_dur, stim_tr, "response_FA"]))
            elif hit:
                description.append(
                    "/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                              stim_dur, stim_tr, "response_Hit"]))
            else:
                description.append("/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                             stim_dur, stim_tr, "Response"]))

        elif full_logs.eventType[i] == "Fixation":
            event_type = "stimulus offset"
            # Fetch the previous stimulus onset information:
            for ii in range(len(description), 0, -1):
                if "stimulus onset" in description[ii-1]:
                    ind = ii - 1
                    break
            # Get all the info for this response:
            block, miniblock, stim_category, stim_id, stim_or, stim_dur, stim_tr = description[ind].split("/")[1:-1]
            # If the stimulus row is not the last entry of the table (as it can
            if i + 1 < len(full_logs.index):
                # happen in case of interruptions)
                duration.append(
                    full_logs.time_PD[i + 1] - full_logs.time_PD[i])
                description.append("/".join([event_type, block, miniblock, stim_category, stim_id, stim_or,
                                             stim_dur, stim_tr, "n.a."]))
            else:
                # Then again, in case of interruption, you don't want to keep that as it might be spurious:
                onsets = onsets.drop(i)

        elif full_logs.eventType[i] == "Jitter":
            event_type = "jitter onset"
            # Fetch the previous stimulus onset information:
            for ii in range(len(description), 0, -1):
                if "stimulus onset" in description[ii-1]:
                    ind = ii - 1
                    break
            # Get all the info for this response:
            block, miniblock, stim_category, stim_id, stim_or, stim_dur, stim_tr = description[ind].split("/")[1:-1]

            # If the jitter is no the last entry of the table:
            if i + 1 < len(full_logs.index):
                # another one. But for the last trial of the experiment, we can't compute it because we have no onset of
                # what comes next
                interval = full_logs.time_PD[i + 1] - full_logs.time_PD[i]
                if interval < 2 + 0.5 * 2:
                    # Also, for the transition between miniblocks, the interval between end of a
                    # trial and start of the next, we have the duration of the pause added. So here, I check if the
                    # duration of the interval is above 3sec, because the max jitter is actually 2sec. So if it is
                    # inferior, all good
                    duration.append(interval)
                else:  # But if we are above that, we are at the transition between miniblocks and can't really compute
                    # what it actually was. Therefore, taking the plnd jitter instead
                    duration.append(full_logs.plndJitterDur[i])

                description.append("/".join([event_type, block, miniblock, stim_category, stim_id, stim_or, stim_dur,
                                             stim_tr,"n.a."]))
            else:  # But if the jitter is the last entry of the table, things are as they should be: the last event of
                # the experiment should be a jitter. Now if there was an interruption in the end, as the jitter was
                # presented, at least, there was a full trial, so I keep it. In any case, any sort of weird things that
                # happened in the end of a block and interruption will be sorted out by manual annotations later on
                duration.append(full_logs.plndJitterDur[i])

                description.append("/".join([event_type, block, miniblock, stim_category, stim_id, stim_or, stim_dur,
                                             stim_tr, "n.a."]))

    # Convert list to NP:
    duration = np.array(duration)

    # Creating annotation from onset, duration and description:
    my_annotations = mne.Annotations(onset=onsets,
                                     duration=duration,
                                     description=description)

    # Setting the annotation in the raw signal
    raw_sig.set_annotations(my_annotations)

    return raw_sig
