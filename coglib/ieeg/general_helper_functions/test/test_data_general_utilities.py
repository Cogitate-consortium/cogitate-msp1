import unittest
import mne
from mne.datasets import sample
from general_helper_functions.data_general_utilities import cluster_test
from general_helper_functions import data_general_utilities
from mne.baseline import rescale
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pandas as pd


def use_sample_data():
    # Prepare the data:
    data_path = sample.data_path()
    # Load and filter data, set up epochs
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    events_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # Pick MEG channels
    raw.filter(1., 30., fir_design='firwin')  # Band pass filtering signals
    events = mne.read_events(events_fname)
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
                'Visual/Left': 3, 'Visual/Right': 4}
    tmin = -0.050
    tmax = 0.400
    # decimate to make the example faster to preprocessing, but then use verbose='error' in
    # the Epochs constructor to suppress warning about decimation causing aliasing
    decim = 2
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        proj=True, picks=picks, baseline=None, preload=True,
                        reject=dict(mag=5e-12), decim=decim, verbose='error')
    return epochs


class TestClusterTest(unittest.TestCase):

    def test_positive_2d(self):
        # Generate random z score data:
        x = np.random.normal(loc=0, scale=.1, size=(80, 80))
        # Set part of the matrix to something we know must be significant:
        x[10:20, 10:20] = x[10:20, 10:20] + 2
        # Generate a null distribution with a bit of noise:
        h0 = np.zeros([1024, *x.shape]) + np.random.normal(loc=0, scale=1.0, size=[1024, *x.shape])
        for i in range(h0.shape[0]):
            # In 1% of the cases, create a cluster that adds up to being more than the observed highest cluster:
            if np.random.rand() < 0.04:
                h0[i, 5:10, 5:10] = 10
        x, h0, clusters, cluster_pv, p_values_, H0 = \
            cluster_test(x, h0, z_threshold=1.5, adjacency=None, tail=1, max_step=None, exclude=None,
                         t_power=1, step_down_p=0.05, do_zscore=False)

        # Checking vs expected:
        obs_sig = p_values_ < 0.05
        exp_sig = np.zeros(p_values_.shape)
        exp_sig[:, :] = False
        exp_sig[10:20, 10:20] = True
        self.assertTrue((obs_sig == exp_sig).all())

        print("A")

    def test_positive_1d(self):
        # Generate random z score data:
        x = np.random.normal(loc=0, scale=.1, size=(80))
        # Set part of the matrix to something we know must be significant:
        x[10:20] = x[10:20] + 2
        # Generate a null distribution with a bit of noise:
        h0 = np.zeros([1024, *x.shape]) + np.random.normal(loc=0, scale=1.0, size=[1024, *x.shape])
        for i in range(h0.shape[0]):
            # In 1% of the cases, create a cluster that adds up to being more than the observed highest cluster:
            if np.random.rand() < 0.04:
                h0[i, 5:10] = 10
        x, h0, clusters, cluster_pv, p_values_, H0 = \
            cluster_test(x, h0, z_threshold=1.5, adjacency=None, tail=1, max_step=None, exclude=None,
                         t_power=1, step_down_p=0.05, do_zscore=False)

        # Checking vs expected:
        obs_sig = p_values_ < 0.05
        exp_sig = np.zeros(p_values_.shape)
        exp_sig[:] = False
        exp_sig[10:20] = True
        self.assertTrue((obs_sig == exp_sig).all())


class TestComputeDependentVariable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create the info for the channels objects:
        info = mne.create_info(ch_names=['ch1', 'ch2'],
                               ch_types=['misc'] * 2,
                               sfreq=512)
        # Create an epochs only made of ones:
        data = np.array([[np.ones(5), np.ones(5)],
                         [np.ones(5), np.ones(5)],
                         [np.ones(5), np.ones(5)],
                         [np.ones(5), np.ones(5)]])
        cls.epochs_ones = mne.EpochsArray(data, info)
        # Create an epochs only made of zeros:
        data = np.array([[np.zeros(5), np.zeros(5)],
                         [np.zeros(5), np.zeros(5)],
                         [np.zeros(5), np.zeros(5)],
                         [np.zeros(5), np.zeros(5)]])
        cls.epochs_zeros = mne.EpochsArray(data, info)
        # Create an epochs made of random numbers:
        data = np.array([[np.random.rand(100), np.random.rand(100)],
                         [np.random.rand(100), np.random.rand(100)],
                         [np.random.rand(100), np.random.rand(100)],
                         [np.random.rand(100), np.random.rand(100)]])
        cls.epochs_random = mne.EpochsArray(data, info)

        # For each create the metadata:
        metadata = pd.DataFrame({
            "conditionA": ["cond1", "cond2", "cond1", "cond2"],
            "conditionB": ["cond3", "cond4", "cond3", "cond4"]},
            index=range(len(cls.epochs_random)))
        # Adding it to each of the created epochs:
        cls.epochs_ones.metadata = metadata
        cls.epochs_zeros.metadata = metadata
        cls.epochs_random.metadata = metadata

    def test_mean(self):
        # In this test, we test the mean functionality of the function:
        df_ones_mean = data_general_utilities.compute_dependent_variable(self.epochs_ones, metric="mean")
        df_zeros_mean = data_general_utilities.compute_dependent_variable(self.epochs_zeros, metric="mean")
        df_rand_mean = data_general_utilities.compute_dependent_variable(self.epochs_random, metric="mean")
        # First of all, checking that the length of the data frame is correct:
        expected_length = len(self.epochs_ones.ch_names) * len(self.epochs_ones)
        self.assertEqual(len(df_ones_mean), expected_length)
        self.assertEqual(len(df_zeros_mean), expected_length)
        self.assertEqual(len(df_rand_mean), expected_length)
        # Then for the ones and zeros, we expect the values to be only ones and zeros:
        self.assertEqual(set(df_ones_mean["value"].to_list()).pop(), 1.)
        self.assertEqual(set(df_zeros_mean["value"].to_list()).pop(), 0.)
        # For the random numbers, looping through each epochs
        df_control = pd.DataFrame()
        for trial_ind, epoch in enumerate(self.epochs_random):
            # Get the data of each channel:
            for ind, ch in enumerate(self.epochs_random.ch_names):
                data = epoch[ind, :]
                # Now computing the average:
                avg = np.mean(np.squeeze(data))
                # Appending this to the results data  frame:
                df_control = df_control.append(pd.DataFrame({"channel": ch, "epoch": int(trial_ind), "value": avg},
                                                            index=[1]), ignore_index=True)
        df_control = df_control.sort_values(by=["channel"]).reset_index(drop=True)
        # Now comparing the data frame created that way vs the other:
        assert_almost_equal(df_control["value"].to_list(), df_rand_mean["value"].to_list())

    def test_range(self):
        # In this test, we test the range (or peak to peak) functionality of the function:
        df_ones_range = data_general_utilities.compute_dependent_variable(self.epochs_ones, metric="ptp")
        df_zeros_range = data_general_utilities.compute_dependent_variable(self.epochs_zeros, metric="ptp")
        df_rand_range = data_general_utilities.compute_dependent_variable(self.epochs_random, metric="ptp")
        # First of all, checking that the length of the data frame is correct:
        expected_length = len(self.epochs_ones.ch_names) * len(self.epochs_ones)
        self.assertEqual(len(df_ones_range), expected_length)
        self.assertEqual(len(df_zeros_range), expected_length)
        self.assertEqual(len(df_rand_range), expected_length)
        # Then for the ones and zeros, we expect the values to be only ones and zeros:
        self.assertEqual(set(df_ones_range["value"].to_list()).pop(), 0.)
        self.assertEqual(set(df_zeros_range["value"].to_list()).pop(), 0.)
        # For the random numbers, looping through each epochs
        df_control = pd.DataFrame()
        for trial_ind, epoch in enumerate(self.epochs_random):
            # Get the data of each channel:
            for ind, ch in enumerate(self.epochs_random.ch_names):
                data = epoch[ind, :]
                # Now computing the average:
                avg = np.ptp(np.squeeze(data))
                # Appending this to the results data  frame:
                df_control = df_control.append(pd.DataFrame({"channel": ch, "epoch": int(trial_ind), "value": avg},
                                                            index=[1]), ignore_index=True)
        df_control = df_control.sort_values(by=["channel"]).reset_index(drop=True)
        # Now comparing the data frame created that way vs the other:
        assert_almost_equal(df_control["value"].to_list(), df_rand_range["value"].to_list())

    def test_auc(self):
        # In this test, we test the area under the curve functionality of the function:
        df_ones_auc = data_general_utilities.compute_dependent_variable(self.epochs_ones, metric="auc")
        df_zeros_auc = data_general_utilities.compute_dependent_variable(self.epochs_zeros, metric="auc")
        df_rand_auc = data_general_utilities.compute_dependent_variable(self.epochs_random, metric="auc")
        # First of all, checking that the length of the data frame is correct:
        expected_length = len(self.epochs_ones.ch_names) * len(self.epochs_ones)
        self.assertEqual(len(df_ones_auc), expected_length)
        self.assertEqual(len(df_zeros_auc), expected_length)
        self.assertEqual(len(df_rand_auc), expected_length)
        # Then for the ones and zeros, we expect the values to be only ones and zeros:
        self.assertEqual(set(df_ones_auc["value"].to_list()).pop(), 4.)
        self.assertEqual(set(df_zeros_auc["value"].to_list()).pop(), 0.)
        # For the random numbers, looping through each epochs
        df_control = pd.DataFrame()
        for trial_ind, epoch in enumerate(self.epochs_random):
            # Get the data of each channel:
            for ind, ch in enumerate(self.epochs_random.ch_names):
                data = epoch[ind, :]
                # Now computing the average:
                avg = np.trapz(np.squeeze(data))
                # Appending this to the results data  frame:
                df_control = df_control.append(pd.DataFrame({"channel": ch, "epoch": int(trial_ind), "value": avg},
                                                            index=[1]), ignore_index=True)
        df_control = df_control.sort_values(by=["channel"]).reset_index(drop=True)
        # Now comparing the data frame created that way vs the other:
        assert_almost_equal(df_control["value"].to_list(), df_rand_auc["value"].to_list())

    def test_one_conditions(self):
        # In this test, we test the appending of the condition to the df:
        df_ones_auc = data_general_utilities.compute_dependent_variable(self.epochs_ones, metric="auc",
                                                                        conditions="conditionA")
        # Expected df:
        expected_df = pd.DataFrame({
            "channel": ["ch1", "ch1", "ch1", "ch1", "ch2", "ch2", "ch2", "ch2"],
            "epoch": [1, 2, 3, 4, 1, 2, 3, 4],
            "condition": ["cond1", "cond2", "cond1", "cond2", "cond1", "cond2", "cond1", "cond2"]
        })

        # Now comparing the conditions of the df to the expected one:
        self.assertListEqual(df_ones_auc["condition"].to_list(), expected_df["condition"].to_list())

    def test_two_conditions(self):
        # In this test, we test the appending of the condition to the df:
        df_ones_auc = data_general_utilities.compute_dependent_variable(self.epochs_ones, metric="auc",
                                                                        conditions=["conditionA", "conditionB"])
        # Expected df:
        expected_df = pd.DataFrame({
            "channel": ["ch1", "ch1", "ch1", "ch1", "ch2", "ch2", "ch2", "ch2"],
            "epoch": [1, 2, 3, 4, 1, 2, 3, 4],
            "conditionA": ["cond1", "cond2", "cond1", "cond2", "cond1", "cond2", "cond1", "cond2"],
            "conditionB": ["cond3", "cond4", "cond3", "cond4", "cond3", "cond4", "cond3", "cond4"]
        })

        # Now comparing the conditions of the df to the expected one:
        self.assertListEqual(df_ones_auc["conditionA"].to_list(), expected_df["conditionA"].to_list())
        self.assertListEqual(df_ones_auc["conditionB"].to_list(), expected_df["conditionB"].to_list())


class TestBaselineScaling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.epochs = use_sample_data()

    def test_1_channel_mean(self):
        epochs = self.epochs.copy()
        epochs.pick(0)
        # Trying on the first channel:
        control_channel = epochs.get_data()
        # Rescaling the data using the mne rescale function:
        control_data = rescale(control_channel, epochs.times, (None, 0), mode="mean", picks=None)
        # Doing the rescaling using my function:
        data_general_utilities.baseline_scaling(epochs, correction_method="mean", baseline=(None, 0),
                                                picks=None)
        test_data = epochs.get_data()
        # Assert whether that went well:
        assert_almost_equal(control_data, test_data)

    def test_1_channel_ratio(self):
        epochs = self.epochs.copy()
        epochs.pick(0)
        # Trying on the first channel:
        control_channel = epochs.get_data()
        # Rescaling the data using the mne rescale function:
        control_data = rescale(control_channel, epochs.times, (None, 0),
                               mode="ratio", picks=None)
        # Doing the rescaling using my function:
        data_general_utilities.baseline_scaling(epochs, correction_method="ratio",
                                                baseline=(None, 0),
                                                picks=None)
        test_data = epochs.get_data()
        # Assert whether that went well:
        assert_almost_equal(control_data, test_data)

    def test_5_channel_mean(self):
        epochs = self.epochs.copy()
        picks = np.random.choice(range(len(epochs.ch_names)), 5, replace=False)
        epochs.pick(picks)
        # Trying on the first channel:
        control_channel = epochs.get_data()
        # Rescaling the data using the mne rescale function:
        control_data = rescale(control_channel, epochs.times, (None, 0),
                               mode="mean", picks=None)
        # Doing the rescaling using my function:
        data_general_utilities.baseline_scaling(epochs, correction_method="mean",
                                                baseline=(None, 0),
                                                picks=None)
        test_data = epochs.get_data()
        # Assert whether that went well:
        assert_almost_equal(control_data, test_data)

    def test_5_channel_ratio(self):
        epochs = self.epochs.copy()
        picks = np.random.choice(range(len(epochs.ch_names)), 5, replace=False)
        epochs.pick(picks)
        # Trying on the first channel:
        control_channel = epochs.get_data()
        # Rescaling the data using the mne rescale function:
        control_data = rescale(control_channel, epochs.times, (None, 0),
                               mode="ratio", picks=None)
        # Doing the rescaling using my function:
        data_general_utilities.baseline_scaling(epochs, correction_method="ratio",
                                                baseline=(None, 0),
                                                picks=None)
        test_data = epochs.get_data()
        # Assert whether that went well:
        assert_almost_equal(control_data, test_data)


class TestFindOutliers(unittest.TestCase):

    def test_no_outlier(self):
        # Generate a vector of 5 zeros:
        data = np.zeros([5])
        data[:] = 1
        # Check for outliers:
        outlier_idx = data_general_utilities.find_outliers(data, m=4., func="median")
        # Assert whether that went well:
        self.assertEqual(len(outlier_idx), 0)

    def test_many_outliers(self):
        # Looping 10 times to see if we have a problem with random numbers:
        for i in range(10):
            # Generate a vector of 5 zeros:
            data = np.random.rand(100)
            data_mean = np.median(data, axis=0)
            # Compute the standard deviation
            data_std = np.std(data, axis=0)
            # Create two data points that are 5 times the std (not 4 because that would be on the edge):
            data = np.concatenate([data, [data_mean + 5 * data_std, data_mean - 5 * data_std]], axis=0)
            # Check for outliers:
            outlier_idx = data_general_utilities.find_outliers(data, m=4., func="median")
            # The outliers should be the two last point of the array:
            self.assertEqual(list(outlier_idx), [len(data) - 2, len(data) - 1])


class TestMovingAverage(unittest.TestCase):

    def test_1d_overlap(self):
        # Generate a vector of 5 zeros:
        data = np.array([1, 1, 2, 2, 3, 3])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=True)
        expected_output = np.array([1, 1.5, 2, 2.5, 3])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_2d_overlap(self):
        # Generate a vector of 5 zeros:
        data = np.array([[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=True)
        expected_output = np.array([[1, 1.5, 2, 2.5, 3], [4, 4.5, 5, 5.5, 6]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_3d_overlap(self):
        # Generate a vector of 5 zeros:
        data = np.array([[[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]], [[7, 7, 8, 8, 9, 9], [10, 10, 11, 11, 12, 12]]])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=True)
        expected_output = np.array([[[1, 1.5, 2, 2.5, 3], [4, 4.5, 5, 5.5, 6]],
                                    [[7, 7.5, 8, 8.5, 9], [10, 10.5, 11, 11.5, 12]]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_1d_bin(self):
        # Generate a vector of 5 zeros:
        data = np.array([1, 1, 2, 2, 3, 3])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=False)
        expected_output = np.array([1, 2, 3])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_2d_no_overlap(self):
        # Generate a vector of 5 zeros:
        data = np.array([[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=False)
        expected_output = np.array([[1, 2, 3], [4, 5, 6]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_3d_no_overlap(self):
        # Generate a vector of 5 zeros:
        data = np.array([[[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]], [[7, 7, 8, 8, 9, 9], [10, 10, 11, 11, 12, 12]]])
        window_size = 2
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=False)
        expected_output = np.array([[[1, 2, 3], [4, 5, 6]],
                                    [[7, 8, 9], [10, 11, 12]]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_3d_overlap_win_5(self):
        # Generate a vector of 5 zeros:
        data = np.array([[[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]], [[7, 7, 8, 8, 9, 9], [10, 10, 11, 11, 12, 12]]])
        window_size = 5
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=True)
        expected_output = np.array([[[1.8, 2.2], [4.8, 5.2]],
                                    [[7.8, 8.2], [10.8, 11.2]]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_3d_no_overlap_win_5(self):
        # Generate a vector of 5 zeros:
        data = np.array([[[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]], [[7, 7, 8, 8, 9, 9], [10, 10, 11, 11, 12, 12]]])
        window_size = 5
        # Check for outliers:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=False)
        expected_output = np.array([[[1.8], [4.8]],
                                    [[7.8], [10.8]]])
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)

    def test_comp_loop(self):
        # Generate an array of random numbers:
        data = np.random.rand(50, 20, 100)
        window_size = 5
        expected_output = np.zeros((data.shape[0], data.shape[1], int(data.shape[-1] / window_size)))
        for i in range(data.shape[0]):
            for ii in range(data.shape[1]):
                for ind, iii in enumerate(range(0, data.shape[-1], window_size)):
                    if iii == data.shape[-1]:
                        continue
                    expected_output[i, ii, ind] = np.mean(data[i, ii, iii:iii + 5])
        # Now, compute the moving average with the other function:
        observed_output = data_general_utilities.moving_average(data, window_size, axis=-1, overlapping=False)
        # Assert whether that went well:
        assert_almost_equal(observed_output, expected_output)


if __name__ == '__main__':
    unittest.main()
