import unittest
import numpy as np
import mne
from scipy.stats import truncnorm
import math
from Preprocessing.PreprocessingHelperFunctions import *
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids


def create_mne_info(sfreq, channel_number_per_types, ch_types=None):
    """
    This function creates an mne info object, that has an equal number of ECOG and seeg channels + n bad ecog channels.
    The channel_number_per_types is an integer controlling how many channels / type. The channels get named ECOG +
    channel number and so on
    :param sfreq: sampling frequency of the signal
    :param channel_number_per_types: number of channels per type. So if you set it to 10, you will have 10 seeg and 10
    :param ch_types: type of channels to create. Default is both seeg and ECoG
    ecog channels + 10 bad ecog channels
    :return: mne info object
    """
    if ch_types is None:
        ch_types = ['seeg', 'ecog']

    ch_names = []
    ch_kind = []
    if isinstance(ch_types, list):
        for ch_type in ch_types:
            ch_names += [ch_type.upper() + str(n) for n in range(1, channel_number_per_types + 1)]
            ch_kind += [ch_type] * channel_number_per_types
    else:
        ch_names += [ch_types.upper() + str(n) for n in range(1, channel_number_per_types + 1)]
        ch_kind += [ch_types] * channel_number_per_types

    # Adding n bad channels:
    ch_names += ["BAD" + str(n) for n in range(1, channel_number_per_types + 1)]
    ch_kind += ['ecog'] * channel_number_per_types

    # Create the info object
    info = mne.create_info(ch_names, ch_types=ch_kind, sfreq=sfreq)

    # Add the bad channels:
    info['bads'] = [f'BAD{n}' for n in range(1, channel_number_per_types + 1)]

    return info


class TestRemoveChannelTsvDescription(unittest.TestCase):

    def test_one_desc(self):
        """

        :return:
        """
        # Generate test data:
        test_df = pd.DataFrame({
            "name": ["G1", "G2", "G3", "G4", "G5"],
            "status_description": ["bad_1", float("nan"), "bad_1", float("nan"), "bad_1"]
        })

        # Run the script:
        test_df = remove_channel_tsv_description(test_df, "bad_1")

        # Get the test list:
        obs_list = test_df["status_description"].to_list()
        exp_list = ["", float("nan"), "", float("nan"), ""]
        # Converting the nan to 0 as otherwise, comparing nan leads to inequality:
        obs_list = ["nan" if not isinstance(x, str) and math.isnan(x) else x for x in obs_list]
        exp_list = ["nan" if not isinstance(x, str) and math.isnan(x) else x for x in exp_list]
        self.assertEqual(obs_list, exp_list)


    def test_several_desc(self):
        """

        :return:
        """
        # Generate test data:
        test_df = pd.DataFrame({
            "name": ["G1", "G2", "G3", "G4", "G5"],
            "status_description": ["bad_1/bad_2", float("nan"), "bad_1", float("nan"), "bad_1/bad_2"]
        })

        # Run the script:
        test_df = remove_channel_tsv_description(test_df, "bad_1")

        # Get the test list:
        obs_list = test_df["status_description"].to_list()
        exp_list = ["bad_2", float("nan"), "", float("nan"), "bad_2"]
        # Converting the nan to 0 as otherwise, comparing nan leads to inequality:
        obs_list = ["nan" if not isinstance(x, str) and math.isnan(x) else x for x in obs_list]
        exp_list = ["nan" if not isinstance(x, str) and math.isnan(x) else x for x in exp_list]
        self.assertEqual(obs_list, exp_list)


class TestAutomatedBadChannelDetection(unittest.TestCase):

    def test_ecog(self):
        """

        :return:
        """
        # Load mne ecog data set:
        bids_root = mne.datasets.epilepsy_ecog.data_path()
        sample_path = mne.datasets.sample.data_path()
        # first define the bids path
        bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                             task='ictal', datatype='ieeg', extension='vhdr')
        # Load the raw:
        raw = read_raw_bids(bids_path=bids_path, verbose=False).load_data()

        # Run the script:
        raw, detected_bad_channels = automated_bad_channel_detection(raw, epoch_length=1.0, max_range_muv=2000,
                                                                     min_range_muv=10, max_slope_muvsec=100,
                                                                     segment_proportion_cutoff=0.5,
                                                                     channel_types={"ecog": True},
                                                                     reject_bad_channels=False)

    def test_seeg(self):
        """

        :return:
        """
        # Load mne ecog data set:
        misc_path = mne.datasets.misc.data_path()
        raw = mne.io.read_raw(Path(misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
        # Run the script:
        raw, detected_bad_channels = automated_bad_channel_detection(raw, epoch_length=1.0, max_range_muv=2000,
                                                                     min_range_muv=10, max_slope_muvsec=100,
                                                                     segment_proportion_cutoff=0.1,
                                                                     channel_types={"seeg": True},
                                                                     reject_bad_channels=False)


class TestCustomCar(unittest.TestCase):

    def test_one_elect_type_ones_zeros(self):
        """
        This test tests that the common average referencing works well on single electrodes types. Common average
        referencing computes the mean across channels and subtract it to each data point. Therefore, I generate data
        points that average to a specific value and make sure that the output makes sense
        :return:
        """
        # Create infos:
        info = create_mne_info(1000, 2, ch_types='ecog')
        # Generate an array of ones and zeros, plus crazy numbers for the bad channels:
        data = np.array([np.ones(10), np.zeros(10), np.zeros(10) + 10000, np.zeros(10) + 200])
        # Generating the expected output data of the function. The function must take the average across the first and
        # second channel, which should be 0.5 and subtract it to both these channels. Therefore, the output data should
        # look like:
        output_data_expected = np.array([np.ones(10) - 0.5, np.zeros(10) - 0.5,
                                         np.zeros(10) + 10000, np.zeros(10) + 200])
        # Generate mne object based on this:
        simulated_raw = mne.io.RawArray(data, info)
        # Run the common average referencing:
        custom_car(simulated_raw, ecog=True)
        # Extracting the data from the modified simulated raw object:
        output_data_observed = simulated_raw.get_data()

        # Asserting equality:
        self.assertTrue((output_data_observed == output_data_expected).all(),
                        "Common average fails on single channel type with ones and zeros")

    def test_two_elect_type_ones_zeros(self):
        """
        The idea is the same as above, but this time with channels of different types. We want our common average
        referencing to work across channel types
        :return:
        """
        # Create infos:
        info = create_mne_info(1000, 1)
        # Generate an array of ones and zeros for each channel type, plus crazy numbers for the bad channels:
        data = np.array([np.ones(10), np.zeros(10), np.zeros(10) + 10000])
        # Generating the expected output data of the function. The function must take the average across the first and
        # second channel, which should be 0.5 and subtract it to both these channels. Therefore, the output data should
        # look like:
        output_data_expected = np.array([np.ones(10) - 0.5, np.zeros(10) - 0.5,
                                         np.zeros(10) + 10000])
        # Generate mne object based on this:
        simulated_raw = mne.io.RawArray(data, info)
        # Run the common average referencing:
        custom_car(simulated_raw, ecog=True, seeg=True)
        # Extracting the data from the modified simulated raw object:
        output_data_observed = simulated_raw.get_data()

        # Asserting equality:
        self.assertTrue((output_data_observed == output_data_expected).all(),
                        "Common average fails on ecog and seeg channels type with ones and zeros")

    def test_one_elect_type_random_numbers(self):
        """
        The test above are limited to ones and zero. To encompass more options, generating random numbers, computing
        the average by other means and subtracting it from each data point to estimate output
        :return:
        """
        # Create infos:
        info = create_mne_info(1000, 2, ch_types='ecog')
        # Generate an array of ones and zeros, plus crazy numbers for the bad channels:
        data = np.array([np.random.rand(10) * 10, np.random.rand(10), np.zeros(10) + 10000, np.zeros(10) + 200])
        # The function should subtract the average across the two channels that aren't bad and subtract it from each
        # data points of the good channels. Therefore, I first compute the mean:
        average_reference = np.mean(data[0:2, :], axis=0)
        # Subtracting the common average for the two first channels:
        output_data_expected = np.array([data[0, :] - average_reference, data[1, :] - average_reference,
                                         np.zeros(10) + 10000, np.zeros(10) + 200])
        # Generate mne object based on this:
        simulated_raw = mne.io.RawArray(data, info)
        # Run the common average referencing:
        custom_car(simulated_raw, ecog=True)
        # Extracting the data from the modified simulated raw object:
        output_data_observed = simulated_raw.get_data()

        # Asserting equality:
        self.assertTrue((output_data_observed == output_data_expected).all(),
                        "Common average fails on single channel type with random numbers")

    def test_two_elect_type_random_numbers(self):
        """
        The idea is the same as above, but this time with channels of different types. We want our common average
        referencing to work across channel types
        :return:
        """
        # Create infos:
        info = create_mne_info(1000, 1)
        # Generate an array of ones and zeros, plus crazy numbers for the bad channels:
        data = np.array([np.random.rand(10) * 10, np.random.rand(10), np.zeros(10) + 10000])
        # The function should subtract the average across the two channels that aren't bad and subtract it from each
        # data points of the good channels. Therefore, I first compute the mean:
        average_reference = np.mean(data[0:2, :], axis=0)
        # Subtracting the common average for the two first channels:
        output_data_expected = np.array([data[0, :] - average_reference, data[1, :] - average_reference,
                                         np.zeros(10) + 10000])
        # Generate mne object based on this:
        simulated_raw = mne.io.RawArray(data, info)
        # Run the common average referencing:
        custom_car(simulated_raw, ecog=True, seeg=True)
        # Extracting the data from the modified simulated raw object:
        output_data_observed = simulated_raw.get_data()

        # Asserting equality:
        self.assertTrue((output_data_observed == output_data_expected).all(),
                        "Common average fails on seeg and ecog channel types type with random numbers")

    def test_many_elect_type_random_numbers(self):
        """
        The idea is the same as above, but this time with channels of different types. We want our common average
        referencing to work across channel types
        :return:
        """
        # Create infos:
        info = create_mne_info(1000, 10)

        # Generating random data:
        data = np.zeros([len(info.ch_names), 10])
        for elec_ind, elec in enumerate(info.ch_names):
            multiplier = int(np.random.rand(1) * 100)
            data[elec_ind, :] = np.random.rand(data.shape[1]) * multiplier

        # The function should subtract the average across the two channels that aren't bad and subtract it from each
        # data points of the good channels. Therefore, I first compute the mean:
        # Getting the indices of the good channels:
        good_chan = list(range(0, int(len(info.ch_names) * 2 / 3)))
        average_reference = np.mean(data[good_chan, :], axis=0)
        # Creating the expected output as the original data with the good channels - commong average, bad channels
        # intact:
        output_data_expected = np.concatenate([data[good_chan, :] - average_reference,
                                               data[good_chan[-1] + 1: data.shape[0], :]])

        # Generate mne object based on this:
        simulated_raw = mne.io.RawArray(data, info)
        # Run the common average referencing:
        custom_car(simulated_raw, ecog=True, seeg=True)
        # Extracting the data from the modified simulated raw object:
        output_data_observed = simulated_raw.get_data()

        # Asserting equality:
        self.assertTrue((output_data_observed == output_data_expected).all(),
                        "Common average fails on seeg and ecog channel type with many channels!")


class TestLaplaceReference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create the info for the channels objects:
        sfreq = 100
        info = mne.create_info(ch_names=['G1', 'G2', 'G3', 'G4', 'G5'],
                               ch_types=['ecog'] * 5,
                               sfreq=sfreq)
        # Create an epochs only made of ones:
        data = np.squeeze(np.array([np.zeros((1, sfreq)) + i for i in range(len(info["ch_names"]))]))
        cls.raw = mne.io.RawArray(data, info)
        # Create the laplacian mapping:
        cls.reference_mapping = {
            "G1": {"ref_1": "G2", "ref_2": None},
            "G2": {"ref_1": "G1", "ref_2": "G3"},
            "G3": {"ref_1": "G2", "ref_2": "G4"},
            "G4": {"ref_1": "G3", "ref_2": "G5"},
            "G5": {"ref_1": "G4", "ref_2": None},
        }

    def test_validate_laplace_mapping(self):

        # Creating different wrong mapping (i.e. with channels that don't exist):
        wrong_mapping_1 = {
            "G1": {"ref_1": "G2", "ref_2": None},
            "G2": {"ref_1": "G1", "ref_2": "G2"},
            "G3": {"ref_1": "G2", "ref_2": "G4"},
            "G4": {"ref_1": "G3", "ref_2": "G5"},
            "G6": {"ref_1": "G4", "ref_2": None},
        }
        wrong_mapping_2 = {
            "G1": {"ref_1": "G9", "ref_2": None},
            "G2": {"ref_1": "G1", "ref_2": "G2"},
            "G3": {"ref_1": "G2", "ref_2": "G4"},
            "G4": {"ref_1": "G3", "ref_2": "G5"},
            "G6": {"ref_1": "G4", "ref_2": None},
        }
        wrong_mapping_3 = {
            "G1": {"ref_1": "G2", "ref_2": "G23"},
            "G2": {"ref_1": "G1", "ref_2": "G2"},
            "G3": {"ref_1": "G2", "ref_2": "G4"},
            "G4": {"ref_1": "G3", "ref_2": "G5"},
            "G6": {"ref_1": "G4", "ref_2": None},
        }

        self.assertRaises(Exception, laplace_mapping_validator, wrong_mapping_1, self.raw.info["ch_names"])
        self.assertRaises(Exception, laplace_mapping_validator, wrong_mapping_2, self.raw.info["ch_names"])
        self.assertRaises(Exception, laplace_mapping_validator, wrong_mapping_3, self.raw.info["ch_names"])
        laplace_mapping_validator(self.reference_mapping, self.raw.info["ch_names"])

    def test_remove_bad_references(self):
        # Making a copy of the epochs to set bad channels:
        test_epochs_1 = self.raw.copy()
        test_epochs_1.info["bads"] = ["G1", "G3"]
        expected_bad_channels = ["G1", "G2", "G3"]
        expected_reference_mapping = {
            "G4": {"ref_1": None, "ref_2": "G5"},
            "G5": {"ref_1": "G4", "ref_2": None}
        }
        # Running the bad channels:
        reference_mapping, bad_channels = remove_bad_references(
            self.reference_mapping, test_epochs_1.info["bads"],
            self.raw.ch_names)
        # Comparing the output with the expected output:
        self.assertTrue(all([ch in expected_bad_channels for ch in bad_channels]))
        self.assertDictEqual(expected_reference_mapping, reference_mapping)

    def test_laplace_ref_fun_1d(self):
        # Generating simple data:
        to_ref = np.zeros((0, 100))
        ref_1 = to_ref + 1
        ref_2 = to_ref + 2
        ref_data = laplace_ref_fun(to_ref, ref_1=ref_1, ref_2=ref_2)
        expected_output = to_ref - 1.5
        self.assertTrue(np.array_equal(expected_output, ref_data, equal_nan=True))

    def test_laplace_ref_fun_nd(self):
        # Generating data wit more dim:
        to_ref = np.squeeze(np.array([np.zeros((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))]))
        ref_1 = np.squeeze(np.array([np.zeros((1, 100)) + 1, np.zeros((1, 100)) + 1, np.zeros((1, 100)) - 1]))
        ref_2 = np.squeeze(np.array([np.zeros((1, 100)) + 2, np.zeros((1, 100)) + 3, np.zeros((1, 100)) - 3]))
        ref_data = laplace_ref_fun(to_ref, ref_1=ref_1, ref_2=ref_2)
        expected_output = np.squeeze(np.array([np.zeros((1, 100)) - 1.5, np.zeros((1, 100)) - 2,
                                               np.zeros((1, 100)) + 2]))
        self.assertTrue(np.array_equal(expected_output, ref_data, equal_nan=True))

    def test_laplace_ref_fun_nan(self):
        # Generating data wit more dim:
        to_ref = np.squeeze(np.array([np.zeros((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))]))
        ref_1 = np.squeeze(np.array([np.zeros((1, 100)) + 1, np.zeros((1, 100)) + 1, np.zeros((1, 100)) - 1]))
        ref_2 = np.squeeze(np.array([np.zeros((1, 100)) + 2, np.zeros((1, 100)) + 3, np.empty((1, 100))]))
        ref_2[2, :] = np.nan
        ref_data = laplace_ref_fun(to_ref, ref_1=ref_1, ref_2=ref_2)
        expected_output = np.squeeze(np.array([np.zeros((1, 100)) - 1.5, np.zeros((1, 100)) - 2,
                                               np.zeros((1, 100)) + 1]))
        self.assertTrue(np.array_equal(expected_output, ref_data, equal_nan=True))

    def test_laplacian_referencing(self):
        # Generate the expected output through alternative computation:
        data_to_ref = self.raw.get_data(picks=list(self.reference_mapping.keys()))
        # Looping through the channels to reference:
        ref_1_data = []
        ref_2_data = []
        for ch in self.reference_mapping.keys():
            if self.reference_mapping[ch]["ref_1"] is None:
                empty_mat = np.empty((1, data_to_ref.shape[1]))
                empty_mat[:] = np.nan
                ref_1_data.append(empty_mat)
            else:
                ref_1_data.append(self.raw.get_data(picks=[self.reference_mapping[ch]["ref_1"]]))
            if self.reference_mapping[ch]["ref_2"] is None:
                empty_mat = np.empty((1, data_to_ref.shape[1]))
                empty_mat[:] = np.nan
                ref_2_data.append(empty_mat)
            else:
                ref_2_data.append(self.raw.get_data(picks=[self.reference_mapping[ch]["ref_2"]]))
        ref_1_data = np.squeeze(np.array(ref_1_data))
        ref_2_data = np.squeeze(np.array(ref_2_data))
        # Use the laplace_ref_fun because it was validated above:
        ref_data = laplace_ref_fun(data_to_ref, ref_1=ref_1_data, ref_2=ref_2_data)
        # Running the pipeline
        raw, reference_mapping, bad_channels = \
            laplacian_referencing(self.raw, reference_mapping=self.reference_mapping, n_jobs=1)
        # Comparing what the pipeline gives compared to what's expected:
        self.assertTrue(np.array_equal(raw.get_data(), ref_data, equal_nan=True))

    def test_laplacian_referencing_bads(self):
        # Checking if things work fine when adding bad channels:
        self.raw.info["bads"] = ["G1"]
        reference_mapping, bad_channels = remove_bad_references(self.reference_mapping,
                                                                self.raw.info["bads"],
                                                                self.raw.ch_names)
        # Generate the expected output through alternative computation:
        data_to_ref = self.raw.get_data(picks=list(reference_mapping.keys()))
        # Looping through the channels to reference:
        ref_1_data = []
        ref_2_data = []
        for ch in reference_mapping.keys():
            if reference_mapping[ch]["ref_1"] is None:
                empty_mat = np.empty((1, data_to_ref.shape[1]))
                empty_mat[:] = np.nan
                ref_1_data.append(empty_mat)
            else:
                ref_1_data.append(self.raw.get_data(picks=[reference_mapping[ch]["ref_1"]]))
            if reference_mapping[ch]["ref_2"] is None:
                empty_mat = np.empty((1, data_to_ref.shape[1]))
                empty_mat[:] = np.nan
                ref_2_data.append(empty_mat)
            else:
                ref_2_data.append(self.raw.get_data(picks=[reference_mapping[ch]["ref_2"]]))
        ref_1_data = np.squeeze(np.array(ref_1_data))
        ref_2_data = np.squeeze(np.array(ref_2_data))
        ref_data = laplace_ref_fun(data_to_ref, ref_1=ref_1_data, ref_2=ref_2_data)
        # Running the pipeline
        raw, reference_mapping, bad_channels = \
            laplacian_referencing(self.raw, reference_mapping=self.reference_mapping, n_jobs=1)
        # Comparing what the pipeline gives compared to what's expected:
        observed_data = raw.get_data(picks=[ch for ch in raw.ch_names if ch not in raw.info["bads"]])
        self.assertTrue(np.array_equal(observed_data, ref_data, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
