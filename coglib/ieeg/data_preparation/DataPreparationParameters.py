"""Handle parameters for data preparation."""
from pathlib import Path
import json


class DataPreparationParameters:
    """Container class for data preparation parameters."""

    def __init__(self, init_file):

        with open(init_file) as json_file:
            params = json.load(json_file)

        # ---------------------------------------------------------------------
        # Raw data path:
        # Setting the raw root:
        self.raw_root = params['raw_root']
        self.ecog_files_naming = params['ecog_files_naming']
        self.ecog_files_extension = params['ecog_files_extension']
        self.beh_files_naming = params['beh_files_naming']
        self.beh_files_extension = params['beh_files_extension']
        self.elec_loc = params['elec_loc']
        self.elec_loc_root = params['elec_loc_root']
        self.elec_loc_extension = params['elec_loc_extension']
        self.additional_channel_conv = params['additional_channel_conv']
        self.BIDS_root = params['BIDS_root']
        self.channels_description_file = params["channels_description_file"]

        # ---------------------------------------------------------------------
        # Experiment parameters:
        self.session = params['session']
        self.task_name = params['task_name']
        self.data_type = params['data_type']
        # Refresh rate of the screen:
        self.ref_rate_ms = params['ref_rate_ms']
        self.line_freq = params['line_freq']

        # ---------------------------------------------------------------------
        # Misc:
        self.HPC = params['HPC']
        self.debug = params['debug']
        self.show_check_plots = params['show_check_plots']
        self.save_output = False

    def save(self, file_path, file_prefix):
        """Save the data preparation parameters into json file."""
        # Converting the object to dict:
        obj_dict = self.__dict__

        # Setting the full file name:
        full_file = Path(file_path, file_prefix
                         + "Data_preparation_parameters.json")

        with open(full_file, "w") as fp:
            json.dump(obj_dict, fp, indent=4)
