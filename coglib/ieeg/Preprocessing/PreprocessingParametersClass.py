"""Handling preprocessing parameters."""
from pathlib import Path
import json
import subprocess


class PreprocessingParameters:
    """Container class for preprocessing parameters."""

    def __init__(self, init_file):

        with open(init_file) as json_file:
            params = json.load(json_file)

        # First things first, adding the git id and url to keep track of things.
        # But if git is not installed, unknown:
        try:
            self.github_repo_url = str(subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url']))
            self.git_commit_id = str(
                subprocess.check_output(['git', 'rev-parse', 'HEAD']))
        except FileNotFoundError:
            self.github_repo_url = "unknown"
            self.git_commit_id = "unknown"

        # ---------------------------------------------------------------------
        # Raw data path:
        # Setting the raw root:
        self.raw_root = params['raw_root']
        self.BIDS_root = params['BIDS_root']
        self.fs_dir = params['fs_dir']

        # ---------------------------------------------------------------------
        # Experiment parameters:
        self.session = params['session']
        self.task_name = params['task_name']
        self.data_type = params["data_type"]
        # Refresh rate of the screen:
        self.ref_rate_ms = params['ref_rate_ms']
        self.montage_space = params["montage_space"]
        self.interuption_landmark = params["interuption_landmark"]

        # ---------------------------------------------------------------------
        # Preprocessing paramaters:
        # List of preprocessing parameters to follow:
        self.preprocessing_steps = params['preprocessing_steps']

        # Parameters for each of the supported preprocessing steps:
        self.atlas_mapping = params['atlas_mapping']
        self.notch_filtering = params['notch_filtering']
        self.automated_bad_channels_rejection = params['automated_bad_channels_rejection']
        self.manual_bad_channels_rejection = params['manual_bad_channels_rejection']
        self.description_bad_channels_rejection = params['description_bad_channels_rejection']
        self.car = params['car']
        self.laplace_reference = params['laplace_reference']
        self.epoching = params['epoching']
        self.manual_artifact_detection = params['manual_artifact_detection']
        self.automated_artifact_detection = params['automated_artifact_detection']
        self.frequency_bands_computations = params['frequency_bands_computations']
        self.erp_computations = params['erp_computations']
        self.plot_epochs = params['plot_epochs']

        # ----------------------------------------------------------------------------------------------------------
        # General controls of the behavior of the preprocessing scripts:
        self.show_check_plots = params['show_check_plots']
        # Control computations and saving. Njobs control how many parallel processes you can use
        self.njobs = params['njobs']
        # If debug is set to true, you will only load 4 channels to do things super quick
        self.debug = params['debug']
        # If set to true, all the preprocessing steps will save data to fif files
        self.save_intermediary_steps = params['save_intermediary_steps']
        self.save_output = params['save_output']

        # Set the analysis performed as you go:
        self.analyses_performed = []

    def add_performed_analysis(self, analysis):
        """
        This function adds an analysis that has been carried out so that we can
        keep track of which analyses has been carried out.
        :param: analysis: str
        """

        self.analyses_performed.append(analysis)

    def save(self, file_path, file_prefix):
        """Save the preprocessing parameters into a json file."""
        # Converting the object to dict:
        obj_dict = self.__dict__

        # Setting the full file name:
        full_file = Path(file_path, file_prefix
                         + "Preprocessing_parameters.json")

        with open(full_file, "w") as fp:
            json.dump(obj_dict, fp, indent=4)
