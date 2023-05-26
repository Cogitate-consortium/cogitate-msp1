"""Handle subject information."""
import os
from pathlib import Path
import json


class SubjectInfo:
    """Container for subject information."""

    def __init__(self, subject_id, analysis_parameters, interactive=True):

        # --------------------------------------------------------------------------------------------------------------
        # Preparing the Subject info object:
        # --------------------------------------------------------------------------------------------------------------
        # If no subject ID was given, ask for input:
        if subject_id is None and interactive:
            self.SUBJ_ID = input("Enter subject ID (SX___): ")
        elif subject_id is None:
            raise Exception("You must pass a subject ID if you are not running in interactive mode!")
        else:
            self.SUBJ_ID = subject_id

        # Getting the information required from the analysis parameters file (session, task names and so on...)
        self.session = analysis_parameters.session
        self.task_name = analysis_parameters.task_name
        self.data_type = analysis_parameters.data_type

        # --------------------------------------------------------------
        # Preparing paths and files variables:
        # Setting the root to the raw:
        self.RAW_ROOT = analysis_parameters.raw_root
        self.BIDS_ROOT = analysis_parameters.BIDS_root
        # Generating the preprocessing root:
        self.preprocessing_root = str(
            Path(self.BIDS_ROOT, "derivatives", "preprocessing"))
        # Creating the participant info folder if it doesn't exists yet:
        if not os.path.isdir(self.preprocessing_root):
            # Creating the directory:
            os.makedirs(self.preprocessing_root)
        # Create the path to find the info files:
        if self.session is not None:
            self.participant_save_root = str(Path(self.preprocessing_root, "sub-"
                                                  + self.SUBJ_ID, "ses-" + self.session, self.data_type))
        else:
            self.participant_save_root = str(Path(self.preprocessing_root, "sub-"
                                                  + self.SUBJ_ID, self.data_type))
        # Creating the participant info folder if it doesn't exists yet:
        if not os.path.isdir(self.participant_save_root):
            # Creating the directory:
            os.makedirs(self.participant_save_root)
        # Creating the adequate file prefix string
        if self.session is not None:
            self.files_prefix = "sub-" + self.SUBJ_ID + "_ses-" + \
                self.session + "_task-" + self.task_name + "_desc-"
        else:
            self.files_prefix = "sub-" + self.SUBJ_ID + "_task-" + self.task_name + "_desc-"
        # Generate the json file name for this specific participant:
        self.json_file_name = self.files_prefix + "SubjectInfo.json"

        # --------------------------------------------------------------
        # Preparing and loading the participant json file:
        # Generating the path to the subject info file:
        self.participant_info_file = str(
            Path(self.participant_save_root, "info"))
        # Creating the participant info folder if it doesn't exists yet:
        if not os.path.isdir(self.participant_info_file):
            # Creating the directory:
            os.makedirs(self.participant_info_file)
        # Create the json file name:
        json_file = Path(self.participant_info_file, self.json_file_name)
        # Loading the content of existing participant info file:
        # If a json file exists for this participant, load it up and use it:
        if json_file.exists():
            print('')
            print('-' * 40)
            print(
                'An existing subject info json was found for this participant and will be used for the analysis')
            print('Loading:' + str(json_file))
            print('If you wish to modify the entries for this participant, you can either directly modify the json or '
                  'delete it and rerun the script')

            # Loading the json dict:
            with open(json_file) as f:
                json_dict = json.load(f)

            # Setting the subject ID:
            # If this is these infos are missing, the user needs to fill that in. Done only once for initialization:
            self.SUBJ_ID = json_dict['SUBJ_ID']

            # Setting the json file:
            self.json_file_name = self.files_prefix + "SubjectInfo.json"

            # Setting the photodiode triggers parameters:
            self.PD_THRESHOLD = float(json_dict['PD_THRESHOLD'])
            self.TRIGGER_CHANNEL = json_dict['TRIGGER_CHANNEL']
            self.TRIGGER_REF_CHANNEL = json_dict['TRIGGER_REF_CHANNEL']

            # EDF file name:
            self.EDF_FILE = json_dict['EDF_FILE']

            # Preallocating a variable for the bad channels rejection of different types:
            self.auto_bad_channels = []
            self.manu_bad_channels = []
            self.desc_bad_channels = {}

            # Calibration indices:
            self.start_inds_trigger_noise = json_dict["start_inds_trigger_noise"]
            self.end_inds_trigger_noise = json_dict["end_inds_trigger_noise"]

            # Finally, reading the annotated and clean_logs_file. Only populated after instantiation:
            self.annotated_signal_file_name = json_dict["annotated_signal_file_name"]
            self.clean_logs_file = json_dict["clean_logs_file"]

            # Handle interruption index. It might be that in some of the json files, this is not saved:
            try:
                self.interruption_index = json_dict["interruption_index"]
            except KeyError:
                self.interruption_index = None
        # If no existing json file are found for the given participant, more input is necessary to generate one:
        else:

            # Setting the photodiode triggers parameters:
            self.PD_THRESHOLD = 1
            if interactive:
                self.TRIGGER_CHANNEL = input(
                    "Input name of the photodiode trigger channel: ")
                self.TRIGGER_REF_CHANNEL = input(
                    "Input name of the reference for the  photodiode trigger channel (leave empty if none): ")
                self.EDF_FILE = input(
                    "Input name of the edf file (leave empty if you want to use all the files in your data "
                    "ECoG/Experiment1 dir): ")
            else:
                self.TRIGGER_CHANNEL = None
                self.TRIGGER_REF_CHANNEL = None
                self.EDF_FILE = None

            # Setting the bad channels:
            self.bad_channels = []

            # Calibration indices:
            self.start_inds_trigger_noise = []
            self.end_inds_trigger_noise = []

            # Finally, reading the annotated and clean_logs_file. Only populated after instantiation:
            self.annotated_signal_file_name = ""
            self.clean_logs_file = ""

            # Adding the placeholder for interruption marker:
            self.interruption_index = None

        if analysis_parameters.save_output:
            self.update_json()

    def add_bad_channels(self, bad_channel_name):
        """
        Add bad channels but making sure that they are not already there
        :param bad_channel_name (str)
        :return: None
        """
        for bad in bad_channel_name:
            if bad not in self.bad_channels:
                self.bad_channels.extend(bad_channel_name)

    def update_json(self):
        """
        This function saves the updated json:
        :return:
        """

        # Converting the object to dict:
        obj_dict = self.__dict__

        # Copying the object
        obj_dict_to_save = obj_dict.copy()

        with open(Path(self.participant_info_file, self.json_file_name), "w") \
                as fp:
            json.dump(obj_dict_to_save, fp, indent=4)

    def save(self, file_path):
        # Converting the object to dict:
        obj_dict = self.__dict__.copy()

        # Setting the full file name:
        full_file = Path(file_path, self.json_file_name)

        with open(full_file, "w") as fp:
            json.dump(obj_dict, fp, indent=4)
