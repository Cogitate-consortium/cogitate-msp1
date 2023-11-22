# -*- coding: utf-8 -*-
"""
# ====================
# 00. BIDS convertion
# ====================

https://mne.tools/mne-bids/stable/index.html

Questions/Issues:
    - what to write in the participants and dataset_description metadata
    files?
    - what session ID should we give to the anat scan (e.g., v0, v2, mri)?
    - for visit 2, what to count the replay runs? Continue the count from
    where it was left from the VG (run 4) or restart from run 1?

Notes:
    - the conversion must be done after reading the events. Here, the event
    list includes all the triggers/events
    - the T1 scan can be added under 'anat' together with the transformation
    matrix obtained from the coregistraction. T1s can be defaced at this stage.
    - participant info must be updated manually (e.g., age, sex)
    - datatype can be 'meg', 'eeg', or a few options (no meeg). Moreover,
    fif files are automatically read as 'meg' datatype and this cannot
    be overwritten by the datatype option. Concurrent MEG-EEG data type is MEG
    - For the anat conversion, you need to run FreeSurfer and complete the
    co-registration step first.
    - BIDS does not allow DICOM scans. NIfTI conversion is required.

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import argparse

import mne
from mne_bids import (write_raw_bids, write_meg_calibration,
                      write_meg_crosstalk, BIDSPath, write_anat)
# from mne_bids import (print_dir_tree, make_report, write_anat)
# from mne_bids.write import get_anat_landmarks
# from mne_bids.stats import count_events

# import dicom2nifti  # conda install -c conda-forge dicom2nifti

# from config import (subject_list, file_exts, raw_path, cal_path, t1_path,
#                     bids_root)


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CA101',
                    help='site_id + subject_id (e.g. "CA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--in_raw',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/Raw/projects/CoG_MEG_PhaseII',
                    help='Path to the RAW data directory')
# RAW: /mnt/beegfs/XNAT/COGITATE/MEG/Raw/projects/CoG_MEG_PhaseII
# MEG: /mnt/beegfs/XNAT/COGITATE/MEG/Raw/projects/CoG_MEG_PhaseII/CA101/CA101_MEEG_V1/SCANS/DurR1/FIF/CA101_MEEG_V1_DurR1.fif
# T1:  /mnt/beegfs/XNAT/COGITATE/MEG/Raw/projects/CoG_MEG_PhaseII/CA101/CA101_MR_V0/SCANS/5/DICOM/xxx.dcm
parser.add_argument('--in_cal',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/preprocessing/cal_files',
                    help='Path to the fine-calibration and cross-talk files')
parser.add_argument('--out_bids',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
                    help='Path to the BIDS root directory')
opt=parser.parse_args()


def init_par(opt):  #subject_id, visit_id, t1_path, file_exts, bids_root
    # Prepare BIDS fields
    bids = {}
    bids["root"] = opt.out_bids
    bids["subject"] = opt.sub
    bids["site"] = bids["subject"][0:2]
    bids["session"] = opt.visit
    bids["datatype"] = 'meg'

    # Set MEG data path and name
    if bids["session"] == 'V1':
        file_exts = ['%s_MEEG_V1_DurR1',
                     '%s_MEEG_V1_DurR2',
                     '%s_MEEG_V1_DurR3',
                     '%s_MEEG_V1_DurR4',
                     '%s_MEEG_V1_DurR5']
    elif bids["session"] == 'V2':
        file_exts = ['%s_MEEG_V2_VGR1',
                     '%s_MEEG_V2_VGR2',
                     '%s_MEEG_V2_VGR3',
                     '%s_MEEG_V2_VGR4',
                     '%s_MEEG_V2_ReplayR1',
                     '%s_MEEG_V2_ReplayR2']
    meg = {}
    meg["subject"] = bids["subject"]
    meg["fnames"] = [f % bids["subject"] for f in file_exts]
    meg["data_path"] = op.join(opt.in_raw, bids["subject"],
                               bids["subject"]+"_MEEG_"+bids["session"],
                               "SCANS",
                               "%s",  #task+run (e.g., DurR1)
                               "FIF",
                               )
    meg["cal_path"] = op.join(opt.in_cal, bids["site"])

    # Set anat MRI path and names
    t1 ={}
    t1["subject"] = bids["subject"]
    t1["fname"] = bids["subject"] + '_MR_V0_anat'
    t1["nifti_path"] = op.join(bids["root"],
                         "derivatives",
                         "dicom2nifti",
                         bids["subject"],
                         )  #TODO
    t1["dicom_path"] = op.join(opt.in_raw, bids["subject"],
                               bids["subject"]+"_MR_V0",
                               "SCANS",
                               "%s",  #task+run (e.g., DurR1)
                               "5",  #TODO: what is this folder?
                               "DICOM",
                               )
    t1["fs_path"] = op.join(t1["nifti_path"], "fs")  #TODO
    t1["mgz_path"] = op.join(t1["fs_path"], bids["subject"], 'mri', 'T1.mgz')  #TODO
    t1["trans_path"] = op.join(t1["fs_path"], bids["subject"]+"-trans.fif")  #TODO

    return bids, meg, t1

def raw_to_bids(bids, meg):
    for file_name in meg["fnames"]:
        run = file_name[-1]

        # Set task
        if 'Dur' in file_name:
            bids["task"] = 'Dur'
        elif 'VG' in file_name:
            bids["task"] = 'VG'
        elif 'Replay' in file_name:
            bids["task"] = 'Replay'
        else:
            raise ValueError("Error: could not find the task for %s" % file_name)

        # Read raw
        raw_fname = op.join(meg["data_path"] % (bids["task"]+'R'+run), file_name + '.fif')
        raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True)

        # Read events
        # events_data = op.join(meg["out_path"],
        # meg["fnames"][0]+'-bids_eve.fif')

        events = mne.find_events(raw,
                                 stim_channel='STI101',
                                 consecutive = True,
                                 min_duration=0.001001,
                                 mask = 65280,
                                 mask_type = 'not_and',
                                 verbose=True)

        # Set event IDs
        if bids["session"] == 'V1':
            # Stimulus type and image ID
            stimulus_id = {}
            types = ['face','object','letter','false']
            for j,t in enumerate(types):
                for i in range(1,21):
                    stimulus_id[t+f'{i:02d}'] = i + j * 20
            # Trial number (block-wise)
            trial_id = {}
            for i in range(111,149):
                trial_id['trial'+f'{i-110:02d}'] = i
            # Sequence number
            sequence_id = {}
            for i in range(161,201):
                sequence_id['sequence'+f'{i-160:02d}'] = i
            # Other events
            other_id = {'onset of  recording':81, 'offset of recording':83,
                        'start experiment': 86, 'stimulus offset':96,
                        'blank offset':97,
                        'center': 101, 'left':102, 'right':103,
                        '500ms': 151, '1000ms': 152, '1500ms': 153,
                        'task relevant target': 201,
                        'task relevant non target': 202,
                        'task irrelevant': 203, 'response': 255}
            # Merge all event IDs in event_id
            event_id = stimulus_id | trial_id | sequence_id | other_id

        elif bids["session"] == "V2":
            # Stimulus type and image ID during the game
            stimulus_id = {}
            stimulus_id['blank'] = 50
            types = ['face','object']
            for j,t in enumerate(types):
                for i in range(1,11):
                    stimulus_id[t+f'{i:02d}'] = i + j * 20
            # Stimulus type and image ID during the replay
            stimulus_id['blanks during face target'] = 150
            stimulus_id['blanks during object target'] = 250
            types = ['face target','object non-target']
            for j,t in enumerate(types):
                for i in range(1,11):
                    stimulus_id[t+f'{i:02d}'] = (i + j * 20) + 100
            types = ['face non-target','object target']
            for j,t in enumerate(types):
                for i in range(1,11):
                    stimulus_id[t+f'{i:02d}'] = (i + j * 20) + 200
            # Stimulus location
            location_id = {}
            location_id['upper left'] = 60
            location_id['upper left probed or target'] = 61
            location_id['upper right'] = 70
            location_id['upper right probed or target'] = 71
            location_id['lower right'] = 80
            location_id['lower right probed or target'] = 81
            location_id['lower left'] = 90
            location_id['lower left probed or target'] = 91
            # Response
            response_id = {}
            response_id['seen'] = 98
            response_id['unseen'] = 99
            response_id['response during replay'] = 198
            response_id['end replay response window'] = 196
            # Other events
            other_id = {'probe onset': 100, 'filler':95,
                        'filler during replay':195,
                        'level begin': 251, 'level end':252,
                        'animation peak end':253}

            # Merge all event IDs in event_id
            event_id = stimulus_id | location_id | response_id | other_id

        # Set BIDS path
        bids_path = BIDSPath(subject=bids["subject"],
                             session=bids["session"],
                             task=bids["task"].lower(),
                             run='0'+run,
                             datatype=bids["datatype"],
                             root=bids["root"])
        # Write BIDS
        write_raw_bids(raw,
                       bids_path=bids_path,
                       events_data=events,
                       event_id=event_id,
                       overwrite=True)

    return raw

def rest_to_bids(bids, meg):
    # Add resting state data  #TODO: declare that it is 5-min eyes open RS
    rs_raw_fname = op.join(meg["data_path"] % "RestinEO", bids["subject"] + '_MEEG_' + bids["session"] + '_RestinEO.fif')
    rs_raw = mne.io.read_raw_fif(rs_raw_fname, allow_maxshield=True)

    # Write to bids
    rs_bids_path = BIDSPath(subject=bids["subject"],
                            session=bids["session"],
                            task='rest',
                            datatype=bids["datatype"],
                            root=bids["root"])
    write_raw_bids(rs_raw, rs_bids_path, overwrite=True)

def empty_to_bids(bids, meg):
    # Add empty room data
    er_raw_fname = op.join(meg["data_path"] % "Rnoise", bids["subject"] + '_MEEG_' + bids["session"] + '_Rnoise.fif')
    er_raw = mne.io.read_raw_fif(er_raw_fname, allow_maxshield=True)

    # For empty room data we need to specify the recording date
    er_date = er_raw.info['meas_date'].strftime('%Y%m%d')
    print(er_date)

    # Write to bids
    er_bids_path = BIDSPath(subject=bids['site']+'emptyroom',
                            session=er_date,
                            task='noise',
                            datatype=bids["datatype"],
                            root=bids["root"])
    write_raw_bids(er_raw, er_bids_path, overwrite=True)

def maxfiles_to_bids(bids, meg):
    # Find fine-calibration and crosstalk files
    cal_fname = op.join(meg["cal_path"], 'sss_cal_' + bids["site"] + '.dat')
    ct_fname = op.join(meg["cal_path"], 'ct_sparse_' + bids["site"] + '.fif')

    # Set BIDS path
    bids_path = BIDSPath(subject=bids["subject"],
                         session=bids["session"],
                         task=bids["task"],
                         run=1,
                         datatype=bids["datatype"],
                         root=bids["root"])

    # Add files to the bids structure
    write_meg_calibration(cal_fname, bids_path)
    write_meg_crosstalk(ct_fname, bids_path)

# def dicom_to_nifti(t1):
#     # Convert dicom to nifti
#     dicom2nifti.convert_directory(t1["dicom_path"],
#                                   t1["nifti_path"],
#                                   compression=True,
#                                   reorient=True)

#     # Rename nifti.gz file
#     for file in os.listdir(t1["nifti_path"]):
#         if file.endswith(".gz"):
#             os.rename(op.join(t1["nifti_path"], file),
#                       op.join(t1["nifti_path"], t1["fname"] + '.nii.gz'))

# def t1_to_bids(t1, bids, raw):
#     # Load the transformation matrix and show what it looks like
#     if op.exists(t1["trans_path"]):
#         trans = mne.read_trans(t1["trans_path"])
#     else:
#         raise FileNotFoundError("No such file or directory: " + t1["trans_path"])

#     # Use trans to transform landmarks to the voxel space of the T1
#     t1["nifti"] = op.join(t1["nifti_path"], t1["fname"] + '.nii.gz')
#     landmarks = get_anat_landmarks(
#         t1["mgz_path"],
#         info=raw.info,
#         trans=trans,
#         fs_subject=t1["subject"],
#         fs_subjects_dir=t1["fs_path"])

#     # Create the BIDSPath object.
#     t1w_bids_path = BIDSPath(subject=t1["subject"],
#                              session="V0",
#                              root=bids["root"],
#                              suffix='T1w')

#     # We use the write_anat function
#     t1w_bids_path = write_anat(
#         image = t1["nifti"],  # path to the MRI scan
#         bids_path = t1w_bids_path,
#         landmarks=landmarks,  # the landmarks in MRI voxel space
#         deface=True,
#         overwrite=True)


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    # First, convert visit 1 MEG data
    # visit_id = "V1"
    bids, meg, t1 = init_par(opt)
    # Convert raw (task) data to BIDS
    raw = raw_to_bids(bids, meg)
    # Add resting-state data
    rest_to_bids(bids, meg)
    # Add empty room data
    empty_to_bids(bids, meg)
    # Add fine-calibration and crosstalk files (maxfilter files)
    maxfiles_to_bids(bids, meg)
    print("\n#######################################"
          +"\nBIDS conversion completed successfully!"
          +"\n#######################################")

    # # Then, convert visit 2 MEG data
    # # visit_id = "V2"
    # bids, meg, t1 = init_par(subject_id, visit_id, t1_path, file_exts, bids_root)
    # # Convert raw (task) data to BIDS
    # raw = raw_to_bids(bids, meg)
    # # Add resting-state data
    # rest_to_bids(bids, meg)
    # # Add empty room data
    # empty_to_bids(bids, meg)
    # # Add fine-calibration and crosstalk files (maxfilter files)
    # maxfiles_to_bids(bids, meg)

    # # Eventually, convert T1 anat data
    # # Convert DICOM to NIFTI
    # dicom_to_nifti(t1)
    # # Add T1 anatomical scan
    # t1_to_bids(t1, bids, raw)

    # # Show BIDS tree
    # print_dir_tree(bids_root, max_depth=4)

    # # Show report
    # print(make_report(bids_root))

    # # Count events
    # count_events(bids_root)