#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finds sessions in raw data dir to be converted to BIDS, then runs converter. 
First finds session folders in raw dir, checks if these already exist in bids 
dir. Then copies not yet bids converted session dirs to a temporary folder, 
while adjusting the subject and session labels to be compliant with bidscoin 
requirements. Finally, runs bidscoin to convert raw to bids and cleans up 
the temporary dir afterwards.
Assumes existing bidsmap (for bidscoin doc see https://bidscoin.readthedocs.io)

v2: added support to only convert the first 50% of data based on tsv file

Created on Tue Jun  8 13:34:26 2021

@author: David Richter
@tag: prereg_v4.2
"""

# raw data dir
raw_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/Raw/projects/CoG_fMRI_PhaseII'
# scratch dir for temporary data storage
temp_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/scratch/temp_raw_for_bidscoin'
# bids dir
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# bidsmap for bidscoin
bids_map = bids_dir + '/code/bidscoin/bidsmap_fix_for_SD_data.yaml'

import os, shutil


#%% check for existing bids data
def find_sessions(raw_dir, bids_dir, temp_dir):
    """
    Find sessions in raw data dir to be processed. Checks if session already 
    exists in bids data dir, otherwise returns the unprocessed (raw) session 
    paths and associated temporary data dir paths in sessions_to_process.
    :param raw_dir: path to the raw data folder.
    :param temp_dir: path to the temporary data folder.
    :param bids_dir: path to the bids folder.
    """
    sessions_to_process = [];
    n_raw_sessions = 0
    
    # find subjects (raw)
    for (_, raw_subjects, _) in os.walk(raw_dir):break
    
    # loop over raw subjects and check if sessions are already in bids 
    for sid in raw_subjects:
        
        # check that the raw dir starts with a valid site label (i.e. is a valid participant)
        if sid[0:2]=='SC' or sid[0:2]=='SD':
            
            # current subjects raw data
            for (sub_dir, ses_dirs, _) in os.walk(raw_dir + os.sep + sid):break
            
            # find sessions to be processed
            for cur_ses in ses_dirs:
                n_raw_sessions += 1
                bids_session_label = 'ses-' + cur_ses[-2::]
                
                # some sessions have not been converted due to incorrect SD 
                # sequence header labels. Handle 3 cases of data
                
                # 1. bids subject session dir does not exist -> process
                expected_bids_sub_dir = bids_dir + os.sep + 'sub-' + sid + os.sep + bids_session_label
                if not os.path.isdir(expected_bids_sub_dir):
                    print('. no bids dir for: ' + sid + ', ' + bids_session_label + ' -> adding session to dicom2bids')
                    sessions_to_process.append((sub_dir + os.sep + cur_ses, temp_dir + os.sep + 'sub-' + sid + os.sep + bids_session_label))
                    continue
                
                # 2. bids subject session dir does exist, but does not contain func folder -> delete session bids dir, then process
                expected_func_path = expected_bids_sub_dir + os.sep + 'func'
                if os.path.isdir(expected_bids_sub_dir) and not os.path.isdir(expected_func_path):
                    print('. found bids dir, but no func dir for: ' + sid + ', ' + bids_session_label + ' -> deleting existing bids data, then adding session to dicom2bids')
                    delete_tree(expected_bids_sub_dir)
                    sessions_to_process.append((sub_dir + os.sep + cur_ses, temp_dir + os.sep + 'sub-' + sid + os.sep + bids_session_label))
                    continue
                    
                # 3. bids subject session dir does exist and does contain func -> do nothing                
    
    # print current status
    print('==============================================================')
    print('Found RAW data of -> ' + str(n_raw_sessions) + ' <- MRI SESSIONS in: ' + raw_dir)
    print('. of which -> ' + str(len(sessions_to_process)) + ' <- SESSIONS remain to be converted to BIDS dir: ' + bids_dir)
    
    # return raw and associated bids destination session paths to be processed
    return sessions_to_process


#%% copy flat dicom folder
def copy_dicoms_flat(raw_session, temp_destination):
    """
    Copy dicoms for current session to temporary folder while flattening all 
    sub directories; i.e. all dicoms are copied directly into the session dir.
    :param raw_session: path to the raw data for one session to be processed.
    :param temp_destination: path to the temporary data folder for one session.
    """
    from glob import glob
    
    print('==============================================================')
    print('Copying Session -> ' + raw_session[raw_session.find('projects')+8::] + ' to -> ' + temp_destination[temp_destination.find('processed')+9::])
    
    files = []
    
    # find all dicom files
    for cur_dir,_,_ in os.walk(raw_session):
        files.extend(glob(os.path.join(cur_dir,"*.dcm"))) 
    print('. Found: ' + str(len(files)) + ' dcm files to copy. Copying...')
    
    # copy files
    if not os.path.isdir(temp_destination):
        os.makedirs(temp_destination)
    for cur_file in files:
        shutil.copy(cur_file, temp_destination)


# sort dicom images into series folders
def run_dicomsort(temp_destination):
    """
    Run dicomsort on current session to sort flattened dicom files into series 
    folders.
    :param temp_destination: path to the temporary data folder for one session.
    """
    import subprocess
    
    print('==============================================================')
    print('Running DICOMSORT on -> ' + temp_destination[temp_destination.find('processed')+9::])
        
    # make & run dicomsort command
    dicom_sort_cmd = 'dicomsort ' + temp_destination
    subprocess.call(dicom_sort_cmd, shell=True)


# remove _discarded runs
def remove_discarded_runs(temp_destination):
    """
    Removes runs from temporary folder that have been marked as "_discarded".
    """
    print('==============================================================')
    print('Removing any discarded runs in  -> ' + temp_destination[temp_destination.find('processed')+9::])
    
    # loop over dirs in temp folder arranged by series labels
    for cur_dir,_,_ in os.walk(temp_destination):
        
        # check for discarded flags
        if "_discarded" in cur_dir:
            print('. Removing run ' + cur_dir)
            
            try:
                shutil.rmtree(cur_dir)
            except OSError as e:
                print("Error: %s : %s" % (cur_dir, e.strerror))
            

# copy raw data to temporary data dir
def prepare_dcm_to_bids_for_session(raw_session, temp_destination):
    """
    Process current session.
    Creates a temporary copy of raw dicom data, adjusting the folder name 
    structure (prepending sub- and ses- keys), then rearranges series into
    subfolders to allow subsequent processing of data with bidscoin; dicom to 
    bids converter. Finally concerts data to bids and removes temporary data.
    :param raw_session: path to the raw data for one session to be processed.
    :param temp_destination: path to the temporary data folder for one session.
    """
    # double check that subject and session IDs correspond between raw and temp dir
    if (not temp_destination[temp_destination.find('sub-')+4:temp_destination.find('sub-')+9] in raw_session) or (not temp_destination[temp_destination.find('ses-')+4:temp_destination.find('ses-')+6] in raw_session) :
        raise ValueError('WARNING raw dir subject or session label does NOT correspond to temp dir label for: ' + raw_session)
    
    # copy data to temp folder
    copy_dicoms_flat(raw_session, temp_destination)
    
    # sort data into series folders
    run_dicomsort(temp_destination)
    
    # remove discarded runs
    remove_discarded_runs(temp_destination)


#%% run bidscoin
def run_bidscoin(temp_dir, bids_dir, bids_map):
    """
    Run bidscoin (dicom to bids converter) for all session in temp_dir. Only
    runs converter for sessions not already processed and contained in 
    bids_dir. Uses bids_map for mapping between dicom and bids data.
    :param temp_dir: path to the temporary data folder.
    :param bids_dir: path to the bids folder.
    :param bids_map: path to the bidsmap.yaml, as used by bidscoin.
    """
    import subprocess
    
    print('==============================================================')
    print('Running BIDSCOIN for data in -> ' + temp_dir)
    print('Output to BIDS dir -> ' + bids_dir)
    print('Using bidsmap -> ' + bids_map)
        
    # make & run dicomsort command
    bidscoin_cmd = 'bidscoiner ' + temp_dir + ' ' + bids_dir + ' -b ' + bids_map
    subprocess.call(bidscoin_cmd, shell=True)


# clean up temporary files and dirs
def delete_tree(temp_dir):
    """
    Removes temporary/unnecessary folders.
    :param temp_dir: path to the data folder to be removed
    """
    print('. Removing folder  -> ' + temp_dir)
    
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        print("Error: %s : %s" % (temp_dir, e.strerror))


#%% process all sessions  
if __name__ == "__main__":
    """
    Run dicom to bids converter for all sessions. Finds all sessions
    in raw_dir, creates temporary data in temp_dir, outputs bids data 
    to bids_dir, using bids_map to map raw data to bids using bidscoin.
    """
    # get sessions to process
    sessions_to_process = find_sessions(raw_dir, bids_dir, temp_dir)
    
    # loop over sessions, preparing them for dicom2bids (with bidscoin)
    for session in sessions_to_process:
        print('')
        raw_session = session[0]
        temp_destination = session[1]
        prepare_dcm_to_bids_for_session(raw_session, temp_destination)
    
    # run bidscoin
    run_bidscoin(temp_dir, bids_dir, bids_map)
    
    # cleanup temp files
    print('==============================================================')
    print('Removing temporary files')
    delete_tree(temp_dir)
    