"""
Modified by Urszula Gorska (gorska@wisc.edu)
===================
04. Extract events
===================

Extract events from the stimulus channel


"""
import numpy as np
import mne
import pandas as pd



def run_events(raw, experiment_id):
        
        ###############
        # Read events #
        ###############
    
    # Find response events
    response = mne.find_events(raw,
                            stim_channel='STI101',
                            consecutive = False,
                            mask = 65280,
                            mask_type = 'not_and'
                            )
    response = response[response[:,2] == 255]
        
    # Find all other events
    events = mne.find_events(raw,
                            stim_channel='STI101',
                            consecutive = True,
                            min_duration=0.001001,
                            mask = 65280,
                            mask_type = 'not_and'
                            )
    events = events[events[:,2] != 255]
        
    # Concatenate all events
    events = np.concatenate([response,events],axis = 0)
    events = events[events[:,0].argsort(),:]
        
        
    #################
    # Read metadata #
    #################
        
    # # Generate metadata table
    if experiment_id == 'V1':
        eve = events.copy()
        events = eve[eve[:, 2] < 81].copy()
        metadata = {}
        metadata = pd.DataFrame(metadata, index=np.arange(len(events)),
                                columns=['Stim_trigger', 'Category',
                                        'Orientation', 'Duration',
                                        'Task_relevance', 'Trial_ID',
                                        'Response', 'Response_time(s)'])
        Category = ['face', 'object', 'letter', 'false']
        Orientation = ['Center', 'Left', 'Right']
        Duration = ['500ms', '1000ms', '1500ms']
        Relevance = ['Relevant target', 'Relevant non-target', 'Irrelevant']
        k = 0
        for i in range(eve.shape[0]):
            if eve[i, 2] < 81:
            ##find the end of each trial (trigger 97)
                t = [t for t, j in enumerate(eve[i:i + 9, 2]) if j == 97][0]
                metadata.loc[k]['Stim_trigger'] = eve[i,2]
                metadata.loc[k]['Category'] = Category[int((eve[i,2]-1)//20)]
                metadata.loc[k]['Orientation'] = Orientation[[j-100 for j in eve[i:i+t,2]
                                                            if j in [101,102,103]][0]-1]
                metadata.loc[k]['Duration'] = Duration[[j-150 for j in eve[i:i+t,2]
                                                        if j in [151,152,153]][0]-1]
                metadata.loc[k]['Task_relevance'] = Relevance[[j-200 for j in eve[i:i+t,2]
                                                            if j in [201,202,203]][0]-1]
                metadata.loc[k]['Trial_ID'] = [j for j in eve[i:i+t,2]
                                            if (j>110) and (j<149)][0]
                metadata.loc[k]['Response'] = True if any(eve[i:i+t,2] == 255) else False
                if metadata.loc[k]['Response'] == True:
                    r = [r for r,j in enumerate(eve[i:i+t,2]) if j == 255][0]
                    metadata.loc[k]['Response_time(s)'] = (eve[i+r,0] - eve[i,0])
                k += 1
                            
    elif experiment_id == 'V2':
        eve = events.copy()
        metadata = {}
        metadata = pd.DataFrame(metadata, index=np.arange(np.sum(events[:, 2] < 51)),
                                columns=['Trial_type', 'Stim_trigger',
                                        'Stimuli_type',
                                        'Location', 'Response',
                                        'Response_time(s)'])
        types0 = ['Filler', 'Probe']
        type1 = ['Face', 'Object', 'Blank']
        location = ['Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']
        response = ['Seen', 'Unseen']
        k = 0
        for i in range(eve.shape[0]):
            if eve[i, 2] < 51:
                metadata.loc[k]['Stim_trigger'] = eve[i, 2]
                t = int(eve[i + 1, 2] % 10)
                metadata.loc[k]['Trial_type'] = types0[t]
                if eve[i, 2] == 50:
                    metadata.loc[k]['Stimuli_type'] = type1[2]
                else:
                    metadata.loc[k]['Stimuli_type'] = type1[eve[i, 2] // 20]
                    metadata.loc[k]['Location'] = location[eve[i + 1, 2] // 10 - 6]
                if t == 1:
                    metadata.loc[k]['Response'] = response[int(eve[i + 4, 2] - 98)]
                    metadata.loc[k]['Response_time(s)'] = (eve[i + 4, 0] - eve[i + 3, 0]) 
                k += 1


    return eve, metadata
