"""
================
S05. Grand-average source epochs
================

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp, wilcoxon

import mne
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


parser=argparse.ArgumentParser()
parser.add_argument('--method',
                    type=str,
                    default='dspm',
                    help='method used for the inverse solution')
# parser.add_argument('--bids_root',
#                     type=str,
#                     default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
#                     help='Path to the BIDS root directory')
opt=parser.parse_args()


# Set params
inv_method = opt.method
visit_id = "V1"

debug = False


factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant non-target','Irrelevant'],
              ['500ms', '1000ms', '1500ms']]


# Set participant list
phase = 3

if debug:
    sub_list = ["SA124", "SA126"]
elif phase == 2:
    sub_list = ["SA106", "SA107", "SA109", "SA112", "SA113", "SA116", "SA124",
                "SA126", "SA127", "SA128", "SA131",
                "SA110", "SA142", "SA152", "SA160", "SA172",
                "SB002", "SB015", "SB022", "SB030" ,"SB038", "SB041", "SB061", 
                "SB065", "SB071", "SB073", "SB085",
                "SB013", "SB024", "SB045", "SB050", "SB078"
                ]
elif phase == 3:
    # Read the .txt list
    f = open(op.join(bids_root,
                  'participants_MEG_phase3_included.txt'), 'r').read()
    # Add spaces and split elements in the list
    sub_list = f.replace('\n', ' ').split(" ")


def source_dur_ga():
    # Set directory paths
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur")
    if not op.exists(source_deriv_root):
        os.makedirs(source_deriv_root)
    source_figure_root =  op.join(source_deriv_root,
                                f"sub-groupphase{phase}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(source_figure_root):
        os.makedirs(source_figure_root)
    
    # Set task
    if visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
    # Read the group data table
    bids_path_source = mne_bids.BIDSPath(
        root=source_deriv_root, 
        subject=f"groupphase{phase}",  
        datatype="meg",  
        task=bids_task,
        session=visit_id, 
        suffix="datatable",
        extension=".tsv",
        check=False)
    
    df = pd.read_csv(bids_path_source.fpath, sep="\t")
    
    # Move power values to a single column
    df['values'] = [np.array(df.iloc[i,1:-5]) for i in range(len(df))]
    
    # Drop the columns left
    df = df.drop(df.columns[1:-6],axis=1)
    
    # Select task-irrelevant trials only
    df = df[df['Task_relevance'] == 'Irrelevant']
    
    # Create info
    info = mne.create_info(
        ch_names=['gnw_all', 'iit_all'], 
        sfreq=1000)
    
    # Create empy data frame
    results = pd.DataFrame()
    
    # Loop over analysis
    for analysis in ['onset', 'offset']:
        
        # Loop over freq bands
        for band in ['alpha', 'gamma']:
            print('\nfreq_band:', band)
            
            # Create empty list
            all_df = np.empty((len(sub_list),2,(3501)))
            
            # Loop over labels
            for i, label in enumerate(['gnw_all', 'iit_all']):
                print('\nlabel:', label)
    
                # Get data for given conditions
                data = df[(df['band'] == band) & (df['label'] == label)]
                
                # If offset analysis, lock data to stim offset
                if analysis == 'offset':
                    for dur in np.unique(data["Duration"]):
                        data.loc[data['Duration'] == dur, "values"] = \
                            data.loc[data['Duration'] == dur, "values"].apply(
                                lambda temp: np.concatenate(
                                    [temp[int(dur[:-2]):],temp[:int(dur[:-2])]]))
                
                # Average across conditions
                data = data.groupby(
                    ['sub','band','label'])['values'].apply(
                        np.mean,0).to_frame().reset_index()
                
                # Append data to group array
                all_df[:,i,:] = np.stack(data['values'])
            
            # Empty list
            data_df = []
            
            # Loop across subjects
            for i, sub in enumerate(sub_list):
                print('\nsubject:', sub)
                
                # Create epoch object
                epochs = mne.EpochsArray(all_df[np.newaxis,i,:,:],
                                          info,
                                          tmin=-1.)
                
                # Format the data for the test
                data_df.append(format_tim_win_comp_data(
                    epochs,
                    sub,
                    baseline_window=[-0.2, 0.0],
                    test_window=[0.3, 0.5]))
                
            # Convert list to data frame:
            data_df= pd.concat(data_df, 
                                axis=0, 
                                ignore_index=True)
            
            # Performing the moving window test
            if band == "gamma":
                alternative = "greater"
            elif band == "alpha":
                alternative = "less"
            test_results = moving_window_test(
                data_df, 
                onset=[0.3, 0.5][0],
                alternative=alternative)
            
            # Append results to data frame
            test_results["band"] = band
            test_results["analysis"] = analysis
            results = results.append(test_results)
            
            # Save results as .tsv
            bids_path_source = bids_path_source.copy().update(
                root=source_deriv_root,
                subject=f"groupphase{phase}",
                suffix="onset_offset_results",
                check=False)
            results.to_csv(bids_path_source.fpath, 
                               sep="\t",
                               index=False)
            # # Plot
            # fig, axs = plt.subplots(2,1, figsize=(8,6))
            # for i, label in enumerate(['gnw_all', 'iit_all']):
            #     data = data_df.loc[data_df['channel'] == label, 'values']
            #     axs[i].plot(np.mean(data))
            #     axs[i].axhline(0, color='k', linestyle='--')
            #     # axs[i].set_ylim([-.01,.01])
            #     axs[i].set_xlim([0,201])
            # plt.suptitle(f"{band}-{analysis}", fontsize='xx-large', fontweight='bold')
            # plt.tight_layout()


def format_tim_win_comp_data(epochs, subject, baseline_window, test_window):
    """
    This function formats data to compare the activation between different time windows. It will reformat the epochs
    into data frames cropped into the specified time windows and take the subtraction between the two passed time
    windows. One can then test whether that difference is above chance for an extended period of time
    :param epochs: (mne epochs object) contains the data to compute the difference
    :param subject: (string) name of the subject
    :param baseline_window: (list of two floats) contains the onset and offset of the baseline
    :param test_window: (list of two floats) contains the onset and offset of the test data
    :return:
    """
    print("=" * 40)
    print("Welcome to format_cluster_based_data")
    data_df = pd.DataFrame()
    # Compute baseline and onset:
    baseline_data = epochs.copy().crop(tmin=baseline_window[0],
                                       tmax=baseline_window[1])
    onset_data = epochs.copy().crop(tmin=test_window[0],
                                    tmax=test_window[1])
    # Looping through each channel to compute the difference between the two:
    for channel in baseline_data.ch_names:
        bs = np.squeeze(baseline_data.get_data(picks=channel))
        ons = np.squeeze(onset_data.get_data(picks=channel))
        # It can  be that because of rounding the two arrays are not the same size, in which case, equating size
        # by taking the smallest
        if bs.shape[0] != ons.shape[0]:
            min_len = min([bs.shape[0], ons.shape[0]])
            bs = bs[:, 0:min_len]
            ons = ons[:, 0:min_len]
        diff = ons - bs
        # Add to the data_df frame:
        data_df = data_df.append(pd.DataFrame(
            {"subject": subject,
             "channel": channel,
             "values": [diff]
             }
        ))
    return data_df


def test_sustained_threshold(y, stat_test="t-test", threshold=0.05, 
                             window_sec=0.02, sr=1000,
                             alternative="greater", fdr_method=None):
    """
    :param y:
    :param stat_test:
    :param threshold:
    :param window_sec:
    :param sr:
    :param alternative:
    :param fdr_method:
    :return:
    """
    # Handling data dimensions
    if isinstance(y, np.ndarray):
        if len(y.shape) > 2:
            raise Exception("You have passed an numpy array of more than 2D! This function only works with 2D numpy "
                            "array or unnested list!")
    elif isinstance(y, list):
        if isinstance(y[0], list):
            raise Exception("You have passed a nested list! This function only works with 1D numpy "
                            "array or unnested list!")
        elif isinstance(y[0], np.ndarray):
            raise Exception("You have passed a list of numpy arrays!This function only works with 1D numpy "
                            "array or unnested list!")
    # Compute the test:
    if stat_test == "t-test":
        pop_mean = np.zeros(y.shape[0])
        y_stat, y_pval = ttest_1samp(y, pop_mean, axis=1, alternative=alternative)
    elif stat_test == "wilcoxon":
        y_stat, y_pval = wilcoxon(y, y=None, axis=1, alternative=alternative)
    else:
        raise Exception("You have passed a test that is not supported!")
    # Do fdr correction if needed:
    if fdr_method is not None:
        y_bin, y_pval, _, _ = multipletests(y_pval, alpha=threshold, method=fdr_method)
    else:
        y_bin = y_pval < threshold
    # Convert the time window from ms to samples:
    window_samp = int(window_sec * (sr / 1))
    h0 = True
    # Looping through each True in the binarize y:
    for ind in np.where(y_bin)[0]:
        if ind + window_samp < len(y_bin):
            if all(y_bin[ind:ind + window_samp]):
                h0 = False
                # Finding the offset of the significant window:
                onset_samp = ind
                if len(np.where(np.diff(y_bin[ind:].astype(int)) == -1)[0]) > 0:
                    offset_samp = onset_samp + np.where(np.diff(y_bin[ind:].astype(int)) == -1)[0][0]
                else:
                    offset_samp = len(y) - 1
                # Convert to me:
                onset_sec, offset_sec = onset_samp * (1 / sr), offset_samp * (1 / sr)
                break
        else:
            break
    if h0:
        onset_samp, offset_samp = None, None
        onset_sec, offset_sec = None, None
    return h0, [onset_sec, offset_sec], [onset_samp, offset_samp]


def moving_window_test(data_df, onset, groups="channel", alternative="greater"):
    print("=" * 40)
    print("Welcome to moving_window_test")
    # Var to store the results
    test_results = pd.DataFrame()
    for group in data_df[groups].unique():
        print("Performing test for group: {}".format(group))
        # Get the data of this group
        y = data_df.loc[data_df[groups] == group, "values"]
        # Convert to array
        y = np.array([np.array(yy) for yy in y])
        # Testing the sustained
        h0, sig_window_sec, sig_window_samp = test_sustained_threshold(y, alternative=alternative)

        # Create results table:
        test_results = test_results.append(pd.DataFrame({
            "channel": group,
            "sign": not h0,
            "onset": onset + sig_window_sec[0] if sig_window_sec[0] is not None
            else None,
            "offset": onset + sig_window_sec[1] if sig_window_sec[0] is not None
            else None,
        }, index=[0]))

    return test_results


if __name__ == '__main__':
    source_dur_ga()
