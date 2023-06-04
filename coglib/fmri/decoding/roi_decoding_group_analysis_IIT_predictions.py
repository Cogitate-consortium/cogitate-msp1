# 1 - Runs one sample permutation test on the irrelevant category decoding accuracy values of the IIT+GNW prediction model vs those from the IIT prediction model
"""
Author: Aya Khalaf
Date created: 09-13-2021
Date modified: 02-20-2023
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import mne.stats
decoding_problem = 'category'
approach = 'within_condition'
condition = 'irrelevant'
# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'

# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['face', 'object']
#stimulus_categories = ['letter', 'falsefont']
# Number of voxels per ROI (ROI size)
n_voxels = 300
# Chance level (50% for category decoding (binary) and 33.33% for orientation decoding (3-class) )
chance_level = 0.5

csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding/',approach, classifier, decoding_problem, stimulus_categories[0] + '_' + stimulus_categories[1] + '_' + str(n_voxels))
data_df = pd.read_csv(os.path.join(csv_dir, 'IIT_predictions_accuracy_scores_' + condition + '.csv'))

# Create a new dataframe to save the output values
df = pd.DataFrame(list())
df = data_df.loc[:, ['ROI','Theory']]
df.insert(2, 'Average Accuracy', '')
df.insert(3, 'Significance', '')
accuracy_values = data_df.iloc[:,3:].to_numpy()
df.loc[:,'Average Accuracy'] = accuracy_values.mean(axis=1)

# Run one sample  permutation test
ROI_count= len(accuracy_values[:,1])
p_values= np.zeros([ROI_count])

statistic, p_value, H0 = mne.stats.permutation_t_test(np.reshape(accuracy_values[2, :] - accuracy_values[1, :], (len(accuracy_values[2, :]), 1)), n_permutations=5000, tail=0)

df.loc[2, 'Significance'] = np.multiply(p_value < 0.05, 1)
# Save average accuracy and significance for all ROIs in a csv file
df.to_csv(os.path.join(csv_dir, 'accuracy_stats_IIT_predictions_' + condition + '.csv'))
