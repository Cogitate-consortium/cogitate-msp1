# 1 - Runs one sample permutation test on the accuracy values of each ROI against the chance level
# 2 - Performs correction for multiple comparisons of the p-values obtained in #1 across ROIs using FDR method
"""
Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 09-13-2021
Date modified: 02-20-2023
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import mne.stats
# Select whether to apply group analysis on category or orientation decoding problems - options 'category' and 'orientation'
decoding_problem = 'category'
#decoding_problem = 'orientation'
# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
# For orientation decoding, approach should be set to 'within_condition'
#approach = 'within_condition'
approach = 'generalization'
# For category decoding, if approach = 'within_condition', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# If approach = 'generalization', the line below selects whether to perform category decoding within relevant or irrelevant condition - options 'relevant-irrelevant' and 'irrelevant-relevant'
# For orientation decoding, no need to change 'condition' as it will be specified at line 39.
condition = 'irrelevant'
condition = 'irrelevant-relevant'
# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'

# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
# For orientation decoding, do not change 'stimulus_categories' here and change line 43 instead.
stimulus_categories = ['face', 'object']
#stimulus_categories = ['letter', 'falsefont']
# Number of voxels per ROI (ROI size)
n_voxels = 300
# Chance level (50% for category decoding (binary) and 33.33% for orientation decoding (3-class) )
chance_level = 0.5

csv_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding/',approach, classifier, decoding_problem, stimulus_categories[0] + '_' + stimulus_categories[1] + '_' + str(n_voxels))

if decoding_problem == 'orientation':
    condition = 'relevant+irrelevant'
    chance_level = 0.33
    stimulus_category = 'face'
    csv_dir = os.path.join( '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding/',approach, classifier, decoding_problem, stimulus_category + '_' + str(n_voxels))

# Load the accuracy values from the csv table:
data_df = pd.read_csv(os.path.join(csv_dir, 'accuracy_scores_' + condition + '.csv'))

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
for i in range(ROI_count):
     statistic, p_value, H0 = mne.stats.permutation_t_test(np.reshape(accuracy_values[i,:]-chance_level,(len(accuracy_values[i, :]),1)), n_permutations=5000, tail=1)
     p_values[i]=p_value

# Correct for multiple comparisons
reject, p_values_corrected= mne.stats.fdr_correction(p_values, alpha=0.05, method='indep')
df.loc[:, 'Significance'] = np.multiply(p_values_corrected<0.05, 1)
# Save average accuracy and significance for all ROIs in a csv file
df.to_csv(os.path.join(csv_dir, 'accuracy_stats_' + condition + '.csv'))
