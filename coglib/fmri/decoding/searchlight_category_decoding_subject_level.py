"""
Performs category decoding and obtains decoding accuracy for each voxel within each subject using a searchlight approach.
The code uses single trial estimates calculated using nibetaseries 6.0 as input features to an SVM classifier

Author: Aya Khalaf
Date created: 09-13-2021
Date modified: 12-20-2022
"""


import os
import operator
import pandas as pd
import numpy as np
from mansfield import get_searchlight_neighbours_matrix

# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'

# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
approach = 'within_condition'
#approach = 'generalization'
# if approach = 'within_condition', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# if approach = 'generalization', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant-irrelevant' and 'irrelevant-relevant'
condition = 'relevant'
#condition = 'relevant-irrelevant'
# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'
# Radius of the searchlight sphere
searchlight_radius = 4
# Number of runs
number_of_runs = 8
# Scan repetition time
TR = 1.5
# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['face', 'object']

# Output path
#output_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/category/faces_objects'
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/searchlight_decoding',approach, condition, classifier, str(searchlight_radius) + 'mm', 'category/' + stimulus_categories[0] + '_' + stimulus_categories[1])


def prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs):
    # Extracts nibetaseries trial estimates
    from nilearn.image import index_img
    import nibabel as nb
    # Get number of runs
    unique_runs = range(number_of_runs)
    # Define lists to be filled with relevant trial estimates and the corresponding labels
    beta_maps_list = list()
    trial_labels = list()
    run_labels = list()
    # Loop over sessions to extract relevant trial estimates
    for run in unique_runs:
        i = 0
        # Get stimuli representing the categories in the decoding problem of interest
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')
        stim_1_all = behavioral[behavioral.trial_type == stimulus_categories[0]]
        selected_stim_1 = stim_1_all.task_relevance == condition
        stim_2_all= behavioral[behavioral.trial_type == stimulus_categories[1]]
        selected_stim_2 = stim_2_all.task_relevance == condition
        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[0] + '_betaseries.nii.gz') != -1:
                nibetaseries_stim_1_files = index_img(filename, selected_stim_1)
                beta_maps_list.append(nibetaseries_stim_1_files)
            if filename.find('run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_categories[1] + '_betaseries.nii.gz') != -1:
                nibetaseries_stim_2_files = index_img(filename, selected_stim_2)
                beta_maps_list.append(nibetaseries_stim_2_files)


        # Create labels corresponding to the selected trial estimates
        trial_labels = trial_labels + [stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2)
        run_labels = run_labels + [run] * len([stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2))
    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels

def searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, mask_filename):
    import nilearn.decoding
    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score

    print("Running searchlight")
    approach = 'within_condition'
    # Create a mask so that the searchlight analysis is performed within the brain voxels only
    func_mask = nilearn.masking.intersect_masks(mask_filename, threshold=1)
    masker = NiftiMasker(mask_img=func_mask, standardize=True)
    func_mask.to_filename('func_mask.nii')
    # Get neighbours of each voxel in the brain
    searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask.nii', radius= searchlight_radius)
    data = masker.fit_transform(beta_maps)
    voxels = range(data.shape[1])
    searchlight_scores = np.zeros((1, data.shape[1]))
    # Loop over voxels and get the corresponding accuracies
    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        data_in_the_sphere = data[:, sphere_indices]
        scores = classification(data_in_the_sphere, trial_labels, run_labels, [], classifier, approach)
        searchlight_scores[:, center_voxel] = scores.mean()
    # Create a searchlight image with accuracies corresponding to each voxel
    searchlight_img = masker.inverse_transform(searchlight_scores)

    return searchlight_img

def searchlight_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, searchlight_radius, mask_filename):
    import nilearn.decoding
    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score

    print("Running searchlight")
    approach = 'generalization'
    # Create a mask so that the searchlight analysis is performed within the brain voxels only
    func_mask = nilearn.masking.intersect_masks(mask_filename, threshold=1)
    masker = NiftiMasker(mask_img=func_mask, standardize=True)
    func_mask.to_filename('func_mask.nii')
    # Get neighbours of each voxel in the brain
    searchlight_neighbours_matrix = get_searchlight_neighbours_matrix('func_mask.nii', radius= searchlight_radius)
    data1 = masker.fit_transform(beta_maps1)
    data2 = masker.fit_transform(beta_maps2)
    voxels = range(data1.shape[1])
    searchlight_scores = np.zeros((1, data1.shape[1]))
    # Loop over voxels and get the corresponding accuracies
    for center_voxel in voxels:
        sphere_indices = searchlight_neighbours_matrix[center_voxel].toarray()[0].astype('bool')
        data_in_the_sphere1 = data1[:, sphere_indices]
        data_in_the_sphere2 = data2[:, sphere_indices]
        predicted_labels = classification(data_in_the_sphere1, trial_labels1, [], data_in_the_sphere2, classifier, approach)
        searchlight_scores[:, center_voxel] = accuracy_score(predicted_labels, trial_labels2)
    # Create a searchlight image with accuracies corresponding to each voxel
    searchlight_img = masker.inverse_transform(searchlight_scores)

    return searchlight_img

def classification(data1, trial_labels1, run_labels1, data2, clf, approach):
    # Classifies trial estimates, depending on the selected approach, either using
    # 1) leave-one-run-out cross validation scheme or 2) specified training and testing sets
    from sklearn import svm
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import cross_val_score
    cv = LeaveOneGroupOut()
    # Classification Options
    if clf == 'SVM':
        classifier = svm.SVC(kernel='linear')
    elif clf == 'LDA':
        classifier = LDA()
    elif clf == 'NB':
        classifier = GaussianNB()
    elif clf == 'LR':
        classifier = LogisticRegression()
    elif clf == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif clf == 'RF':
        classifier = RandomForestClassifier()

    if approach =='within_condition':
        scores = cross_val_score(classifier, data1, trial_labels1, cv=cv, groups=run_labels1)
    elif approach == 'generalization':
        classifier.fit(data1, trial_labels1)
        scores = classifier.predict(data2)

    return scores

# Define lists to be filled with subject-specific data
i=0
mask_filename = list()
nibetaseries_filename = list()
tsv_filename = list()
os.makedirs(output_dir, exist_ok=True)

# Get phase3 subjects
tsv_file = os.path.join(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv')
tsv_data= pd.read_csv(tsv_file, sep='\t')
subjects_phase3 = tsv_data.participant_id
subject_list=subjects_phase3.tolist()
searchlight_filename = list()

# Loop over subjects
for sub in subject_list:
    for root, sess_dirs, session_files in os.walk(os.path.join(nibetaseries_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            if ((sess_directory.find('ses-V1') != -1) &  operator.not_(os.path.isdir(os.path.join(output_dir, sub,'ses-V1')))):
                os.makedirs(os.path.join(output_dir, sub, sess_directory))
                # Loop over single trial estimates files in nibetaseries directory of the subject
                for filecnt, filename in enumerate( os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory, 'func'))):
                    if ((filename.find('face') != -1) | (filename.find('object') != -1) | (filename.find('letter') != -1) | (filename.find('falseFont') != -1)):
                        nibetaseries_filename.append(os.path.join(nibetaseries_dir, sub, sess_directory, 'func', filename))
                # Loop over functional mask files in fmriprep directory of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(preprocessed_dir, sub, sess_directory, 'func'))):
                    if filename.find('MNI152NLin2009cAsym_desc-brain_mask.nii.gz') != -1:
                        mask_filename.append(os.path.join(preprocessed_dir, sub, sess_directory, 'func', filename))
                # Loop over event tsv files of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory, 'func'))):
                    if filename.find('.tsv') != -1:
                        tsv_filename.append(os.path.join(bids_dir, sub, sess_directory, 'func', filename))

                mask_filename.sort()
                nibetaseries_filename.sort()
                tsv_filename.sort()

                print("Running " + sub)

                if approach == 'within_condition':
                    beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs)
                    searchlight_img = searchlight_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, searchlight_radius, mask_filename)
                elif approach == 'generalization':
                    conditions = condition.split('-')
                    condition1 = conditions[0]
                    condition2 = conditions[1]
                    beta_maps1, trial_labels1, run_labels1 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition1, stimulus_categories, number_of_runs)
                    beta_maps2, trial_labels2, run_labels2 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition2, stimulus_categories, number_of_runs)
                    searchlight_img = searchlight_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, searchlight_radius, mask_filename)
                # Save searchlight image as a nifti file
                searchlight_img.to_filename(os.path.join(output_dir, sub, sess_directory, 'searchlight_accuracy_map.nii'))
                i = i + 1
                print(i)
                # Clear all lists before processing next subject
                mask_filename = list()
                nibetaseries_filename = list()
                tsv_filename = list()











