"""
Tests the decoding predictions of IIT on irrelevant category decoding by evaluating accuracies of a prediction model employing features from IIT ROIs against accuracies of
a prediction model employing features from GNW+ IIT ROIs combined

Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 12-01-2022
Date modified: 04-22-2023
"""

import os
import operator
import pandas as pd
import numpy as np



# BIDS path
bids_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids'
# fMRIprep path
preprocessed_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/fmriprep'
# nibetaseries trial estimates path
nibetaseries_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/nibetaseries'
# functional ROIs path
func_rois_dir = '/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding_rois'




# GNW/IIT ROIs combining all ROIs that represent GNW/IIT
roi_labels_list = ['GNW', 'IIT']
selected_roi_list = ['GNW', 'IIT']

approach ='within_condition'
condition ='irrelevant'

# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'
# Number of voxels per ROI (ROI size)
n_voxels = 300
# Number of runs
number_of_runs = 8
# Scan repetition time
TR = 1.5
# Stimulus categories to be decoded - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_categories = ['face', 'object']
# Select functional ROIs that were created using the data of the selected decoding condition
if ((condition == 'relevant') | (condition == 'relevant-irrelevant') ):
    roi_condition ='rel'
elif ((condition == 'irrelevant') | (condition == 'irrelevant-relevant')):
    roi_condition = 'irrel'

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding',approach, classifier, 'category/' + stimulus_categories[0] + '_' + stimulus_categories[1] + '_' + str(n_voxels))


def prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs):
    # Extracts nibetaseries trial estimates
    from nilearn.image import index_img
    import nibabel as nb
    # Get number of sessions
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
        #trial_label_relevant.append([stim_1] * sum(selected_stim_1) + [stim_2] * sum(selected_stim_2))
        run_labels = run_labels + [run] * len([stimulus_categories[0]] * sum(selected_stim_1) + [stimulus_categories[1]] * sum(selected_stim_2))
    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels


def roi_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, func_roi_filename):
    # Decodes stimulus category within each ROI and gives the corresponding accuracy scores

    from nilearn.input_data import NiftiMasker
    approach = 'within_condition'
    print("Running ROI Decoding")
    # Define an empty list to be filled with accuracy scores
    roi_accuracy_score = list()
    # Number of runs
    number_of_runs=8
    data_relevant = []
    # Loop over ROIs and get accuracy scores
    for run in range(number_of_runs):
        func_masker1 = NiftiMasker(mask_img=func_roi_filename[run], standardize=True)
        func_masker2 = NiftiMasker(mask_img=func_roi_filename[run + number_of_runs], standardize=True)
        index= run_labels.index(run % number_of_runs)
        if (run % number_of_runs ==0):
            data1 = func_masker1.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(run % number_of_runs)])
            data2 = func_masker2.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(run % number_of_runs)])
        else:
            data1 = np.concatenate((data1, func_masker1.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(run % number_of_runs)])), axis=0)
            data2 = np.concatenate((data2, func_masker2.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(run % number_of_runs)])), axis=0)

        if (run % number_of_runs == number_of_runs-1):
            data = np.concatenate((data1, data2), axis=1)
            scores_GNW = classification(data1, trial_labels, run_labels, [], classifier, approach)
            scores_IIT = classification(data2, trial_labels, run_labels, [], classifier, approach)
            scores_combined = classification(data, trial_labels, run_labels, [], classifier, approach)
            roi_accuracy_score.append(scores_GNW.mean())
            roi_accuracy_score.append(scores_IIT.mean())
            roi_accuracy_score.append(scores_combined.mean())

    return roi_accuracy_score


def roi_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, func_roi_filename):
    # Decodes stimulus category within each ROI and gives the corresponding accuracy scores

    from nilearn.input_data import NiftiMasker
    from sklearn.metrics import accuracy_score
    approach ='generalization'
    print("Running ROI Decoding")
    roi_accuracy_score = list()
    for i, func_roi in enumerate(func_roi_filename):
        func_masker = NiftiMasker(mask_img=func_roi, standardize=True)
        data1 = func_masker.fit_transform(beta_maps1)
        data2 = func_masker.fit_transform(beta_maps2)
        predicted_labels = classification(data1, trial_labels1, [], data2, classifier, approach)
        roi_accuracy_score.append(accuracy_score(predicted_labels, trial_labels2))

    return roi_accuracy_score


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


# Define lists to be filled with ROI information and subject-specific data
i=0
nibetaseries_filename = list()
func_roi_filename = list()
tsv_filename = list()
roi_labels = list()
roi_labels_extended = list()
theory_labels = list()

# Create a data frame to save accuracy scores
df = pd.DataFrame(list())

# Get phase3 subjects
tsv_file = os.path.join(bids_dir,'participants_fMRI_QC_included_phase3_sesV1.tsv')
tsv_data= pd.read_csv(tsv_file, sep='\t')
subjects_phase3 = tsv_data.participant_id
subject_list=subjects_phase3.tolist()
# Loop over subjects
for sub in subject_list:
    for root, sess_dirs, session_files in os.walk(os.path.join(nibetaseries_dir, sub)):
        # Loop over sessions
        for sess_directory in sess_dirs:
            if ((sess_directory.find('ses-V1') != -1) & os.path.isdir(os.path.join(func_rois_dir, sub))):
                # Loop over single trial estimates files in nibetaseries directory of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(nibetaseries_dir, sub, sess_directory,'func'))):
                    if ((filename.find('face') != -1) | (filename.find('object') != -1) | (filename.find('letter') != -1) | (filename.find('falseFont') != -1)):
                        nibetaseries_filename.append(os.path.join(nibetaseries_dir, sub, sess_directory,'func', filename))
                # Loop over functional ROIs of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(func_rois_dir, sub))):
                    if approach == 'within_condition':
                        # Select ROIs relevant to the decoding problem of interest
                        if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_categories [0]) != -1) &  (filename.find(stimulus_categories[1]) != -1) & (filename.find('_' + roi_condition) != -1) & (filename.find('leave_run') != -1) & (filename.find('extended') == -1) & (filename.find('excluded') == -1) & (filename.find('S_front_inf') == -1) & (filename.find('_' + str(n_voxels) +'_') != -1) & bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                            func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                            index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                            roi_labels_extended.append(selected_roi_list[index[0]])
                            if (filename.find('run1') != -1):
                                index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                                theory_labels.append(roi_labels_list[index[0]])
                                roi_labels.append(selected_roi_list[index[0]])
                    elif approach == 'generalization':
                        if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_categories [0]) != -1) &  (filename.find(stimulus_categories[1]) != -1) & (filename.find('_' + roi_condition) != -1) & operator.not_((filename.find('leave_run') != -1)) & (filename.find('extended') == -1) & (filename.find('excluded') == -1) & (filename.find('S_front_inf') == -1) & (filename.find('_' + str(n_voxels) +'_') != -1) & bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                            func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                            index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                            theory_labels.append(roi_labels_list[index[0]])
                            roi_labels.append(selected_roi_list[index[0]])
                # Loop over event tsv files of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory,'func'))):
                    if filename.find('.tsv') != -1:
                        tsv_filename.append(os.path.join(bids_dir, sub, sess_directory,'func', filename))

                nibetaseries_filename.sort()
                tsv_filename.sort()
                if approach == 'within_condition':
                    func_roi_filename = [x for _, x in sorted(zip(roi_labels_extended, func_roi_filename))]
                elif approach == 'generalization':
                    func_roi_filename = [x for _, x in sorted(zip(roi_labels, func_roi_filename))]
                theory_labels = [x for _, x in sorted(zip(roi_labels, theory_labels))]
                roi_labels.sort()

                print("Running " + sub)
                if approach =='within_condition':
                    beta_maps, trial_labels, run_labels = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_categories, number_of_runs)
                    roi_accuracy_scores = roi_decoding_crossval(beta_maps, trial_labels, run_labels, classifier, func_roi_filename)
                elif approach == 'generalization':
                    conditions = condition.split('-')
                    condition1 = conditions[0]
                    condition2 = conditions[1]
                    beta_maps1, trial_labels1, run_labels1 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition1, stimulus_categories, number_of_runs)
                    beta_maps2, trial_labels2, run_labels2 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition2, stimulus_categories, number_of_runs)
                    roi_accuracy_scores = roi_decoding_generalization(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, func_roi_filename)

                # Insert column names in the accuracy data frame
                if i==0:
                    roi_labels.append('Combined')
                    df.insert(0, 'ROI', roi_labels)
                    theory_labels.append('Combined')
                    df.insert(1, 'Theory', theory_labels)

                # Increment row index and insert accuracy scores for the current subject
                i = i + 1
                df.insert(i+1, sub, roi_accuracy_scores)
                print(i)



                # Clear all lists before processing next subject
                nibetaseries_filename = list()
                func_roi_filename = list()
                tsv_filename = list()
                roi_labels = list()
                roi_labels_extended = list()
                theory_labels = list()

# Create a directory to save the output
os.makedirs(output_dir, exist_ok=True)
# Save accuracy scores in a csv file
df.to_csv(os.path.join(output_dir, 'IIT_predictions_accuracy_scores_' + condition + '.csv'))
