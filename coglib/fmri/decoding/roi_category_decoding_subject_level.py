"""
Performs category decoding and obtains decoding accuracy for each ROI within each subject using an ROI-based decoding approach.
The code uses single trial estimates calculated using nibetaseries 6.0 as input features to an SVM classifier

Author: Aya Khalaf
Email: aya.khalaf@yale.edu
Date created: 07-04-2022
Date modified: 12-11-2022
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

# GNW ROI list
GNW_roi_list = ['G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant', 'G_and_S_cingul-Mid-Post', 'G_front_inf-Opercular', 'G_front_inf-Orbital', 'G_front_inf-Triangul', 'G_front_middle', 'Lat_Fis-ant-Horizont', 'Lat_Fis-ant-Vertical', 'S_front_inf', 'S_front_middle', 'S_front_sup']
# IIT Basic ROI list
IIT_roi_list_1 = ['G_temporal_inf', 'Pole_temporal', 'G_cuneus', 'G_occipital_sup', 'G_oc-temp_med-Lingual', 'Pole_occipital', 'S_calcarine', 'G_and_S_occipital_inf', 'G_occipital_middle', 'G_oc-temp_lat-fusifor', 'G_oc-temp_med-Parahip', 'S_intrapariet_and_P_trans', 'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_temporal_sup']
# IIT extended ROI list
IIT_roi_list_2 = ['G_precentral', 'G_temp_sup-Lateral', 'G_temp_sup-Plan_tempo', 'G_pariet_inf-Supramar', 'G_temporal_middle', 'S_temporal_inf', 'G_orbital', 'G_pariet_inf-Angular', 'S_interm_prim-Jensen', 'S_occipital_ant', 'S_oc-temp_lat', 'S_precentral-inf-part']
# IIT excluded ROI List
IIT_roi_list_3 = ['G_and_S_frontomargin', 'G_and_S_transv_frontopol', 'G_front_sup', 'G_rectus', 'G_subcallosal', 'S_orbital_lateral', 'S_orbital_med-olfact', 'S_orbital-H_Shaped', 'S_suborbital']
# IIT ROI list
IIT_roi_list = IIT_roi_list_1 + IIT_roi_list_2 + IIT_roi_list_3

# Create IIT and GNW labels for each ROI
GNW_label = list()
IIT_label = list()
GNW_label = ['GNW'] * len(GNW_roi_list)
IIT_label = ['IIT'] * len(IIT_roi_list)
roi_labels_list = GNW_label + IIT_label
selected_roi_list = GNW_roi_list+IIT_roi_list

# GNW/IIT ROIs combining all ROIs that represent GNW/IIT
combined_roi_list =['GNW',  'IIT_extended', 'IIT_excluded', 'IIT']
# Add combined GNW/IIT ROIs to the list and add corresponding labels
roi_labels_list = roi_labels_list + ['GNW', 'IIT', 'IIT', 'IIT']
selected_roi_list = selected_roi_list + combined_roi_list

# Select whether to do within condition decoding or test generalization across conditions - options 'within_condition' and 'generalization'
approach = 'generalization'
# if approach = 'within_condition', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant' and 'irrelevant'
# if approach = 'generalization', the line below selects whether to perform decoding within relevant or irrelevant condition - options 'relevant-irrelevant' and 'irrelevant-relevant'
condition = 'irrelevant-relevant'

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
    for i, func_roi in enumerate(func_roi_filename):
        func_masker = NiftiMasker(mask_img=func_roi, standardize=True)
        index= run_labels.index(i % number_of_runs)
        if (i % number_of_runs ==0):
            data = func_masker.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(i % number_of_runs)])
        else:
            data = np.concatenate((data, func_masker.fit_transform(beta_maps.slicer[:, :, :, index:index + run_labels.count(i % number_of_runs)])), axis=0)
        if (i % number_of_runs == number_of_runs-1):
            scores = classification(data, trial_labels, run_labels, [], classifier, approach)
            roi_accuracy_score.append(scores.mean())

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
                        if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_categories [0]) != -1) &  (filename.find(stimulus_categories[1]) != -1) & (filename.find('_' + roi_condition) != -1) & (filename.find('leave_run') != -1) & (filename.find('_' + str(n_voxels) +'_') != -1) & bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                            func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                            index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                            roi_labels_extended.append(selected_roi_list[index[0]])
                            if (filename.find('run1') != -1):
                                index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                                theory_labels.append(roi_labels_list[index[0]])
                                roi_labels.append(selected_roi_list[index[0]])
                    elif approach == 'generalization':
                        if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_categories [0]) != -1) &  (filename.find(stimulus_categories[1]) != -1) & (filename.find('_' + roi_condition) != -1) & operator.not_((filename.find('leave_run') != -1)) & (filename.find('_' + str(n_voxels) +'_') != -1) & bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
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
                    df.insert(0, 'ROI', roi_labels)
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
df.to_csv(os.path.join(output_dir, 'accuracy_scores_' + condition + '.csv'))








