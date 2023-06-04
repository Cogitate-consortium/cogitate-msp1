"""
Performs orientation decoding and obtains decoding accuracy for each ROI within each subject in a leave-one-run-out cross validation scheme.
The code uses single trial estimates calculated using nibetaseries 6.0 as input features to an SVM classifier

Author: Aya Khalaf
Date created: 07-04-2022
Date modified: 12-11-2022
"""

import os
import operator
import pandas as pd
import numpy as np
import nibabel as nb


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
# The current implementation allows orientation decoding only using the 'within condition' approach
approach = 'within_condition'
# Decoding orientation within relevant and irrelevant conditions combined
condition = 'relevant+irrelevant'

# Select classifier - options 'SVM', 'LR', 'LDA', 'NB', 'KNN', and 'RF'
classifier = 'SVM'

# Number of voxels per ROI (ROI size)
n_voxels = 300
# Number of runs
number_of_runs = 8
# Scan repetition time
TR = 1.5
# Stimulus category to decode orientation for - options include 'face', 'object', 'letter' , and 'falseFont'
stimulus_category = 'face'
# Select functional ROIs that were created using the data of the selected decoding condition

roi_condition = ['rel', 'irrel']

# Output path
output_dir = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/decoding/nibetaseries/roi_decoding',approach, classifier, 'orientation/' + stimulus_category + '_' + str(n_voxels))


def prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition, stimulus_category, number_of_runs):
    # Extracts nibetaseries trial estimates
    from nilearn.image import index_img
    import nibabel as nb
    # Get number of sessions
    unique_runs = range(number_of_runs)
    orientation_1 = 'center'
    orientation_2 = 'right'
    orientation_3 = 'left'
    # Define lists to be filled with relevant trial estimates and the corresponding labels
    beta_maps_list = list()
    trial_labels = list()
    run_labels = list()
    # Loop over sessions to extract relevant trial estimates
    for run in unique_runs:
        i = 0
        # Get stimuli representing the categories in the decoding problem of interest
        behavioral = pd.read_csv(tsv_filename[run], sep='\t')
        stim_all = behavioral[behavioral.trial_type == stimulus_category]
        selected_stim_front = (stim_all.task_relevance == condition) & (stim_all.stimulus_orientation == orientation_1)
        selected_stim_right = (stim_all.task_relevance == condition) & (stim_all.stimulus_orientation == orientation_2)
        selected_stim_left = (stim_all.task_relevance == condition) & (stim_all.stimulus_orientation == orientation_3)
        # Get trial estimates corresponding to selected stimuli
        for filename in nibetaseries_filename:
            if filename.find('run-' + str(run+1) +'_space-MNI152NLin2009cAsym_desc-' + stimulus_category + '_betaseries.nii.gz') != -1:
                nibetaseries_stim_1_files = index_img(filename, selected_stim_front)
                beta_maps_list.append(nibetaseries_stim_1_files)
                nibetaseries_stim_2_files = index_img(filename, selected_stim_right)
                beta_maps_list.append(nibetaseries_stim_2_files)
                nibetaseries_stim_3_files = index_img(filename, selected_stim_left)
                beta_maps_list.append(nibetaseries_stim_3_files)


        # Create labels corresponding to the selected trial estimates
        trial_labels= trial_labels + [orientation_1] * sum(selected_stim_front) + [orientation_2] * sum(selected_stim_right) + [orientation_3] * sum(selected_stim_left)
        run_labels = run_labels + [run] * len([orientation_1] * sum(selected_stim_front) + [orientation_2] * sum(selected_stim_right) + [orientation_3] * sum(selected_stim_left))
    # Concatenate selected trial estimates into one 4D image
    beta_maps = nb.concat_images(beta_maps_list, axis=3)

    return beta_maps, trial_labels, run_labels


def roi_decoding_crossval(beta_maps_relevant, trial_labels_relevant, run_labels_relevant, beta_maps_irrelevant, trial_labels_irrelevant, run_labels_irrelevant, classifier, func_roi_filename):
    # Decodes stimulus category within each ROI and gives the corresponding accuracy scores

    from nilearn.input_data import NiftiMasker
    approach = 'within_condition'
    print("Running ROI Decoding")
    #roi_img_list = list()
    # Define an empty list to be filled with accuracy scores
    roi_accuracy_score = list()
    # Number of runs
    number_of_runs=8
    data_relevant = []
    # Loop over ROIs and get accuracy scores
    for i, func_roi in enumerate(func_roi_filename):
        func_masker = NiftiMasker(mask_img=func_roi, standardize=True)
        index_rel = run_labels_relevant.index(i % number_of_runs)
        index_irrel = run_labels_irrelevant.index(i % number_of_runs)
        if (i % number_of_runs ==0):
            data_relevant = func_masker.fit_transform(beta_maps_relevant.slicer[:, :, :, index_rel:index_rel + run_labels_relevant.count(i % number_of_runs)])
            data_irrelevant = func_masker.fit_transform(beta_maps_irrelevant.slicer[:, :, :, index_irrel:index_irrel + run_labels_irrelevant.count(i % number_of_runs)])
        else:
            data_relevant = np.concatenate((data_relevant, func_masker.fit_transform(beta_maps_relevant.slicer[:, :, :, index_rel:index_rel + run_labels_relevant.count(i % number_of_runs)])), axis=0)
            data_irrelevant = np.concatenate((data_irrelevant, func_masker.fit_transform(beta_maps_irrelevant.slicer[:, :, :, index_irrel:index_irrel + run_labels_irrelevant.count(i % number_of_runs)])),axis=0)
        if (i % number_of_runs == number_of_runs-1):
            data = np.concatenate((data_relevant, data_irrelevant), axis=0)
            trial_labels = trial_labels_relevant + trial_labels_irrelevant
            run_labels = run_labels_relevant + run_labels_irrelevant
            scores = classification(data, trial_labels, run_labels, classifier)
            roi_accuracy_score.append(scores.mean())

    return roi_accuracy_score

def classification(data, trial_labels, run_labels, clf):
    # Classifies trial estimates using leave-one-run-out cross validation scheme

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
        classifier = svm.SVC(kernel='linear', class_weight='balanced')
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

    scores = cross_val_score(classifier, data, trial_labels, cv=cv, groups=run_labels, scoring='balanced_accuracy')

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
                    # Select ROIs relevant to the decoding problem of interest
                    if ((filename.find('.nii.gz') != -1) & (filename.find(stimulus_category) != -1)  & (filename.find('_' + roi_condition[0]) != -1) & (filename.find('orientation') != -1) & (filename.find('leave_run') != -1) & (filename.find('_' + str(n_voxels) +'_') != -1) & bool([selected_roi_list.index(x) for x in selected_roi_list if x in filename])):
                        func_roi_filename.append(os.path.join(func_rois_dir, sub, filename))
                        index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                        roi_labels_extended.append(selected_roi_list[index[0]])
                        if (filename.find('run1') != -1):
                            index = [selected_roi_list.index(x) for x in selected_roi_list if x in filename]
                            theory_labels.append(roi_labels_list[index[0]])
                            roi_labels.append(selected_roi_list[index[0]])
                # Loop over event tsv files of the subject
                for filecnt, filename in enumerate(os.listdir(os.path.join(bids_dir, sub, sess_directory,'func'))):
                    if filename.find('.tsv') != -1:
                        tsv_filename.append(os.path.join(bids_dir, sub, sess_directory,'func', filename))

                nibetaseries_filename.sort()
                tsv_filename.sort()
                func_roi_filename = [x for _, x in sorted(zip(roi_labels_extended, func_roi_filename))]
                theory_labels = [x for _, x in sorted(zip(roi_labels, theory_labels))]
                roi_labels.sort()

                print("Running " + sub)

                conditions = condition.split('+')
                condition1 = conditions[0]
                condition2 = conditions[1]
                beta_maps1, trial_labels1, run_labels1 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition1, stimulus_category, number_of_runs)
                beta_maps2, trial_labels2, run_labels2 = prepare_nibetaseries_data(nibetaseries_filename, tsv_filename, condition2, stimulus_category, number_of_runs)

                roi_accuracy_scores = roi_decoding_crossval(beta_maps1, trial_labels1, run_labels1, beta_maps2, trial_labels2, run_labels2, classifier, func_roi_filename)

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








