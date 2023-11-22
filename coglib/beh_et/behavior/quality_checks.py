import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
import gc
import itertools

import data_reader
import quality_checks_criteria
import data_saver


"""
This module performs quality checks on subject data of experiment 1.

### LAST UPDATED: 2022-12-20 :  ###

The quality-check module includes:
- Checks of the experimental data : to determine which subjects will be excluded from the general analyses of exp.1.
The module outputs a table in which each row is a subject and each column is a statistic of this subject's behavior
in the screening and/or in the game. Subjects who don't have full game data are analyzed anyway, the game-analysis
columns will remain empty.
Based on the thresholds that exist in the "quality_checks_criteria" module, the last 3 columns of the output table
indicate whether the subject's data is valid.

@author: RonyHirsch
"""

os.environ["OUTDATED_IGNORE"] = "1"

# responseEvaluation types
EVAL = 'responseEvaluation'
TRUEPOSITIVE = 'TruePositive'
FALSEPOSITIVE = 'FalsePositive'
TRUENEGATIVE = 'TrueNegative'
FALSENEGATIVE = 'FalseNegative'

FACE = "face"
OBJ = "object"
LETTER = "letter"
FALF = "falseFont"
BLOCK_TYPE_MAPPING = {FACE: "face & object", OBJ: "face & object", LETTER: "letter & false", FALF: "letter & false"}
RT_COL = "reactionTime"
D_PRIME = "dPrime"
FA_RATE = "falseAlarmRate"
# stimulus duration types
SHORT = 0.5
MEDIUM = 1
LONG = 1.5
DUR_LIST = [SHORT, MEDIUM, LONG]

# names for saving
FILENAME_PHASE2 = "quality_checks_phase2.csv"
FILENAME_PHASE3 = "quality_checks_phase3.csv"

# names and colors for plotting
# XXX: Load the labname dictionary from a local file
LAB_NAMES_DICT = None
LAB_NAME_DICT = LAB_NAMES_DICT  # Sometimes singular var is used sometimes plural

STIM_HUE_DICT = {FACE: "#003544", OBJ: "#ad501d", LETTER: "#397384", FALF: "#601f00"}

LAB_HUE_DICT = {"CA": "#cd7a19", "CB": "#ff991f", "CC": "#93001a", "SD": "#ff8ac2", "SE": "#144E3B",
                "CF": "#26B46C", "SG": "#0e8a69", "SZ": "tab:Blue"}

STIM_TITLE_DICT = {FACE: FACE.title(), OBJ: OBJ.title(), LETTER: LETTER.title(), FALF: "False Font"}
# colors from https://github.com/Cogitate-consortium/plotting_uniformization/blob/main/config.py
TASK_HUE_DICT = {True: "#D55E00", False: "#8B2BE2"}
TASK_TITLE_DICT = {True: "Task Relevant", False: "Task Irrelevant"}
DUR_HUE_DICT = {0.5: "#f6d746", 1.0: "#f37819", 1.5: "#bc3754"}
# fonts
TITLE_SIZE = 20
AXIS_SIZE = 25
TICK_SIZE = 20
LABEL_PAD = 8

PHASE_2_PATH = {data_reader.FMRI: "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/participants_fMRI_QC_included_phase2_sesV1.txt",
                data_reader.MEG: "/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/participants_MEG_phase2_included.txt"}

PHASE_3_PATH = {data_reader.FMRI: "/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/participants_fMRI_QC_included_phase3_sesV1.txt",
                data_reader.MEG: "/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/participants_MEG_phase3_included.txt",
                data_reader.ECOG: "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids/participants_ECoG_phase3_included.txt"}



def evaluate_resp(row):
    """
    Evaluate subject responses based on the responseType column.
    NOTE: the column mapping is based on this page: https://twcf-arc.slab.com/posts/saving-the-data-rah8lglh
    In this mapping, 1 = pressed the target key (up); 2 = wrong key was pressed
    """

    if row[data_reader.EXPECTED_RESP] == 1:  # in this trial, the subject was supposed to press and respond
        if row[data_reader.RESPONSE_TYPE_COL] == 1:  # the correct key was pressed
            return TRUEPOSITIVE
        elif row[data_reader.RESPONSE_TYPE_COL] is None:  # No key was pressed
            return FALSENEGATIVE
        elif row[data_reader.RESPONSE_TYPE_COL] == 2:  # the WRONG key was pressed
            return TRUEPOSITIVE  # *** DMT DECISION : WE ARE TAKING THEM AS RESPONSES ***

    elif row[data_reader.EXPECTED_RESP] == 0:  # in this trial, the subject was NOT supposed to press and respond
        if row[data_reader.RESPONSE_TYPE_COL] == 1:  # the correct key was pressed
            return FALSEPOSITIVE
        elif row[data_reader.RESPONSE_TYPE_COL] == 2:  # the WRONG key was pressed
            return FALSEPOSITIVE  # *** DMT DECISION : WE ARE TAKING THEM AS RESPONSES ***
        else:  # NO KEY WAS PRESSED
            return TRUENEGATIVE
    return


def categorize_responses(subject):
    """
    Cagegorize each trial's response (TP/FP/CR/FN) based on the information in the processed behavioral dataframe.
    """
    sub_df = subject.processed_data
    sub_df.loc[:, EVAL] = sub_df.apply(lambda row: evaluate_resp(row), axis=1)
    subject.processed_data = sub_df
    return sub_df


def calc_dprime(num_hits, num_fas, num_stim_present, num_stim_absent):
    """
    This method calculates SDT's d prime : d' = Z(hit rate) - Z(false alarm rate)
    with Z() being the Z score of these fractions.
    In extreme cases where hit rate / false alarm rate are exactly 0 or 1, the value of d prime will be undefined.
    Therefore, an adjustment needs to be made to the d' calculation.

    As per the preregistration, we will correct for these cases using the log-linear correction mentioned here:
    Hautus, M.J. Corrections for extreme proportions and their biasing effects on estimated values of d′.
    Behavior Research Methods, Instruments, & Computers 27, 46–51 (1995). https://doi.org/10.3758/BF03203619

    We follow the standard way of implementing this, similar to the R implementation of the dPrime method:
    https://github.com/neuropsychology/psycho.R/blob/f3b614ca8bfa0b3498f7eed79d64d6e13d57ec8f/R/dprime.R#L79-L81
    see also: https://github.com/neuropsychology/psycho.R/issues/113
    """
    # regular hit rate: num_hits / num_stim_present
    adjusted_hit_rate = (num_hits + 0.5) / (num_stim_present + 1)
    # regular false alarm rate: num_fas / num_stim_absent
    adjusted_fa_rate = (num_fas + 0.5) / (num_stim_absent + 1)

    """
    The parallel of R's qnorm (see R implementation above) is scipy.stat's norm.ppf
    https://stackoverflow.com/questions/24695174/python-equivalent-of-qnorm-qf-and-qchi2-of-r
    """
    z_hit_rate = scipy.stats.norm.ppf(adjusted_hit_rate)
    z_fa_rate = scipy.stats.norm.ppf(adjusted_fa_rate)

    d_prime = z_hit_rate - z_fa_rate

    return d_prime


def dprime_by_cat_dur_mod(sub: data_reader.Subject):
    """
    *** ANALYSIS 1: Sensitivity by category and duration ***
    LMM: d' : per category and duration (d' ~ category + duration)
    """
    trials = sub.processed_data
    d_prime_cat_dur_list = list()
    for category in [FACE, OBJ, LETTER, FALF]:
        for dur in DUR_LIST:
            # only trials with the right duration
            relevant_trials = trials[trials[data_reader.PLND_STIM_DUR] == dur]
            # and ones presenting the stimulus category
            stim_present = relevant_trials[relevant_trials[data_reader.STIM_TYPE_COL] == category]

            # *TASK RELEVANT* *TARGETS*: target stimulus is present
            targets = stim_present[stim_present[data_reader.EXPECTED_RESP] == 1]
            hits = targets[targets[EVAL] == TRUEPOSITIVE]
            hit_rate = np.nan if targets.shape[0] == 0 else 100 * hits.shape[0] / targets.shape[0]

            # *NON* TARGETS: target stimulus is ABSENT, across trials in this CATEGORY x DURATION
            non_targets = stim_present[stim_present[data_reader.EXPECTED_RESP] != 1]
            fas = non_targets[non_targets[EVAL] == FALSEPOSITIVE]

            d_prime = calc_dprime(num_hits=hits.shape[0], num_fas=fas.shape[0],
                                  num_stim_present=targets.shape[0], num_stim_absent=non_targets.shape[0])

            d_prime_cat_dur_list.append([sub.mod, sub.lab, sub.sub_name, category, dur, d_prime,
                                         hits.shape[0], targets.shape[0], hit_rate, fas.shape[0], non_targets.shape[0]])

    d_prime_cat_dur_table = pd.DataFrame(d_prime_cat_dur_list, columns=[data_reader.MODALITY, data_reader.LAB,
                                                                        data_reader.SUB_CODE, data_reader.STIM_TYPE_COL,
                                                                        data_reader.PLND_STIM_DUR, D_PRIME,
                                                                        "hits", "targets", "hit_rate",
                                                                        "fas", "nontargets"])

    return d_prime_cat_dur_table


def fa_by_cat_tr(sub: data_reader.Subject):
    """
    *** ANALYSIS 2: false alarm rates by category, modality and task relevance with subject and item as random effects ***
    """
    trials = sub.processed_data
    rel_cols = [data_reader.SUB_CODE, data_reader.MODALITY, data_reader.LAB, data_reader.TASK_RELEVANT,
                data_reader.STIM_TYPE_COL, data_reader.STIM_CODE_COL, data_reader.STIM_ID_COL,
                data_reader.EXPECTED_RESP, data_reader.RESPONSE_TYPE_COL, EVAL]
    res_trials = trials[rel_cols]
    res_trials.loc[:, f"is{FALSEPOSITIVE}"] = np.where(res_trials[EVAL] == FALSEPOSITIVE, True, False)
    return res_trials


def hit_rt_by_cat_mod(sub: data_reader.Subject):
    """
    *** ANALYSIS 3: RT ***
    RT for true positives (hits)
    For both of these, let's generate a dataframe that includes all the necessary information
    This returns hit_rt_table which is a *TRIAL* *BASED* *DATAFRAME* containing a row per trial,
    containing the relevant information
    """
    trials = sub.processed_data
    hit_trials = trials[trials[EVAL] == TRUEPOSITIVE]  # only hits
    rel_cols = [data_reader.SUB_CODE, data_reader.MODALITY, data_reader.LAB,
                data_reader.STIM_TYPE_COL, data_reader.STIM_CODE_COL, data_reader.STIM_ID_COL,
                data_reader.PLND_STIM_DUR, RT_COL, EVAL]
    res_trials = hit_trials[rel_cols]
    return res_trials


def fa_rt_by_cat_mod(sub: data_reader.Subject):
    """
    *** ANALYSIS 4: RT ***
    LMM (3) RT for false positives (FAs): per task relevance, category and duration
    """
    trials = sub.processed_data
    # LMM (3)
    non_targets = trials[trials[data_reader.EXPECTED_RESP] != 1]  # target stimulus is ABSENT, across **ALL** trials
    fas = non_targets[non_targets[EVAL] == FALSEPOSITIVE]  # take only trials where a response was made (fas)
    fa_rt_table = fas.loc[:, [data_reader.SUB_CODE, data_reader.MODALITY, data_reader.TASK_RELEVANT,  # task relevance!
                              data_reader.STIM_TYPE_COL, data_reader.PLND_STIM_DUR, RT_COL]]
    return fa_rt_table


def extract_data(root_folder):
    """
    Read all the subjects' data and save each subject as a struct (Subject) in a dictionary holding all subjects.
    Then, return the subjects dictionary for further processing and quality checks
    """
    # Check every subject folder to be valid, then create an instance of the class Subject for each valid subject,
    # containing all the session and info
    subjects = data_reader.data_reader_hpc(root_folder)  # create Subject struct for all subjects, and put in dict
    empty_cnt = 0
    for subject in subjects:
        print(f"----- Subject {subject} -----")
        if subjects[subject].processed_data is None:  # check whether subject has processed BEH data
            print(f"WARNING: Subject {subject} has no v1 session. Skipping!")
            empty_cnt += 1
            continue

        # first, save that subject's trial-based behavioral dataframe (no further processing) into their  QC folder
        # create a folder in that subject's derivatives to hold that data
        sub_qc_path = data_saver.create_sub_qcs_hpc(subject, quality_checks_criteria.METHOD_HPC[subject[:2]], root_folder)
        # then, save the trial-based behavioral (raw) dataframe into the qc location
        subjects[subject].processed_data.to_csv(os.path.join(sub_qc_path, f"sub-{subject}_ses-v1_beh_trials.csv"), index=False)

        # categorize responses and add this processing to the processed data table:
        trials = categorize_responses(subjects[subject])
        # add an RT column to the trial data, as this will be analyzed as well:
        trials.loc[:, RT_COL] = trials[data_reader.RESPONSE_TIME_COL] - trials[data_reader.STIM_ONSET_COL]
        subjects[subject].processed_data = trials

    print(f"DONE EXTRACTING DATA: {empty_cnt} SUBJECTS WERE SKIPPED AS THEY HAVE NO BEHAVIORAL DATA IN V1")
    return subjects


def process_data_for_qc(sub_dict):
    """
    This method prepares an across-subject dataframe containing all the relevant information for behavioral data
    quality-checks. Iterating over each subject, it extracts response information and adds it into a result dataframe.
    In this dataframe, each subejct = row, and columns provide information about responses.
    """
    result_list = list()
    for subject in sub_dict:
        sub_data = sub_dict[subject].processed_data
        # hits
        targets = sub_data[sub_data[data_reader.EXPECTED_RESP] == 1]
        hits = targets[targets[EVAL] == TRUEPOSITIVE]
        # false alarms
        non_targets = sub_data[sub_data[data_reader.EXPECTED_RESP] != 1]
        fas = non_targets[non_targets[EVAL] == FALSEPOSITIVE]
        # breaking down the false alarms:
        # task relevant trials
        non_target_tr = non_targets[non_targets[data_reader.TASK_RELEVANT] == True]
        false_alarms_tr = non_target_tr[non_target_tr[EVAL] == FALSEPOSITIVE]
        # task irrelevant trials
        non_target_ti = non_targets[non_targets[data_reader.TASK_RELEVANT] == False]
        false_alarms_ti = non_target_ti[non_target_ti[EVAL] == FALSEPOSITIVE]
        # average RT for hits
        hit_rt = np.nanmean(hits.loc[:, RT_COL])
        # subject dataframe
        sub_res = pd.DataFrame([[sub_dict[subject].sub_name, sub_dict[subject].mod,  # subject info
                                hits.shape[0], targets.shape[0],  hits.shape[0] / targets.shape[0],  # hits
                                fas.shape[0], non_targets.shape[0], fas.shape[0] / non_targets.shape[0],  # false alarms (overall)
                                false_alarms_tr.shape[0], non_target_tr.shape[0],  false_alarms_tr.shape[0] / non_target_tr.shape[0],  # false alarms (task relevant)
                                false_alarms_ti.shape[0], non_target_ti.shape[0], false_alarms_ti.shape[0] / non_target_ti.shape[0],  # false alarms (task IRrelevant)
                                hit_rt]],
                               columns=[data_reader.SUB_CODE, data_reader.MODALITY,
                                        quality_checks_criteria.HITS, quality_checks_criteria.STIM_PRESENT,
                                        quality_checks_criteria.HIT_RATE,
                                        quality_checks_criteria.FAS, quality_checks_criteria.STIM_ABSENT,
                                        quality_checks_criteria.FA_RATE,
                                        f"{quality_checks_criteria.FAS}_{quality_checks_criteria.TASK_RELEVANT}",
                                        f"{quality_checks_criteria.STIM_ABSENT}_{quality_checks_criteria.TASK_RELEVANT}",
                                         quality_checks_criteria.FA_RATE_TR,
                                        f"{quality_checks_criteria.FAS}_{quality_checks_criteria.TASK_IRRELEVANT}",
                                        f"{quality_checks_criteria.STIM_ABSENT}_{quality_checks_criteria.TASK_IRRELEVANT}",
                                        quality_checks_criteria.FA_RATE_TI, f"avgHit_{RT_COL}"])
        result_list.append(sub_res)
    result_df = pd.concat(result_list, ignore_index=True)
    return result_df


def single_plot_per_lab(data, y_col, ymin, ymax, plot_title, plot_y_label, lab_dict, save_path, save_name, skip=1, legend=True):
    """
    This method plots a single plot where X = lab, and Y is a parameter.
    """
    # plot
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    # X axis params
    lab_names = sorted(list(lab_dict.keys()))
    lab_names_nice = [LAB_NAME_DICT[lab] for lab in lab_names]
    increment = 0.1
    stim_xs = {lab_name: lab_names.index(lab_name) * increment for lab_name in lab_names}

    for lab in lab_names:
        df_lab = data[data[data_reader.LAB] == lab]
        if not df_lab.empty:
            x_loc = stim_xs[lab]
            y_vals = df_lab[y_col]
            # plot violin
            violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.17, showmeans=True, showextrema=False,showmedians=False)
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(lab_dict[lab])

            # change the color of the mean lines (showmeans=True)
            violin['cmeans'].set_color("black")
            violin['cmeans'].set_linewidth(2)
            # control the length like before
            m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

            # then scatter
            scat_x = (np.ones(len(y_vals)) * (x_loc - 0.05)) + (np.random.rand(len(y_vals)) * 0.05)
            plt.scatter(x=scat_x, y=y_vals, marker="o", color=lab_dict[lab], s=50, alpha=0.5, edgecolor=lab_dict[lab])

    # cosmetics
    plt.xticks(ticks=list(stim_xs.values()), labels=lab_names_nice, fontsize=TICK_SIZE)
    plt.yticks([y for y in np.arange(ymin, ymax + (0.5 * skip), skip)], fontsize=TICK_SIZE)
    plt.title(plot_title, fontsize=TITLE_SIZE, pad=LABEL_PAD + 10)
    plt.ylabel(plot_y_label, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel("Site", fontsize=AXIS_SIZE, labelpad=LABEL_PAD + 5)
    if legend:
        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=lab_dict[label], marker='o', linestyle='') for label in lab_names]
        new_labels = [LAB_NAMES_DICT[label] for label in lab_names]
        legend = plt.legend(markers, new_labels, title="Site", markerscale=1, fontsize=TICK_SIZE - 2)
        plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    else:
        legend = plt.legend().set_visible(False)
    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    gc.collect()

    return


def single_plot_per_dur_category(data, y_col, ymin, ymax, plot_title, plot_y_label, save_path, save_name, skip=1, leg=True):
    """
    This method plots a single plot where X = stimulus duration, HUE = stimulus category, and Y is a parameter.
    """
    # X axis params
    stim_xs = {FACE: -0.15, OBJ: -0.05, LETTER: 0.05, FALF: 0.15}
    durs = [0.5, 1, 1.5]
    # plot
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for dur in durs:
        df_dur = data[data[data_reader.PLND_STIM_DUR] == dur]  # only relevant duration
        for stim_type in STIM_HUE_DICT:
            df_stim = df_dur[df_dur[data_reader.STIM_TYPE_COL] == stim_type]  # only relevant stimulus category
            # we want one point PER SUBJECT (x category x duration)
            x_loc = dur + stim_xs[stim_type]
            y_vals = df_stim[y_col]
            # plot violin
            violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.17, showmeans=True, showextrema=False, showmedians=False)
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(STIM_HUE_DICT[stim_type])

            # change the color of the mean lines (showmeans=True)
            violin['cmeans'].set_color("black")
            violin['cmeans'].set_linewidth(2)
            # control the length like before
            m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

            # then scatter
            scat_x = (np.ones(len(y_vals)) * (x_loc-0.045)) + (np.random.rand(len(y_vals)) * 0.045)
            plt.scatter(x=scat_x, y=y_vals, marker="o", color=STIM_HUE_DICT[stim_type], s=50, alpha=0.5, edgecolor=STIM_HUE_DICT[stim_type])
    # cosmetics
    plt.xticks(durs, fontsize=TICK_SIZE)
    #plt.yticks([y for y in np.arange(ymin, ymax + (1 * skip), skip)], fontsize=TICK_SIZE)
    plt.ylim(ymin, ymax + (1 * skip))
    plt.locator_params(axis='y', nbins=10)
    plt.title(plot_title, fontsize=TITLE_SIZE, pad=LABEL_PAD + 10)
    plt.ylabel(plot_y_label, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel("Stimulus Duration (Seconds)", fontsize=AXIS_SIZE, labelpad=LABEL_PAD + 5)
    if leg:
        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=STIM_HUE_DICT[label], marker='o', linestyle='') for label in STIM_TITLE_DICT]
        new_labels = [STIM_TITLE_DICT[label] for label in STIM_TITLE_DICT]
        legend = plt.legend(markers, new_labels, title="Stimulus Category", markerscale=1, fontsize=TICK_SIZE - 2)
        plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    else:
        legend = plt.legend().set_visible(False)
    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    gc.collect()
    return


def plot_hit_rate(dprime_df, save_path, save_name):
    modality_list = [data_reader.ECOG, data_reader.FMRI, data_reader.MEG]

    # global ymin and max across modalities
    ymin = 0
    ymax = 100

    all_lab_dict = {lab: LAB_HUE_DICT[lab] for lab in dprime_df[data_reader.LAB].unique().tolist()}
    dprime_df_per_sub = dprime_df.groupby([data_reader.MODALITY, data_reader.LAB, data_reader.SUB_CODE]).mean().reset_index()
    single_plot_per_lab(data=dprime_df_per_sub, y_col="hit_rate", ymin=ymin, ymax=ymax,
                        plot_title=f"Hit Rate per Lab All", lab_dict=all_lab_dict,
                        plot_y_label="Hit Rate (%)", save_path=save_path,
                        save_name=f"hitrate_avg_{save_name}_ALL_per_lab", skip=20)

    for mod in modality_list:
        df = dprime_df[dprime_df[data_reader.MODALITY] == mod]
        # (1) TOTAL modality df
        if not df.empty:
            lab_dict = {lab: LAB_HUE_DICT[lab] for lab in df[data_reader.LAB].unique().tolist()}
            single_plot_per_dur_category(data=df, y_col="hit_rate", ymin=ymin, ymax=ymax,
                                         plot_title=f"Hit Rate per Stimulus Duration and Category {mod}",
                                         plot_y_label="Hit Rate (%)", save_path=save_path,
                                         save_name=f"{save_name}_{mod}", skip=20)

            # (2) breakdown per lab
            # labs
            labs = df.lab.unique().tolist()
            for lab in labs:
                df_lab = df[df[data_reader.LAB] == lab]
                single_plot_per_dur_category(data=df_lab, y_col="hit_rate", ymin=ymin, ymax=ymin,
                                             plot_title=f"Hit Rate per Stimulus Duration and Category {lab}",
                                             plot_y_label="Hit Rate (%)", save_path=save_path,
                                             save_name=f"{save_name}_{mod}_{lab}", skip=20)

    return


def plot_d_prime(dprime_df, save_path, save_name):
    modality_list = [data_reader.ECOG, data_reader.FMRI, data_reader.MEG]
    dprime_threshold = 2  # this is a lower threshold for closly-examining exceptions

    # global ymin and max across modalities
    ymin = int(dprime_df[D_PRIME].min())
    #if 0 < ymin:
    #    ymin = 0
    ymax = dprime_df[D_PRIME].max()
    ymax = math.ceil(ymax)

    # aggregate extreme subs
    extreme_datapoints = list()

    all_lab_dict = {lab: LAB_HUE_DICT[lab] for lab in sorted(dprime_df[data_reader.LAB].unique().tolist())}
    # collapse across everything, per subject
    dprime_df_per_sub = dprime_df.groupby([data_reader.MODALITY, data_reader.LAB, data_reader.SUB_CODE]).mean().reset_index()
    single_plot_per_lab(data=dprime_df_per_sub, y_col=D_PRIME, ymin=ymin, ymax=ymax, lab_dict=all_lab_dict,
                        plot_title=f"Sensitivity per Lab ALL", legend=False,
                        plot_y_label="Sensitivity (d')", save_path=save_path,
                        save_name=f"dprime_avg_{save_name}_ALL_per_lab")

    for mod in modality_list:
        df = dprime_df[dprime_df[data_reader.MODALITY] == mod]  # only relevant modality
        # (1) TOTAL modality df
        if not df.empty:
            single_plot_per_dur_category(data=df, y_col=D_PRIME, ymin=ymin, ymax=ymax, leg=False,
                                         plot_title=f"Sensitivity per Stimulus Duration and Category {mod}",
                                         plot_y_label="Sensitivity (d')", save_path=save_path,
                                         save_name=f"{save_name}_{mod}")

            # give some information about exceptional subjects
            df_exceptional_mod = df[df[D_PRIME] < dprime_threshold]
            extreme_datapoints.append(df_exceptional_mod)

            # (2) breakdown per lab
            # get the modalities' min max
            mod_ymin = int(df[D_PRIME].min())
            if 0 < mod_ymin:
                mod_ymin = 0
            mod_ymax = df[D_PRIME].max()
            mod_ymax = int(math.ceil(mod_ymax))
            # labs
            labs = df.lab.unique().tolist()
            for lab in labs:
                df_lab = df[df[data_reader.LAB] == lab]
                single_plot_per_dur_category(data=df_lab, y_col=D_PRIME, ymin=mod_ymin, ymax=mod_ymax, leg=False,
                                             plot_title=f"Sensitivity per Stimulus Duration and Category {lab}",
                                             plot_y_label="Sensitivity (d')", save_path=save_path,
                                             save_name=f"{save_name}_{mod}_{lab}")

    df_exceptional = pd.concat(extreme_datapoints, ignore_index=True)
    return df_exceptional


def plot_hit_rt(hit_rt_df, save_path, save_name):
    # prepare dataframe
    rt_df_summ = hit_rt_df.groupby([data_reader.MODALITY, data_reader.SUB_CODE, data_reader.STIM_TYPE_COL,
                                    data_reader.PLND_STIM_DUR]).agg({data_reader.STIM_CODE_COL: "count", RT_COL: "mean"}).reset_index()
    rt_df_summ.rename({data_reader.STIM_CODE_COL: 'count'}, axis=1, inplace=True)
    rt_df_summ.loc[:, data_reader.LAB] = rt_df_summ.apply(lambda row: map_lab(row), axis=1)
    # prepare plot
    modality_list = [data_reader.ECOG, data_reader.FMRI, data_reader.MEG]
    ymin = round(rt_df_summ[RT_COL].min(), 1)
    #if 0 < ymin:
    #    ymin = 0
    ymax = rt_df_summ[RT_COL].max()
    #ymax = int(math.ceil(ymax))
    ymax = math.ceil(ymax)

    # aggregate extreme subs
    extreme_datapoints = list()
    rt_lower_threshold = 0.250  # this is a lower threshold for closly-examining exceptions
    rt_upper_threshold = 2.5

    all_lab_dict = {lab: LAB_HUE_DICT[lab] for lab in rt_df_summ[data_reader.LAB].unique().tolist()}
    single_plot_per_lab(data=rt_df_summ, y_col=RT_COL, ymin=ymin, ymax=ymax, lab_dict=all_lab_dict,
                        plot_title=f"Hit Reaction Times per Lab ALL", legend=False,
                        plot_y_label="Reaction Time (Seconds)", save_path=save_path,
                        save_name=f"{save_name}_ALL_per_lab", skip=0.1)

    for mod in modality_list:
        df = rt_df_summ[rt_df_summ[data_reader.MODALITY] == mod]  # the SUMMARY df, not the trial-by-trial one
        if not df.empty:
            lab_dict = {lab: LAB_HUE_DICT[lab] for lab in df[data_reader.LAB].unique().tolist()}
            local_max = df[RT_COL].max()
            single_plot_per_dur_category(data=df, y_col=RT_COL, ymin=ymin, ymax=local_max, leg=False,
                                         plot_title=f"Hit Reaction Times per Stimulus Duration and Category {mod}",
                                         plot_y_label="Reaction Time (Seconds)", save_path=save_path,
                                         save_name=f"{save_name}_{mod}", skip=0.1)
            single_plot_per_lab(data=df, y_col=RT_COL, ymin=ymin, ymax=local_max, lab_dict=lab_dict,
                                plot_title=f"Hit Reaction Times per Lab {mod}", legend=False,
                                plot_y_label="Reaction Time (Seconds)", save_path=save_path,
                                save_name=f"{save_name}_{mod}_per_lab", skip=0.1)

            # give some information about exceptional subjects
            df_exceptional_mod = df[(df[RT_COL] > rt_upper_threshold) | (df[RT_COL] < rt_lower_threshold)]
            extreme_datapoints.append(df_exceptional_mod)

            # (2) breakdown per lab
            # get the modalities' min max
            mod_ymin = int(df[RT_COL].min())
            #if 0 < mod_ymin:
            #    mod_ymin = 0
            mod_ymax = df[RT_COL].max()
            mod_ymax = int(math.ceil(mod_ymax))
            # labs
            labs = df.lab.unique().tolist()
            for lab in labs:
                df_lab = df[df[data_reader.LAB] == lab]
                single_plot_per_dur_category(data=df_lab, y_col=RT_COL, ymin=mod_ymin, ymax=mod_ymax, leg=False,
                                             plot_title=f"Hit Reaction Times per Stimulus Duration and Category {lab}",
                                             plot_y_label="Reaction Time (Seconds)", save_path=save_path,
                                             save_name=f"{save_name}_{mod}_{lab}", skip=0.1)

    df_exceptional = pd.concat(extreme_datapoints, ignore_index=True)
    return df_exceptional


def fa_single_plot(data, plot_title, save_path, save_name, skip=1):
    """
    This method plots a single plot where X = lab, HUE = task relevance, and Y is the false alarm rate.
    """
    # prepare df
    fa_df_summ = data.groupby([data_reader.MODALITY, data_reader.TASK_RELEVANT, data_reader.SUB_CODE]).agg({data_reader.LAB: "count", f"is{FALSEPOSITIVE}": 'sum'}).reset_index()
    fa_df_summ.rename({data_reader.LAB: 'count'}, axis=1, inplace=True)
    fa_df_summ.loc[:, "False Alarm Rate"] = 100 * (fa_df_summ.loc[:, f"is{FALSEPOSITIVE}"] / fa_df_summ.loc[:, "count"])
    fa_df_summ.loc[:, data_reader.LAB] = fa_df_summ.apply(lambda row: map_lab(row), axis=1)

    # prepare plot
    modality_list = [data_reader.MEG, data_reader.FMRI, data_reader.ECOG, "ALL"]
    ymin = int(fa_df_summ["False Alarm Rate"].min())
    if 0 < ymin:
        ymin = 0
    ymax = fa_df_summ["False Alarm Rate"].max()
    ymax = int(math.ceil(ymax))

    for mod in modality_list:
        if mod != "ALL":
            df = fa_df_summ[fa_df_summ[data_reader.MODALITY] == mod]
        else:
            df = fa_df_summ
        if not df.empty:
            new_title = f"{plot_title} {mod}"
            lab_dict = {lab: LAB_HUE_DICT[lab] for lab in sorted(df[data_reader.LAB].unique().tolist())}
            # X axis params
            lab_names = list(lab_dict.keys())
            increment = 0.5
            x_tick_vals = {lab_name: lab_names.index(lab_name) * increment for lab_name in lab_names}
            stim_xs = {False: -0.05, True: 0.05}

            # plot
            plt.gcf()
            plt.figure()
            sns.reset_orig()

            for lab in lab_names:
                df_lab = df[df[data_reader.LAB] == lab]
                for task_relevance in list(stim_xs.keys()):
                    df_lab_relevance = df_lab[df_lab[data_reader.TASK_RELEVANT] == task_relevance]
                    if not df_lab_relevance.empty:
                        x_loc = x_tick_vals[lab] + stim_xs[task_relevance]
                        y_vals = df_lab_relevance["False Alarm Rate"]
                        # plot violin
                        violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.17, showmeans=True, showextrema=False, showmedians=False)
                        # make it a half-violin plot (only to the LEFT of center)
                        for b in violin['bodies']:
                            # get the center
                            m = np.mean(b.get_paths()[0].vertices[:, 0])
                            # modify the paths to not go further right than the center
                            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                            b.set_color(TASK_HUE_DICT[task_relevance])

                        # change the color of the mean lines (showmeans=True)
                        violin['cmeans'].set_color("black")
                        violin['cmeans'].set_linewidth(2)
                        # control the length like before
                        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                        violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

                        # then scatter
                        scat_x = (np.ones(len(y_vals)) * (x_loc - 0.06)) + (np.random.rand(len(y_vals)) * 0.06)
                        plt.scatter(x=scat_x, y=y_vals, marker="o", color=TASK_HUE_DICT[task_relevance], s=50, alpha=0.5, edgecolor=TASK_HUE_DICT[task_relevance])

            # cosmetics
            plt.xticks(ticks=list(x_tick_vals.values()), labels=[LAB_NAME_DICT[l] for l in list(x_tick_vals.keys())], fontsize=TICK_SIZE)
            plt.yticks([y for y in np.arange(ymin, ymax + (1 * skip), skip)], fontsize=TICK_SIZE)
            plt.title(new_title, fontsize=TITLE_SIZE, pad=LABEL_PAD + 5)
            plt.ylabel("False Alarm Rate (%)", fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
            plt.xlabel("Site", fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
            # legend
            markers = [plt.Line2D([0, 0], [0, 0], color=TASK_HUE_DICT[label], marker='o', linestyle='') for label in TASK_TITLE_DICT]
            new_labels = [TASK_TITLE_DICT[label] for label in TASK_TITLE_DICT]
            legend = plt.legend(markers, new_labels, title="Stimulus Relevance", markerscale=1.5, fontsize=TICK_SIZE - 2)
            plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
            # save plot
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(15, 12)
            plt.savefig(os.path.join(save_path, f"fa_avg_{save_name}_{mod}_TR_per_lab.png"), dpi=1000, bbox_inches='tight', pad_inches=0.01)
            plt.savefig(os.path.join(save_path, f"fa_avg_{save_name}_{mod}_TR_per_lab.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
            del figure
            plt.close()
            gc.collect()
            fa_df_summ.to_csv(os.path.join(save_path, f"fa_avg_{save_name}_{mod}_TR_per_lab.csv"))

    return


def single_plot_per_cat_relevance(data, y_col, ymin, ymax, plot_title, plot_y_label, save_path, save_name, leg=True, skip=1):
    """
    This method plots a single plot where X = stimulus category, HUE = task relevance, and Y is a parameter.
    """
    # X axis params
    stim_xs = {FACE: 1, OBJ: 2, LETTER: 3, FALF: 4}

    plt.gcf()
    plt.figure()
    sns.reset_orig()
    for stim_type in stim_xs:
        df_stim = data[data[data_reader.STIM_TYPE_COL] == stim_type]
        for relevance in TASK_HUE_DICT:
            df_rel = df_stim[df_stim[data_reader.TASK_RELEVANT] == relevance]
            if not df_rel.empty:  # if we even have data in this condition
                x_loc = stim_xs[stim_type]
                # so that conditions won't overlap
                if relevance == True:
                    x_loc -= 0.2
                else:
                    x_loc += 0.2
                y_vals = df_rel[y_col]
                # plot violin
                violin = plt.violinplot(y_vals, positions=[x_loc], showmeans=True, showextrema=False, showmedians=False)
                # make it a half-violin plot (only to the LEFT of center)
                for b in violin['bodies']:
                    # get the center
                    m = np.mean(b.get_paths()[0].vertices[:, 0])
                    # modify the paths to not go further right than the center
                    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                    b.set_color(TASK_HUE_DICT[relevance])

                # change the color of the mean lines (showmeans=True)
                violin['cmeans'].set_color("black")
                violin['cmeans'].set_linewidth(2)
                # control the length like before
                m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

                # then scatter
                scat_x = (np.ones(len(y_vals)) * (x_loc-0.125)) + (np.random.rand(len(y_vals)) * 0.12)
                plt.scatter(x=scat_x, y=y_vals, marker="o", s=75, color=TASK_HUE_DICT[relevance], alpha=0.6, edgecolor=TASK_HUE_DICT[relevance])
    # cosmetics
    plt.xticks([x for x in range(0, 5, 1)], fontsize=TICK_SIZE, labels=["", STIM_TITLE_DICT[FACE], STIM_TITLE_DICT[OBJ],
                                                                        STIM_TITLE_DICT[LETTER], STIM_TITLE_DICT[FALF]])
    #plt.yticks([y for y in range(ymin, ymax + 1, skip)], fontsize=TICK_SIZE)
    plt.ylim(ymin - (0.5 * skip), ymax + (1 * skip))
    plt.locator_params(axis='y', nbins=8)
    plt.title(plot_title, fontsize=TITLE_SIZE, pad=LABEL_PAD)
    plt.ylabel(plot_y_label, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel("Stimulus Category", fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    if leg:
        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=TASK_HUE_DICT[label], marker='o', linestyle='') for label in TASK_TITLE_DICT]
        new_labels = [TASK_TITLE_DICT[label] for label in TASK_TITLE_DICT]
        legend = plt.legend(markers, new_labels, title="Stimulus Relevance", markerscale=1, fontsize=TICK_SIZE - 2)
        plt.setp(legend.get_title(), fontsize=TICK_SIZE - 2)
    else:
        legend = plt.legend().set_visible(False)
    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    data.to_csv(os.path.join(save_path, f"{save_name}.csv"))
    del figure
    plt.close()
    gc.collect()
    return


def map_lab(row):
    sub_lab = row[data_reader.SUB_CODE][:2]
    return sub_lab


def plot_fa(fa_df, save_path, save_name):
    # prepare df
    fa_df_summ = fa_df.groupby([data_reader.MODALITY, data_reader.STIM_TYPE_COL, data_reader.TASK_RELEVANT, data_reader.SUB_CODE]).agg({data_reader.LAB: "count", f"is{FALSEPOSITIVE}": 'sum'}).reset_index()
    fa_df_summ.rename({data_reader.LAB: 'count'}, axis=1, inplace=True)
    fa_df_summ.loc[:, "False Alarm Rate"] = 100 * (fa_df_summ.loc[:, f"is{FALSEPOSITIVE}"] / fa_df_summ.loc[:, "count"])
    fa_df_summ.loc[:, data_reader.LAB] = fa_df_summ.apply(lambda row: map_lab(row), axis=1)

    # REMOVE CF121 - HAS 36% FA RATE IN TASK RELEVANT FACE
    #fa_df_summ = fa_df_summ[fa_df_summ[data_reader.SUB_CODE] != "CF121"]

    # prepare plot
    modality_list = [data_reader.ECOG, data_reader.FMRI, data_reader.MEG]
    ymin = int(fa_df_summ["False Alarm Rate"].min())
    if 0 < ymin:
        ymin = 0
    ymax = fa_df_summ["False Alarm Rate"].max()
    ymax = int(math.ceil(ymax))

    # aggregate extremes
    extreme_datapoints = list()
    fa_threshold = 15

    for mod in modality_list:
        df = fa_df_summ[fa_df_summ[data_reader.MODALITY] == mod]
        if not df.empty:
            ymin_mod = 0
            ymax_mod = int(math.ceil(df["False Alarm Rate"].max()))
            if mod == data_reader.ECOG:
                skip = 5
            else:
                skip = 1
            single_plot_per_cat_relevance(data=df, y_col="False Alarm Rate", ymin=ymin_mod, ymax=ymax_mod,
                                          plot_title=f"False Alarm Rate per Stimulus Category and Relevance {mod}",
                                          plot_y_label="False Alarm Rate (%)", save_path=save_path, leg=False,
                                          skip=skip, save_name=f"{save_name}_{mod}")

            # give some information about exceptional subjects
            df_exceptional_mod = df[df["False Alarm Rate"] > fa_threshold]
            extreme_datapoints.append(df_exceptional_mod)

            # (2) breakdown per lab
            # get the modalities' min max
            mod_ymin = int(df["False Alarm Rate"].min())
            if 0 < mod_ymin:
                mod_ymin = 0
            mod_ymax = df["False Alarm Rate"].max()
            mod_ymax = int(math.ceil(mod_ymax))
            # labs
            labs = df.lab.unique().tolist()
            for lab in labs:
                df_lab = df[df[data_reader.LAB] == lab]
                single_plot_per_cat_relevance(data=df_lab, y_col="False Alarm Rate", ymin=mod_ymin, ymax=mod_ymax,
                                              plot_title=f"False Alarm Rate per Stimulus Category and Relevance {lab}",
                                              plot_y_label="False Alarm Rate (%)", save_path=save_path,
                                              save_name=f"{save_name}_{mod}_{lab}")

    df_exceptional = pd.concat(extreme_datapoints, ignore_index=True)
    return df_exceptional


def process_data_for_analysis(sub_dict, valid_sub_list, save_path, phase_name):
    """
    Create dataframes to be analyzed in R with linear mixed-effects models. Aggregate all VALID subject data (based
    on the list), and save different dfs for different analyses.
    * UPDATED : 2023-01-25 PREREG 4.0 *
    THIS FOLLOWS PRE-REGISTRATION 4.0'S BEHAVIORAL ANALYSIS SECTION, and prepares the data for modelling in R.
    """
    # create lists to aggregate analysis data across subjects
    d_prime_cat_dur_mod_list = list()  # for analysis of d' ~ category + duration + modality
    fa_rel_cat_list = list()
    hit_rt_list = list()

    for subject in valid_sub_list:  # ONLY GO OVER SUBJECTS WHO **PASSED** THE BEHAVIORAL DATA QUALITY CHECKS
        # then, prepare data for behavioral analysis across subjects

        # FIRST MODEL
        d_prime_cat_dur_table = dprime_by_cat_dur_mod(sub_dict[subject])
        d_prime_cat_dur_mod_list.append(d_prime_cat_dur_table)  # summary table (d' per category x duration x modality)

        # SECOND MODEL
        fa_rel_cat_table = fa_by_cat_tr(sub_dict[subject])
        fa_rel_cat_list.append(fa_rel_cat_table)  # summary table (false-alarm rates per category x task relevance)

        # THIRD MODEL
        hit_rt_table = hit_rt_by_cat_mod(sub_dict[subject])
        hit_rt_list.append(hit_rt_table)  # filtered table (row per trial, all relevant trials)

    # analysis dataframes

    d_prime_cat_dur_mod_df = pd.concat(d_prime_cat_dur_mod_list, ignore_index=True)
    d_prime_cat_dur_mod_df.to_csv(os.path.join(save_path, f"lmm_dprime_cat_dur_mod_{phase_name}.csv"), index=False)
    # for d prime
    d_prime_stats = calc_df_stats_permutations(d_prime_cat_dur_mod_df, dependent_col=D_PRIME, permutation_col_list=["modality", "stimType", "plndStimulusDur"])
    d_prime_stats.to_csv(os.path.join(save_path, f"stats_dprime_cat_dur_mod_{phase_name}.csv"), index=False)
    # for hit rate
    hit_rate_stats = calc_df_stats_permutations(d_prime_cat_dur_mod_df, dependent_col="hit_rate", permutation_col_list=["modality", "stimType", "plndStimulusDur"])
    hit_rate_stats.to_csv(os.path.join(save_path, f"stats_hit_rate_cat_dur_mod_{phase_name}.csv"), index=False)

    fa_rel_cat_df = pd.concat(fa_rel_cat_list, ignore_index=True)
    #fa_stats = calc_df_stats_permutations(fa_rel_cat_df, dependent_col=f"is{FALSEPOSITIVE}", permutation_col_list=["modality", "stimType", "isTaskRelevant"])
    fa_stats = calc_df_stats_permutations_per_sub(fa_rel_cat_df, dependent_col=f"is{FALSEPOSITIVE}", permutation_col_list=["modality", "stimType", "isTaskRelevant"])
    fa_rel_cat_df.to_csv(os.path.join(save_path, f"lmm_fa_cat_rt_{phase_name}.csv"), index=False)
    #fa_stats.to_csv(os.path.join(save_path, f"stats_fa_cat_rt_{phase_name}.csv"), index=False)

    hit_rt_df = pd.concat(hit_rt_list, ignore_index=True)
    rt_stats = calc_df_stats_permutations(hit_rt_df, dependent_col=RT_COL, permutation_col_list=["modality", "stimType", "plndStimulusDur"])
    hit_rt_df.to_csv(os.path.join(save_path, f"lmm_hitRTs_cat_dur_mod_trials_{phase_name}.csv"), index=False)
    rt_stats.to_csv(os.path.join(save_path, f"stats_hitRTs_cat_dur_mod_trials_{phase_name}.csv"), index=False)

    # PLOT

    # As I prefer to plot with Python and not with R (where the modelling will be done), we will generate some plots:
    dprime_exceptions = plot_d_prime(d_prime_cat_dur_mod_df, save_path, f"lmm_dprime_cat_dur_mod_{phase_name}")
    dprime_exceptions.to_csv(os.path.join(save_path, f"lmm_dprime_cat_dur_mod_extreme_{phase_name}.csv"))
    # same for hit rate
    hit_rate_df = d_prime_cat_dur_mod_df[d_prime_cat_dur_mod_df["hit_rate"].notna()]
    plot_hit_rate(hit_rate_df, save_path, f"lmm_hit_rate_cat_dur_mod_{phase_name}")

    fa_exceptions = plot_fa(fa_rel_cat_df, save_path, f"lmm_fa_cat_rt_{phase_name}")
    fa_exceptions.to_csv(os.path.join(save_path, f"lmm_fa_cat_rt_extreme_{phase_name}.csv"))
    fa_single_plot(data=fa_rel_cat_df, plot_title=f"False Alarm Rate by Task Relevance {phase_name}",
                   save_path=save_path, save_name=f"lmm_fa_tr_{phase_name}", skip=10)
    fa_rel_cat_df_noCF121 = fa_rel_cat_df[fa_rel_cat_df[data_reader.SUB_CODE] != "CF121"]
    fa_single_plot(data=fa_rel_cat_df_noCF121, plot_title=f"False Alarm Rate by Task Relevance {phase_name}",
                   save_path=save_path, save_name=f"lmm_fa_tr_{phase_name}_noCF121", skip=1)

    # hit_rt_df IS PER **TRIAL**, for plotting purposes we would like to have it per **SUBJECT**
    rt_exceptions = plot_hit_rt(hit_rt_df, save_path, f"lmm_hitRTs_cat_dur_mod_trials_{phase_name}")
    rt_exceptions.to_csv(os.path.join(save_path, f"lmm_hitRTs_cat_dur_mod_trials_extreme_{phase_name}.csv"))

    return


def check_data(root_folder=data_reader.COGITATE_PATH, phase_3=False, phase_name=""):
    """
    This function triggers all the quality checks that will be done on the data.
    :param: root_folder: path to the folder in which all the subject data lies (Cogitate folder on HPC)
    :param: phase_3: True if we want phase 3 subjects, false for phase 2
    :param: phase_name: string, name of the phase for file suffix
    :return: the summary table of the QC results, where each row = subject and each column = stat
    """

    print("********* Extracting Data *********")
    qc_res_path = data_saver.create_hpc_quality_checks(root_folder)  # the result folder on the HPC (DMT QC folder)

    # STEP 1: load and save all subject data
    subject_dict = extract_data(root_folder=root_folder)
    # Save the subject struct to a pickle
    file_name = f'subject_beh.pickle'
    fl = open(os.path.join(qc_res_path, file_name), 'wb')  # 'ab' apppends new info to the existing file; 'wb' overwrites the entire file
    pickle.dump(subject_dict, fl)
    fl.close()

    # STEP 2: BEHAVIORAL QC
    print("********* Behavioral QC *********")
    # phase 3 subject lists
    if phase_3:
        # INTERMEDIATE STEP - TAKE ONLY PHASE 3 LEVEL SUBJECTS
        print("------ PHASE 3 ONLY ------")
        subjects_phase_3_ecog = [line.strip() for line in open(PHASE_3_PATH[data_reader.ECOG], 'r')]  # ECoG are all phase 3
        # in ECoG for some reason, the file contains "sub-CCC" instead of just the sub code
        subjects_phase_3_ecog = [subcode.replace("sub-", "") for subcode in subjects_phase_3_ecog]
        subjects_phase_3_ecog = [subcode.strip() for subcode in subjects_phase_3_ecog]
        subjects_phase_3_mri = [line.strip() for line in open(PHASE_3_PATH[data_reader.FMRI], 'r')]
        subjects_phase_3_meg = [line.strip() for line in open(PHASE_3_PATH[data_reader.MEG], 'r')]
        phase_3_subject_list = subjects_phase_3_mri + subjects_phase_3_meg + subjects_phase_3_ecog
        phase3_subject_dict = {sub: subject_dict[sub] for sub in phase_3_subject_list}
        print(f"{len(phase_3_subject_list)} subjects in phase 3 to be checked")
        filename_phase = FILENAME_PHASE3
        sub_dict = phase3_subject_dict

    else:
        print("------ PHASE 2 ONLY ------")
        subjects_phase_2_mri = [line.strip() for line in open(PHASE_2_PATH[data_reader.FMRI], 'r')]
        subjects_phase_2_meg = [line.strip() for line in open(PHASE_2_PATH[data_reader.MEG], 'r')]
        phase_2_subject_list = subjects_phase_2_mri + subjects_phase_2_meg
        phase2_subject_dict = {sub: subject_dict[sub] for sub in phase_2_subject_list}
        print(f"{len(phase2_subject_dict)} subjects in phase 2 to be checked")
        filename_phase = FILENAME_PHASE2
        sub_dict = phase2_subject_dict

    qc_dataframe = process_data_for_qc(sub_dict)

    data_table_res = quality_checks_criteria.check_data_table(qc_dataframe)
    data_saver.safe_save(qc_res_path, filename_phase)  # check if there's an existing table with that name and alert if so
    data_table_res.to_csv(os.path.join(qc_res_path, filename_phase), index=False)

    # STEP 3: PREPARE DATA FOR BEHAVIORAL DATA ANALYSIS
    print("********* Behavioral Analysis Preparation *********")
    valid_subs = data_table_res[data_table_res[quality_checks_criteria.VALID] == True]  # TAKE ONLY VALID SUBJECTS!
    """
    As ECoG patients are valuable, it was decided in Paris to *NOT* exclude patients EVEN IF THEIR BEHAVIOR IS INVALID.
    Therefore, if we are in phase 3, we need to RE-INCLUDE the ECoG subjects that were just now excluded.
    """
    if phase_3:
        # take excluded subjects WHO ARE ECOG SUBJECTS
        excluded_ecog_subs = data_table_res[(data_table_res[quality_checks_criteria.VALID] != True) & (data_table_res[data_reader.MODALITY] == data_reader.ECOG)]
        # re-include them - take them as valid (but first print their names)
        print("WARNING: these iEEG patients' behavior is INVALID, but they will be included in the analysis anyway")
        print(excluded_ecog_subs.loc[:, data_reader.SUB_CODE].tolist())
        excluded_ecog_subs.to_csv(os.path.join(qc_res_path, "quality_checks_phase3_included_invalid_ecog.csv"), index=False)
        valid_subs = valid_subs.append(excluded_ecog_subs)

    valid_sub_list = valid_subs.loc[:, data_reader.SUB_CODE].tolist()
    process_data_for_analysis(sub_dict=subject_dict, valid_sub_list=valid_sub_list, save_path=qc_res_path, phase_name=phase_name)

    return data_table_res


def calc_df_stats_permutations(df, dependent_col, permutation_col_list):
    """
    This function calculates the mean and std of a single column (dependent_col) in a dataframe (df),
    across all combinations of column values in permutation_col_list.
    (e.g., for face x 1.5 dur x task relevant - average and sd of dependent_col.

    df: a dataframe containing all the columns in permutation_col_list PLUS a dependent variable column (dependent_col)
    dependent_col: the column with the data we want to average (sd) across some conditions (permutation_col_list)
    permutation_col_list: all the columns that by which the combinations should be calculated
    """
    lst = list()
    x = pd.DataFrame(columns=permutation_col_list)  # exists only in order to order the columns
    lst.append(x)
    for i in range(1, len(permutation_col_list) + 1):
        for comb in list(itertools.combinations(permutation_col_list, i)):
            comb_list = list(comb)
            means = df.groupby(comb_list).mean(numeric_only=True).reset_index()
            stds = df.groupby(comb_list).std(numeric_only=True).reset_index()
            means[f"{dependent_col}_std"] = stds[dependent_col]
            means.rename({dependent_col: f"{dependent_col}_mean"}, axis=1, inplace=True)
            means = means.loc[:, comb_list+[f"{dependent_col}_mean", f"{dependent_col}_std"]]
            lst.append(means)

    result = pd.concat(lst)
    return result


def calc_df_stats_permutations_per_sub(df, dependent_col, permutation_col_list):
    """
    This function calculates the mean and std of a single column (dependent_col) in a dataframe (df),
    across all combinations of column values in permutation_col_list.
    (e.g., for face x 1.5 dur x task relevant - average and sd of dependent_col.

    df: a dataframe containing all the columns in permutation_col_list PLUS a dependent variable column (dependent_col)
    dependent_col: the column with the data we want to average (sd) across some conditions (permutation_col_list)
    permutation_col_list: all the columns that by which the combinations should be calculated
    """
    lst = list()
    x = pd.DataFrame(columns=permutation_col_list)  # exists only in order to order the columns
    lst.append(x)
    for i in range(1, len(permutation_col_list) + 1):
        for comb in list(itertools.combinations(permutation_col_list, i)):
            comb_list = list(comb)
            comb_list.append(data_reader.SUB_CODE)
            means = df.groupby(comb_list).mean(numeric_only=True).reset_index()
            stds = df.groupby(comb_list).std(numeric_only=True).reset_index()
            means[f"{dependent_col}_std"] = stds[dependent_col]
            means.rename({dependent_col: f"{dependent_col}_mean"}, axis=1, inplace=True)
            means = means.loc[:, comb_list+[f"{dependent_col}_mean", f"{dependent_col}_std"]]
            lst.append(means)

    result = pd.concat(lst)
    return result



if __name__ == "__main__":
    # phase 2
    check_data(root_folder=r"/mnt/beegfs/XNAT/COGITATE", phase_name="phase2", phase_3=False)
    # phase 3
    check_data(root_folder=r"/mnt/beegfs/XNAT/COGITATE", phase_name="phase3", phase_3=True)


