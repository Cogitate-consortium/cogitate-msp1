"""
This scripts contains all the helper functions for the activation analysis
    authors: Alex Lepauvre
    alex.lepauvre@ae.mpg.de
"""
import os
# os.environ['R_HOME'] = '/hpc/users/alexander.lepauvre/.conda/envs/pymer4/bin'
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp, wilcoxon

import mne.stats

from general_helper_functions.data_general_utilities import (compute_dependent_variable, moving_average, load_epochs,
                                                             moving_average)


def duration_decoding(epochs, channel, metadata, labels_condition="duration", shuffle_label=False, binning_ms=50,
                      do_diff=False, n_folds=5, time_win=None, classifier="svm"):
    """
    This function performs decoding of duration on single channels
    :param epochs: (mne epochs object) contains the mne epochs object
    :param channel: (string) name of the channel on which to perform the decoding
    :param metadata: (pandas dataframe) contains the trials metadata
    :param labels_condition: (string) name of the metadata column to use as labels
    :param shuffle_label: (boolean) whether or not to shuffle the labels before performing the decoding
    :param binning_ms: (int) binning for moving average
    :param do_diff: (boolean) whether or not to compute the diff before doing the decoding
    :param n_folds: (int) number of folds for the classifier
    :param time_win: (list of floats) time window over which to perform the decoding
    :param classifier: (string) name of the classifier to use
    :return:
    """
    if time_win is None:
        time_win = [0, 2.0]
    print("=" * 40)
    print("Welcome to duration decoding")
    print("Computing duration tracking for channel " + channel)
    # Extract the data:
    data = np.squeeze(epochs.copy().crop(*time_win).get_data(picks=channel))
    # Compute a moving average:
    if binning_ms is not None:
        n_samples = int(np.floor(binning_ms * epochs.info["sfreq"] / 1000))
        data = moving_average(data, n_samples, axis=-1, overlapping=False)
    if do_diff:
        data = np.diff(data, n=1, axis=-1)
    # Extract the labels:
    labels = metadata[labels_condition].to_numpy()
    # Shuffling the labels if needed:
    if shuffle_label:
        labels = labels[np.random.permutation(len(labels))]
    # Generate the pipeline:
    if classifier.lower() == "svm":
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC()
        )
    elif classifier.lower() == "logisticregression":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear')
        )
    else:
        raise Exception("The classifier passed was not recogized! Only SVM or logisticregression supported")
    # Score the data:
    scores = cross_val_score(clf, data, labels, cv=n_folds)

    # Get the results:
    # Creating a dataframe:
    # Compute the proportion of trials for which the duration tracking is within threshold:
    results = pd.DataFrame({
        "channel": channel,
        "decoding_accuracy": np.mean(scores)
    }, index=[0]
    )

    return results


def compute_tracking_inaccuracy(epochs, subject, channel, metadata, threshold_condition="1500ms", baseline_time=None,
                                activation_time=None, shuffle_label=False, inaccuracy_threshold=0.15,
                                fast_track=False):
    """
    This function computes duration tracking inaccuracy by computing a threshold as the median between baseline
    and a given time window within the trial. For each trial, the first time point at which the activation drops below
    the threshold his computed. This time point is then compared to the duration of the trial, to estimate how well
    duration was tracked in that specific trial. This analysis is based on
    https://www.sciencedirect.com/science/article/pii/S1053811917306754
    :param epochs: (mne epochs object) contains the data for which to perform the computations
    :param subject: (string) name of the subject for which to compute duration tracking
    :param channel: (string) name of the channel for which to compute duration tracking
    :param metadata: (pandas dataframe) metadata of the trial. This is the metadata from the epoch object but with
    the indices being reset to make sure the right trials get accessed
    :param threshold_condition: (string) condition for which to compute the threshold. This should match an existing
    event
    :param baseline_time: (list of two floats) time points for which to compute baseline
    :param activation_time: (list of two floats) time points for which to compute the activation. The median
    between this and baseline will be taken as thresholdd
    :param shuffle_label: (boolean) whether or not to shuffle the labels. This is useful to generate a null distribution
    :param inaccuracy_threshold: (float)
    :param fast_track: (boolean) whether or not to use a shortcut. Instead of computing the threshold on single trials,
    computed for all trials. It is much better but a bit less clean
    :return: results (pandas dataframe) contains the results
    """
    print("=" * 40)
    print("Welcome to duration tracking")
    print("Computing duration tracking for channel " + channel)
    if baseline_time is None:
        baseline_time = [-0.3, 0]
    if activation_time is None:
        activation_time = [1.3, 1.5]
    if shuffle_label:
        metadata = metadata.sample(frac=1).reset_index(drop=True)
    # Prepare results data frame:
    results = pd.DataFrame()

    if fast_track:
        # In the fast track option, not looping through every single trial but generating one threshold for all:
        threshold_data = epochs[threshold_condition]
        baseline_data = \
            threshold_data.copy().crop(tmin=baseline_time[0],
                                       tmax=baseline_time[1]).get_data(
                picks=channel)
        activation_data = \
            threshold_data.copy().crop(tmin=activation_time[0],
                                       tmax=activation_time[1]).get_data(picks=channel)
        # Compute the activation threshold:
        if baseline_data.shape[-1] != activation_data.shape[-1]:
            print("WARNING: The baseline and activation data have different number of samples: ")
            print("Baseline data: {}".format(baseline_data.shape[-1]))
            print("activation data: {}".format(activation_data.shape[-1]))
            print("The data will be downsampled to the smallest one!")
            min_n_samples = min([baseline_data.shape[-1], activation_data.shape[-1]])
            baseline_data = baseline_data[:, :, :min_n_samples]
            activation_data = activation_data[:, :, :min_n_samples]
        threshold = np.median([np.mean(np.squeeze(baseline_data), axis=0),
                               np.mean(np.squeeze(activation_data), axis=0)])
        # For all trials, check when they go below threshold:
        cropped_epoch = epochs.copy().crop(tmin=0)
        inds = np.argmax(np.squeeze(cropped_epoch.get_data(picks=channel)) < threshold, axis=1)
        # Then, get the onset in ms:
        drop_onsets = [cropped_epoch.times[ind] for ind in inds]
        # Get the tracking inaccuracies:
        tracking_inaccuracies = [np.abs(drop_onset -
                                        int(''.join(i for i in metadata.loc[ind, "duration"] if i.isdigit()))
                                        * 10 ** -3) for ind, drop_onset in enumerate(drop_onsets)]
        # Creating a dataframe:
        results = results.append(pd.DataFrame({
            "subject": subject,
            "channel": channel,
            "trial": metadata.index,
            "condition": metadata["duration"].to_list(),
            "drop_onset": drop_onsets,
            "tracking_inaccuracy": tracking_inaccuracies
        }))

    else:
        # Now, looping through every trial:
        for epoch_ind, row in metadata.iterrows():
            # Get the data from the threshold condition (data of the threshold condition to the
            # exception of the current trial if it is one of the threshold condition):
            threshold_data = epochs.copy().drop(epoch_ind, verbose="ERROR")[threshold_condition]
            # Get the baseline:
            baseline_data = \
                threshold_data.copy().crop(tmin=baseline_time[0],
                                           tmax=baseline_time[1]).get_data(
                    picks=channel)
            activation_data = \
                threshold_data.copy().crop(tmin=activation_time[0],
                                           tmax=activation_time[1]).get_data(picks=channel)
            # Compute the activation threshold:
            threshold = np.median([np.mean(np.squeeze(baseline_data), axis=0),
                                   np.mean(np.squeeze(activation_data), axis=0)])
            # Now, checking at which time for this trial we drop below the threshold:
            cropped_epoch = epochs.copy().crop(tmin=0)
            ind = np.argmax(np.squeeze(cropped_epoch.get_data(picks=channel))[epoch_ind, :] < threshold)
            drop_onset = cropped_epoch.times[ind]
            # Compute the inaccuracy, i.e. the drop onset minus the actual trial duration:
            tracking_inaccuracy = np.abs(drop_onset -
                                         int(''.join(i for i in row["duration"] if i.isdigit()))
                                         * 10 ** -3)
            # Store the results:
            results = results.append(pd.DataFrame({
                "subject": subject,
                "channel": channel,
                "trial": epoch_ind,
                "condition": row["duration"],
                "drop_onset": drop_onset,
                "tracking_inaccuracy": tracking_inaccuracy
            }, index=[0]))

    # Compute the proportion of trials for which the duration tracking is within threshold:
    tracking_proportion = pd.DataFrame({
        "channel": channel,
        "tracking_proportion": np.sum(results["tracking_inaccuracy"].to_numpy() <= inaccuracy_threshold) / len(
            results["tracking_inaccuracy"].to_numpy())}, index=[0]
    )

    return results, tracking_proportion


def test_sustained_threshold(y, stat_test="t-test", threshold=0.05, window_sec=0.05, sr=512,
                             alternative="two-sided", fdr_method="fdr_bh"):
    """
    This function computes a sliding statistical test on the y data and checks whether the results are significant for
    window_sec or longer.
    :param y: (2D numpy array) data on which to run the test
    :param stat_test: (string) name of the statistical test. "t-test" and "wilcoxon" are supported.
    :param threshold: (string) p value threshold to consider something significant
    :param window_sec: (float) for how long the pvalues must be above threshold to be considered significant
    :param sr: (int) sampling rate of y
    :param alternative: (string) alternative of the statstical test: "two-sided", "upper", "lower
    :param fdr_method: (string) which method to use for FDR correction across time points.
    :return:
    h0: boolean, whether or not the test is considered significant
    [onset_sec, offset_sec]: onset and offset of the significant chunk in seconds
    [onset_samp, offset_samp]: onset and offset of the significant chunk in samples
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
        y_stat, y_pval = ttest_1samp(y, 0, axis=1, alternative=alternative)
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


def moving_window_test(data_df, onset, groups="channel", thresh=0.05, dur_thresh=0.050, alternative="upper",
                       sfreq=512, fdr_method="fdr_bh", stat_test="t-test"):
    """
    This function performs a sliding test on the data_df to check whether the results are significant for dur_thresh
    or longer
    :param data_df: (pandas data frame) data on which to run the sliding test
    :param onset: (float) onset of the data, i.e. time of the first time point
    :param groups: (string) name of the group variable on which to run the single tests
    :param thresh: (float) threshold to consider something significant
    :param dur_thresh: (float) duration for which the threshold must be sustained to be significant
    :param alternative: (string) "greater", "lower" or "two_tailed"
    :param sfreq: (int) sampling frequency of the data
    :param fdr_method: (string) multiple comparison correction method. See
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    for the otpions
    :param stat_test: (string) which statistical test to use "t-test" or "wilcoxon"
    :return:
    test_results (pandas data frame) contains the results of the test
    """
    print("=" * 40)
    print("Welcome to moving_window_test")
    # Var to store the results
    test_results = pd.DataFrame()
    for group in data_df[groups].unique():
        print("Performing test for group: {}".format(group))
        # Get the data of this group:
        y = data_df.loc[data_df[groups] == group, "values"].item()
        # Testing the sustained
        h0, sig_window_sec, sig_window_samp = test_sustained_threshold(y,
                                                                       threshold=thresh,
                                                                       window_sec=dur_thresh,
                                                                       sr=sfreq,
                                                                       alternative=alternative,
                                                                       stat_test=stat_test,
                                                                       fdr_method=fdr_method)

        # Create results table:
        test_results = test_results.append(pd.DataFrame({
            "subject": group.split("-")[0],
            "channel": group,
            "metric": None,
            "stat": None,
            "pval": None,
            "reject": not h0,
            "onset": onset + sig_window_sec[0] if sig_window_sec[0] is not None
            else None,
            "offset": onset + sig_window_sec[1] if sig_window_sec[0] is not None
            else None,
        }, index=[0]))

    return test_results


def model_comparison(models_results, criterion="aic", test="linear_mixed_model"):
    """
    The model results contain columns for fit criterion (log_likelyhood, aic, bic) that can be used to investigate
    which model had the best fit. Indeed, because we are trying several models, if several of them are found to be
    significant in the coefficients of interest, we need to arbitrate betweem them. This function does that by
    looping through each channel and checking whether or not more than one model was found signficant. If yes,
    then the best one is selected by checking the passed criterion. The criterion must match the string of one of the
    column of the models_results dataframe
    :param models_results: (dataframe) results of the linear mixed models, as returned by the fit_single_channels_lmm
    function
    :param criterion: (string) name of the criterion to use to arbitrate between models
    :param test: (string) type of test (i.e. model that was preprocessing)
    example, if you have ran several models per channels, pass here channels and it will look separately at each channel
    :return:
    best_models (pandas data frame) contains the results of the best models only, i.e. one model per channel only
    """
    print("-" * 40)
    print("Welcome to model comparison")
    print("Comparing the fitted {} using {}".format(test, criterion))
    # Declare dataframe to store the best models only:
    best_models = pd.DataFrame(columns=models_results.columns.values.tolist())
    # Removing any model that didn't converge in case a linear mixed model was used:
    if test == "linear_mixed_model":
        converge_models_results = models_results.loc[models_results["converged"]]
    else:
        converge_models_results = models_results
    # In the linear mixed model function used before, the fit criterion are an extra column. Therefore, for a given
    # electrode, the best fit is any row of the table that has the max of the criterion. Therefore, looping over
    # the data:
    for channel in converge_models_results["group"].unique():
        # Getting the results for that channel only
        data = converge_models_results.loc[converge_models_results["group"] == channel]
        # Extracting the rows with highest criterion
        best_model = data.loc[data[criterion] == np.nanmin(data[criterion])]
        # Adding it to the best_models dataframe, storing all the best models:
        best_models = pd.concat([best_models, best_model], ignore_index=True)
    return best_models


def fit_lmm(data, models, re_group, group="", alpha=0.05, package="lmer"):
    """
    This function fits the different linear mixed models passed in the model dict on the data
    :param data: (pandas data frame) contains the data to fit the linear mixed model on
    :param models: (dict) contains the different models:
    "null_model": {
        "model": "value ~ 1",
        "re_formula": null
    },
    "time_win": {
        "model": "value ~ time_bin",
        "re_formula": null
    },
    "duration": {
        "model": "value ~ duration",
        "re_formula": null
    },
    the key of each is the name of the model (used to identify it down the line), the model is the formula, the
    re_formula is for the random slopes
    :param re_group: (string) name of the random effect group. If you have measure repeated within trials, this should
    be trial for example
    :param group: (string) name of the column from the data table that corresponds to the groups for which to run the
    model separately. You can run it on single channels, in which case group must be "channel"
    :param alpha: (float) alpha to consider significance. Not really used
    :return:
    """
    print("-" * 40)
    print("Welcome to fit_lmm")
    results = pd.DataFrame()
    anova_results = pd.DataFrame()
    # Looping through the different models to apply to the data of that particular channel:
    for model in models.keys():
        if package == "stats_model":
            print("Fitting {} model to group {}".format(model, group))
            # Applying the linear mixed model specified in the parameters:
            md = smf.mixedlm(models[model]["model"],
                             data, groups=re_group, re_formula=models[model]["re_formula"])
            # Fitting the model:
            mdf = md.fit(reml=False)
            # Printing the summary in the command line:
            print(mdf.summary())
            # Compute the r2:
            # r2 = compute_lmm_r2(mdf)
            # Extracting the results and storing them to the dataframe:
            results = results.append(pd.DataFrame({
                "subject": group.split("-")[0],
                "analysis_name": ["linear_mixed_model"] * len(mdf.pvalues),
                "model": [model] * len(mdf.pvalues),
                "group": [group] * len(mdf.pvalues),
                "coefficient-conditions": mdf.params.index.values,
                "Coef.": mdf.params.values,
                "Std.Err.": mdf.bse.values,
                "z": mdf.tvalues.values,
                "p-value": mdf.pvalues.values,
                "reject": [True if p_val < alpha else False for p_val in mdf.pvalues.values],
                "converged": [mdf.converged] * len(mdf.pvalues),
                "log_likelyhood": [mdf.llf] * len(mdf.pvalues),
                "aic": [mdf.aic] * len(mdf.pvalues),
                "bic": [mdf.bic] * len(mdf.pvalues)
            }), ignore_index=True)
        elif package == "lmer":
            from pymer4.models import Lmer
            # Set the epoch to strings:
            data["epochs"] = data["epoch"].astype(str)
            # Fit the model:
            mdl = Lmer(models[model]["model"], data=data)
            print(mdl.fit(REML=False))
            # Append the coefs to the results table:
            coefs = mdl.coefs
            results = results.append(pd.DataFrame({
                "subject": group.split("-")[0],
                "analysis_name": ["linear_mixed_model"] * len(coefs["Estimate"]),
                "model": [model] * len(coefs["Estimate"]),
                "group": [group] * len(coefs["Estimate"]),
                "coefficient-conditions": coefs.index.values,
                "Coef.": coefs["Estimate"].to_list(),
                "T-stat": coefs["T-stat"].to_list(),
                "p-value": coefs["P-val"].to_list(),
                "reject": [True if p_val < alpha else False for p_val in coefs["P-val"].to_list()],
                "converged": [True] * len(coefs["Estimate"]),
                "log_likelyhood": [mdl.logLike] * len(coefs["Estimate"]),
                "aic": [mdl.AIC] * len(coefs["Estimate"]),
                "bic": [mdl.BIC] * len(coefs["Estimate"])
            }), ignore_index=True)

            # In addition, run the anova on the model to extract the main effects:
            anova_res = mdl.anova()
            # For the null model, since there are no main effects, the anova results are empty:
            if len(anova_res) == 0:
                anova_results = anova_results.append(pd.DataFrame({
                    "subject": group.split("-")[0],
                    "analysis_name": "anova",
                    "model": model,
                    "group": group,
                    "conditions": np.nan,
                    "F-stat": np.nan,
                    "p-value": np.nan,
                    "reject": np.nan,
                    "converged": [True] * len(coefs["Estimate"]),
                    "SS": np.nan,
                    "aic": mdl.AIC,
                    "bic": mdl.BIC
                }, index=[0]), ignore_index=True)
            else:
                anova_results = anova_results.append(pd.DataFrame({
                    "subject": group.split("-")[0],
                    "analysis_name": ["anova"] * len(anova_res),
                    "model": [model] * len(anova_res),
                    "group": [group] * len(anova_res),
                    "conditions": anova_res.index.values,
                    "F-stat": anova_res["F-stat"].to_list(),
                    "p-value": anova_res["P-val"].to_list(),
                    "reject": [True if p_val < alpha else False for p_val in anova_res["P-val"].to_list()],
                    "converged": [True] * len(anova_res),
                    "SS": anova_res["SS"].to_list(),
                    "aic": [mdl.AIC] * len(anova_res),
                    "bic": [mdl.BIC] * len(anova_res)
                }), ignore_index=True)

    return results, anova_results


def create_theories_predictors(df, predictors_mapping):
    """
    This function adds predictors to the data frame based on the predictor mapping passed. The passed predictors
    consist of dictionaries, providing mapping between specific experimental condition and a specific value to give it.
    This function therefore loops through each of the predictor and through each of the condition of that predictor.
    It will then look for the condition combination matching it to attribute it the value the predictor dictates.
    Example: one predictor states: faces/short= 1, faces/intermediate=0... This is what that function does
    DISCLAIMER: I know using groupby would be computationally more efficient, but this makes for more readable and easy
    to encode the predictors, so I went this way.
    :param df: (data frame) data frame to add the predictors to
    :param predictors_mapping: (dict of dict) One dictionary per predictor. For each predictor, one dictionary
    containing
    mapping between condition combination and value to attribute to it
    :return: (dataframe) the data frame that was fed in + predictors values
    """
    print("-" * 40)
    print("Creating theories' derived predictors: ")
    # Getting the name of the columns which are not the ones automatically ouputed by mne, because these are the ones
    # we created and that contain the info we seek:
    col_list = [col for col in df.columns if col not in [
        "epoch", "channel", "value", "condition"]]
    # Looping through the predictors:
    for predictor in predictors_mapping.keys():
        df[predictor] = np.nan
        # Now looping through the key of each predictor, as this contains the mapping info:
        for key in predictors_mapping[predictor].keys():
            # Finding the index of each row matching the key:
            bool_list = \
                [all(x in list(trial_info[col_list].values)
                     for x in key.split("/"))
                 for ind, trial_info in df.iterrows()]
            # Using the boolean list to add the value of the predictor in the concerned row:
            df.loc[bool_list, predictor] = predictors_mapping[predictor][key]

    return df


def segment_epochs(epochs, time_bins=None):
    """
    This function parses the epochs object by conditions and time bins. The mne epochs object has events of specific
    categories. The time bins set which chunk of the epochs should be taken.
    :param epochs: (mne epochs object) epoch object to segment
    :param time_bins: dictionary of time bins. Each entry of the list should contain the onset and offset of the window
    if not specified, the entire epoch will be retrieved.
    :return: data: (dict of epochs) dictionary containing the cropped epochs, the key being an underscore separated
    string from the time boundaries of each segments
    """
    print("-" * 40)
    print("Segmenting the epochs: ")
    if time_bins is None:
        time_bins = [epochs.times[0], epochs.times[-1]]

    # Checking that the correct variables types were fed in:
    if type(time_bins) is not dict:
        raise TypeError("The condition variable should be a dictionary")

    # Creating the dictionary to store the segmented epochs:
    segment_epochs_dict = {}

    # Cropping the data according to the time bins passed to the function. The data will be stored in a dict:
    for time_bin in time_bins.keys():
        segment_epochs_dict[time_bin] = epochs.copy(). \
            crop(time_bins[time_bin][0], time_bins[time_bin][1])

    return segment_epochs_dict


def epochs_mvavg(epochs, window_ms):
    """
    Performs moving average on mne epochs object and returns the epoched object with the moving average data
    :param epochs: (mne epochs object) on which to perform the moving average
    :param window_ms: (int) size of the moving average window in milliseconds
    :return:
    """
    # Convert the window from ms to samples:
    window_samp = int(window_ms * (epochs.info["sfreq"] / 1000))
    # Get the data:
    data = epochs.get_data()
    # Perform the moving average
    data_mvag = moving_average(data, window_samp, axis=-1, overlapping=True)
    # Recreate the epochs array:
    new_epochs = mne.EpochsArray(data_mvag, epochs.info, tmin=epochs.times[0],
                                 events=epochs.events, event_id=epochs.event_id, on_missing="warning",
                                 metadata=epochs.metadata)
    return new_epochs


def prepare_test_data(root, signal, lmm_predictors, metric, segments_time_wins,
                      subject, baseline_method=None, baseline_time=(None, 0), crop_time=None, condition=None,
                      session="V1", task_name="Dur",
                      preprocess_folder="epoching", preprocess_steps="desbadcharej_notfil_autbadcharej_lapref",
                      channel_types=None, select_vis_resp=False, aseg="aparc.a2009s+aseg", montage_space="T1",
                      get_mni_coord=False, picks_roi=None, vis_resp_folder=None, moving_average_ms=None,
                      multitaper_parameters=None, scal=1e0):
    """
    This function loads the epochs and format them according to the test passed
    :param root: (string or pathlib object) path to the bids root
    :param signal: (string) name of the signal to investigate
    :param lmm_predictors: (dict) predictors for the linear mixed models
    :param segments_time_wins: (list of list of floats) time windows to investigate
    :param condition: (list of strings) name of the conditions to use
    :param baseline_method: (string) name of the method to compute the baseline correction, see baseline_rescale from
    mne for more details
    :param metric: (string) name of the method to use to compute the data aggregation if wilcoxon or t_test is passed
    as a test
    :param subject: (string) name of the subject
    :param baseline_time: (list of two floats) onset and offset for baseline correction
    :param crop_time: (list of two floats) time points to crop the epochs
    :param sel_conditions: (string) condition to select epochs from
    :param session: (string) name of the session
    :param task_name: (string) name of the task
    :param preprocess_folder: (string) name of the preprocessing folder
    :param preprocess_steps: (string) name of the preprocessing step to use
    :param channel_types: (dict or None) channel_type: True for the channel types to load
    :param get_mni_coord: (boolean) whether or not to get the mni coordinates for the channels
    :param select_vis_resp: (boolean) whether to use visual responsiveness as a filter for channels to load
    :param vis_resp_folder: boolean) the path to the visual responsiveness folder if visual responsiveness
    is to be used a filter for channels to load
    :param get_mni_coord: (boolean) whether or not to return the MNI coordinates!
    :param montage_space: (string) space of the electrodes localization, either T1 or MNI
    :param picks_roi: (list) list of ROI according to aseg to get the electrodes from. The epochs will be returned only
    with electrodes within this roi list
    :param moving_average_ms: (int) milliseconds for the moving average
    :param multitaper_parameters: (dict) contains info to filtering steps on the data to get specific frequency bands
    {
        "freq_range": [8, 13],
        "step": 1,
        "n_cycle_denom": 2,
        "time_bandwidth": 4.0
    }
    :param aseg: (string) which segmentation to use. Relevant if you want to get channels only from a given ROI
    :param scal: (float) scale of the data if rescale needed
    :return:    :return:
    """
    print("=" * 40)
    print("Preparing sub-{} data".format(subject))
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    # Load the data of the relevant subject with the right parameters (baseline correction...)
    epochs, mni_coord = load_epochs(root, signal, subject, session=session, task_name=task_name,
                                    preprocess_folder=preprocess_folder,
                                    preprocess_steps=preprocess_steps, channel_types=channel_types,
                                    condition=condition,
                                    baseline_method=baseline_method, baseline_time=baseline_time, crop_time=crop_time,
                                    select_vis_resp=select_vis_resp,
                                    vis_resp_folder=vis_resp_folder,
                                    aseg=aseg, montage_space=montage_space, get_mni_coord=get_mni_coord,
                                    picks_roi=picks_roi, filtering_parameters=multitaper_parameters)
    if epochs is None:
        return None, None, None
    # Smooth the data if needed:
    if moving_average_ms is not None:
        epochs = epochs_mvavg(epochs, moving_average_ms)
    # Format the data for the linear mixed model:
    # First, segment the data:
    segmented_data = segment_epochs(
        epochs, segments_time_wins)
    # Compute the matrices for the linear mixed model:
    data_df = pd.DataFrame()
    # We want to keep all the meta data:
    meta_data_cols = list(epochs.metadata.columns)
    for segment in segmented_data.keys():
        segment_df = compute_dependent_variable(segmented_data[segment], metric=metric,
                                                conditions=meta_data_cols)
        # Add the time bin for later:
        segment_df["time_bin"] = segment
        segment_df["subject"] = subject
        # Append to the big dataframe:
        data_df = data_df.append(
            segment_df, ignore_index=True)
    # Adding the additional predictors:
    data_df = create_theories_predictors(data_df, lmm_predictors)

    # Scale the data:
    data_df["value"] = data_df["value"].apply(lambda x: x*scal)

    return data_df, epochs.info["sfreq"], mni_coord


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
        if bs.shape[1] != ons.shape[1]:
            min_len = min([bs.shape[1], ons.shape[1]])
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
