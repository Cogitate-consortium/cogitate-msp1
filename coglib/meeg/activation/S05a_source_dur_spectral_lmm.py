"""
=======================================
S06. Source spectal activation analysis
=======================================

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import ptitprince as pt  #conda install -c conda-forge ptitprince

from pymer4.models import Lmer  #conda install -c ejolly -c conda-forge -c defaults pymer4

import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root


# Set params
visit_id = "V1"


debug = False
bootstrap = False

factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant target','Relevant non-target','Irrelevant'],
              ['500ms', '1000ms', '1500ms']]


# Set participant list
phase = 3

if debug:
    sub_list = ["CA124", "CA124"]
elif bootstrap:
    # Read the .txt file
    f = open(op.join(bids_root,
                  'participants_MEG_phase3_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")
    # Rename phase variable with the random name assigned to the bootstrap (MANUAL STEP!)
    phase = "bsSGa5"
else:
    # Read the .txt file
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")


def run_source_dur_activation(task_rel, tbins):
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

    # Read list with label names
    bids_path_source = mne_bids.BIDSPath(
                        root=source_deriv_root,
                        subject=sub_list[0],
                        datatype='meg',
                        task=bids_task,
                        session=visit_id,
                        suffix="desc-labels",
                        extension='.txt',
                        check=False)
    labels_names = open(bids_path_source.fpath, 'r').read()
    labels_names = labels_names[2:-2].split("', '")

    # Read dataframe
    bids_path_source = mne_bids.BIDSPath(
        root=source_deriv_root,
        subject=f"groupphase{phase}",
        datatype='meg',
        task=bids_task,
        session=visit_id,
        suffix="datatable",
        extension='.tsv',
        check=False)
    df = pd.read_csv(bids_path_source.fpath, sep="\t")

    # Rename "False" as "false"
    df.loc[df['Category']==False, 'Category'] = 'false'

    # Average data in the three time window of interest
    times = np.array([float(t) for t in list(df.columns)[1:-5]])
    for tmin, tmax in tbins:
        imin = (np.abs(times - tmin)).argmin()
        imax = (np.abs(times - tmax)).argmin()
        df[f"[{tmin}, {tmax}]"] = np.mean(df.iloc[:,imin:imax],axis=1)

    # Create theory predictors dict
    predictors = {"iit_predictors": {
                    f"{tbins[0]}/500ms": "decativated",
                    f"{tbins[1]}/500ms": "decativated",
                    f"{tbins[2]}/500ms": "decativated",
                    f"{tbins[0]}/1000ms": "ativated",
                    f"{tbins[1]}/1000ms": "decativated",
                    f"{tbins[2]}/1000ms": "decativated",
                    f"{tbins[0]}/1500ms": "ativated",
                    f"{tbins[1]}/1500ms": "ativated",
                    f"{tbins[2]}/1500ms": "decativated"
                },
                "gnw_predictors": {
                    f"{tbins[0]}/500ms": "ativated",
                    f"{tbins[1]}/500ms": "decativated",
                    f"{tbins[2]}/500ms": "decativated",
                    f"{tbins[0]}/1000ms": "decativated",
                    f"{tbins[1]}/1000ms": "ativated",
                    f"{tbins[2]}/1000ms": "decativated",
                    f"{tbins[0]}/1500ms": "decativated",
                    f"{tbins[1]}/1500ms": "decativated",
                    f"{tbins[2]}/1500ms": "ativated"
                }}

    # Create LMM models dict
    models = create_models()

    # Run analysis
    df_all = pd.DataFrame()
    for band in ['alpha', 'gamma']:
        print('\nfreq_band:', band)

        for label in labels_names:
            print('\nlabel:', label)

            # Select band and label
            df_cond = df.query(f"band == '{band}' and label == '{label}' and Task_relevance == '{task_rel}'")
            df_cond["sub"] = df_cond["sub"].astype(str)

            # Create long table
            df_long = pd.melt(df_cond,
                          id_vars=['sub', 'Category', 'Task_relevance', 'Duration', 'band', 'label'],
                          value_vars=[str(tbins[0]), str(tbins[1]), str(tbins[2])],
                          var_name='time_bin')

            # Create theory predictors
            data_df = create_theories_predictors(df_long, predictors)

            # Append data to list
            df_all = pd.concat([df_all, data_df], ignore_index=True)

            # # Frequency table (used to check the predictors)
            # a = pd.crosstab(index=data_df["iit_predictors"],
            #             columns=[data_df["Duration"], data_df["time_bin"]],
            #             normalize='columns')
            # print(a.iloc[:, 6:9], a.iloc[:, :3], a.iloc[:, 3:6])
            # a = pd.crosstab(index=data_df["gnw_predictors"],
            #             columns=[data_df["Duration"], data_df["time_bin"]],
            #             normalize='columns')
            # print(a.iloc[:, 6:9], a.iloc[:, :3], a.iloc[:, 3:6])

            # Fit linear mixed model
            results, anova = fit_lmm(data_df, models, re_group='sub')

            # Save LMM results
            bids_path_source = bids_path_source.copy().update(
                            root=source_deriv_root,
                            suffix=f"desc-{band},{label},{tbins[0]},{task_rel[:3]}_lmm",
                            extension='.tsv',
                            check=False)
            results.to_csv(bids_path_source.fpath, sep="\t", index=False)

            # Save ANOVA results
            bids_path_source = bids_path_source.copy().update(
                            root=source_deriv_root,
                            suffix=f"desc-{band},{label},{tbins[0]},{task_rel[:3]}_anova",
                            extension='.tsv',
                            check=False)
            anova.to_csv(bids_path_source.fpath, sep="\t", index=False)

            # Compare models
            best_models = model_comparison(results, criterion="bic")

            # Save best LMM model results
            bids_path_source = bids_path_source.copy().update(
                            root=source_deriv_root,
                            suffix=f"desc-{band},{label},{tbins[0]},{task_rel[:3]}_best_model",
                            extension='.tsv',
                            check=False)
            best_models.to_csv(bids_path_source.fpath, sep="\t", index=False)


            # Plot spectral activation time courses

            # Plot 1a #
            # Group by category and duration and averaged across participants
            data_m = df_cond.groupby(['Category','Duration'])[df_cond.keys()[1:-8]].mean()

            # Get 95% condidence intervals
            data_std = df_cond.groupby(['Category','Duration'])[df_cond.keys()[1:-8]].std()
            data_ci = (1.96 * data_std / np.sqrt(len(sub_list)))

            # Cut edges
            tmin = (np.abs(times - -0.5)).argmin()
            tmax = (np.abs(times - 2.5)).argmin()
            t = times[tmin:tmax]

            # Loop over conditions
            fig, axs = plt.subplots(4, 1, figsize=(8,8))
            for c in range(len(conditions[0])):
                print("condition:",conditions[0][c])

                # Get category data
                d500_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '500ms'")
                d1000_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1000ms'")
                d1500_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1500ms'")

                d500_ci = data_ci.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '500ms'")
                d1000_ci = data_ci.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1000ms'")
                d1500_ci = data_ci.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1500ms'")

                # Cut edges
                d500_m = np.squeeze(np.array(d500_m.iloc[:,tmin:tmax]))
                d1000_m = np.squeeze(np.array(d1000_m.iloc[:,tmin:tmax]))
                d1500_m = np.squeeze(np.array(d1500_m.iloc[:,tmin:tmax]))

                d500_ci = np.squeeze(np.array(d500_ci.iloc[:,tmin:tmax]))
                d1000_ci = np.squeeze(np.array(d1000_ci.iloc[:,tmin:tmax]))
                d1500_ci = np.squeeze(np.array(d1500_ci.iloc[:,tmin:tmax]))

                # Plot
                axs[c].plot(t, np.vstack([d500_m, d1000_m, d1500_m]).transpose(), linewidth=2.0)

                for m, ci in zip([d500_m, d1000_m, d1500_m],
                                  [d500_ci, d1000_ci, d1500_ci]):
                    axs[c].fill_between(t, m-ci, m+ci, alpha=.2)

                axs[c].set_xlabel('Time (s)', fontsize='x-large')
                # axs[c].axhline(y=0, color="black", linestyle="--")
                axs[c].axvline(x=0, color="black", linestyle="--")

            for ax in axs.flat:
                ax.set_xlim([-.5, 2.4])
                if band == 'alpha':
                    ax.set_ylim([0.6, 1.4])
                elif band == 'gamma':
                    ax.set_ylim([0.9, 1.1])
                # ax.axvspan(.3, .5, color='grey', alpha=0.25)
                ax.axvspan(tbins[0][0], tbins[0][1], color='red', alpha=0.25)
                ax.axvspan(tbins[1][0], tbins[1][1], color='red', alpha=0.25)
                ax.axvspan(tbins[2][0], tbins[2][1], color='red', alpha=0.25)
                ax.legend(['500ms', '1000ms', '1500ms'], loc='lower left')

            axs[0].set_ylabel('Face', fontsize='x-large', fontweight='bold')
            axs[1].set_ylabel('Object', fontsize='x-large', fontweight='bold')
            axs[2].set_ylabel('Letter', fontsize='x-large', fontweight='bold')
            axs[3].set_ylabel('False-font', fontsize='x-large', fontweight='bold')
            plt.suptitle(f"{band}-band power: time course over {label} source", fontsize='xx-large', fontweight='bold')

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_timecourse.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)


            # Plot 1b #
            # Group by participant and duration and average across categories
            data_m = df_cond.groupby(['sub', 'Category', 'Duration'])[df_cond.keys()[1:-8]].mean()

            # Get category data
            fig, axs = plt.subplots(4, 3, figsize=(8,8))
            for c in range(len(conditions[0])):
                print("condition:",conditions[0][c])

                d500_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '500ms'")
                d1000_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1000ms'")
                d1500_m = data_m.query(f"Category =='{conditions[0][c]}' and \
                                  Duration == '1500ms'")

                # Make raster plot
                if band == 'alpha':
                    v = [0.6, 1.4]
                elif band == 'gamma':
                    v = [0.9, 1.1]

                for d, data in zip(range(len(conditions[2])), [d500_m, d1000_m, d1500_m]):
                    im = axs[c,d].imshow(
                        data, cmap="RdYlBu_r",
                        vmin=v[0], vmax=v[1],
                        origin="lower", aspect="auto",
                        extent=[times[0], times[-1], len(sub_list), 1])
                    axs[c,d].set_xlim([-.5, 2])
                    axs[c,d].axvline(x=0, color="black", linestyle="--")
                    if c == len(conditions[0])-1:
                        axs[c,d].set_xlabel('Time (s)', fontsize='x-large')
                    else:
                        axs[c,d].axes.xaxis.set_ticklabels([])
                    if d != 0:
                        axs[c,d].axes.yaxis.set_ticklabels([])

                axs[c,0].axvline(x=.5, color="black", linestyle="--")
                axs[c,1].axvline(x=1., color="black", linestyle="--")
                axs[c,2].axvline(x=1.5, color="black", linestyle="--")

            axs[0,0].set_ylabel('Face', fontsize='x-large', fontweight='bold')
            axs[1,0].set_ylabel('Object', fontsize='x-large', fontweight='bold')
            axs[2,0].set_ylabel('Letter', fontsize='x-large', fontweight='bold')
            axs[3,0].set_ylabel('False-font', fontsize='x-large', fontweight='bold')

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im, cax=cbar_ax,
                          format=tick.FormatStrFormatter('%.2f'))

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_raster.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)


            # Plot 2a #
            # Group by duration and average across participants and categories
            data_m = df_cond.groupby(['Duration'])[df_cond.keys()[1:-8]].mean()

            # Get 95% condidence intervals
            data_std = df_cond.groupby(['Duration'])[df_cond.keys()[1:-8]].std()
            data_ci = (1.96 * data_std / np.sqrt(len(sub_list)))

            # Get category data
            d500_m = data_m.query("Duration == '500ms'")
            d1000_m = data_m.query("Duration == '1000ms'")
            d1500_m = data_m.query("Duration == '1500ms'")

            d500_ci = data_ci.query("Duration == '500ms'")
            d1000_ci = data_ci.query("Duration == '1000ms'")
            d1500_ci = data_ci.query("Duration == '1500ms'")

            # Cut edges
            d500_m = np.squeeze(np.array(d500_m.iloc[:,tmin:tmax]))
            d1000_m = np.squeeze(np.array(d1000_m.iloc[:,tmin:tmax]))
            d1500_m = np.squeeze(np.array(d1500_m.iloc[:,tmin:tmax]))

            d500_ci = np.squeeze(np.array(d500_ci.iloc[:,tmin:tmax]))
            d1000_ci = np.squeeze(np.array(d1000_ci.iloc[:,tmin:tmax]))
            d1500_ci = np.squeeze(np.array(d1500_ci.iloc[:,tmin:tmax]))

            # Plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(t, np.vstack([d500_m, d1000_m, d1500_m]).transpose(), linewidth=2.0)

            for m, ci in zip([d500_m, d1000_m, d1500_m],
                              [d500_ci, d1000_ci, d1500_ci]):
                ax.fill_between(t, m-ci, m+ci, alpha=.2)

            ax.set_xlabel('Time (s)', fontsize='x-large')
            ax.axvline(x=0, color="black", linestyle="--")

            ax.set_xlim([-.5, 2.4])
            if band == 'alpha':
                ax.set_ylim([0.6, 1.4])
            elif band == 'gamma':
                ax.set_ylim([0.9, 1.1])
            # ax.axvspan(.3, .5, color='grey', alpha=0.25)
            ax.axvspan(tbins[0][0], tbins[0][1], color='red', alpha=0.25)
            ax.axvspan(tbins[1][0], tbins[1][1], color='red', alpha=0.25)
            ax.axvspan(tbins[2][0], tbins[2][1], color='red', alpha=0.25)
            ax.legend(['500ms', '1000ms', '1500ms'], loc='lower left')

            ax.set_ylabel('Power (a.u.)', fontsize='x-large')
            plt.suptitle(f"{band}-band power: time course over {label} source", fontsize='xx-large', fontweight='bold')

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_timecourse_avg.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)


            # Plot 2b #
            # Group by participant and duration and average across categories
            data_m = df_cond.groupby(['sub', 'Duration'])[df_cond.keys()[1:-8]].mean()

            # Get category data
            d500_m = data_m.query("Duration == '500ms'")
            d1000_m = data_m.query("Duration == '1000ms'")
            d1500_m = data_m.query("Duration == '1500ms'")

            # Make raster plot
            fig, axs = plt.subplots(3, 1, figsize=[8,6])
            if band == 'alpha':
                v = [0.6, 1.4]
            elif band == 'gamma':
                v = [0.9, 1.1]

            for ax, data in zip(axs.flat, [d500_m, d1000_m, d1500_m]):
                im = ax.imshow(
                    data, cmap="RdYlBu_r",
                    vmin=v[0], vmax=v[1],
                    origin="lower", aspect="auto",
                    extent=[times[0], times[-1], len(sub_list), 1])
                ax.set_xlim([-.5, 2])
                ax.axvline(x=0, color="black", linestyle="--")

            axs[0].axvline(x=.5, color="black", linestyle="--")
            axs[1].axvline(x=1., color="black", linestyle="--")
            axs[2].axvline(x=1.5, color="black", linestyle="--")
            axs[2].set_xlabel('Time (s)', fontsize='x-large')
            axs[0].axes.xaxis.set_ticklabels([])
            axs[1].axes.xaxis.set_ticklabels([])
            axs[0].set_ylabel('Participant', fontsize='x-large')
            axs[1].set_ylabel('Participant', fontsize='x-large')
            axs[2].set_ylabel('Participant', fontsize='x-large')

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im, cax=cbar_ax,
                          format=tick.FormatStrFormatter('%.2f'))

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_raster_avg.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)


            # Plot spectral activation raincloud

            # Get indivisual data by condition
            data_sub_m = df_long.groupby(['sub','Category','Duration','time_bin'],as_index = False)["value"].mean()

            # Fix order of levels in duration variable
            durations = ['500ms', '1000ms', '1500ms']
            data_sub_m['Duration'] = pd.Categorical(
                data_sub_m['Duration'],
                categories=durations,
                ordered=True)

            # Loop over categories
            fig, axs = plt.subplots(4, 3, figsize=(8,8))
            for c in range(len(conditions[0])):
                print("condition:",conditions[0][c])

                # Get data
                d_m = data_sub_m.query(f"Category =='{conditions[0][c]}'")

                for d in range(len(durations)):
                    print("duration:",durations[d])

                    # Plot violin
                    pt.half_violinplot(
                          x = "time_bin", y = "value",
                          data = d_m.query(f"Duration =='{durations[d]}'"),
                          bw = .2, cut = 0.,
                          scale = "area", width = .6,
                          inner = None,
                          ax = axs[c,d])

                    # Add points
                    sns.stripplot(
                        x = "time_bin", y = "value",
                        data = d_m.query(f"Duration =='{durations[d]}'"),
                        edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0,
                        ax = axs[c,d])

                    # Add boxplot
                    sns.boxplot(
                        x = "time_bin", y = "value",
                        data = d_m.query(f"Duration =='{durations[d]}'"),
                        color = "black", width = .15, zorder = 10,
                        showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                        showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
                        saturation = 1,
                        ax = axs[c,d])

            for ax in axs.flat:
                if band == 'alpha':
                    ax.set_ylim([0.65, 1.35])
                elif band == 'gamma':
                    ax.set_ylim([0.9, 1.1])

            axs[0,0].set_xlabel(None)
            axs[0,1].set_xlabel(None)
            axs[0,2].set_xlabel(None)
            axs[1,0].set_xlabel(None)
            axs[1,1].set_xlabel(None)
            axs[1,2].set_xlabel(None)
            axs[2,0].set_xlabel(None)
            axs[2,1].set_xlabel(None)
            axs[2,2].set_xlabel(None)

            axs[3,0].set_xlabel(f'{durations[0]} duration', fontsize='x-large', fontweight='bold')
            axs[3,1].set_xlabel(f'{durations[1]} duration', fontsize='x-large', fontweight='bold')
            axs[3,2].set_xlabel(f'{durations[1]} duration', fontsize='x-large', fontweight='bold')

            axs[0,0].set_ylabel('Face', fontsize='x-large', fontweight='bold')
            axs[1,0].set_ylabel('Object', fontsize='x-large', fontweight='bold')
            axs[2,0].set_ylabel('Letter', fontsize='x-large', fontweight='bold')
            axs[3,0].set_ylabel('False-font', fontsize='x-large', fontweight='bold')
            plt.suptitle(f"{band}-band power: {label}", fontsize='xx-large', fontweight='bold')

            plt.tight_layout()

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_timebins.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)

            # Plot 2 #
            # Get indivisual data by condition averaged across categories
            data_sub_m = df_long.groupby(['sub','Duration','time_bin'],as_index = False)["value"].mean()

            # Fix order of levels in duration variable
            durations = ['500ms', '1000ms', '1500ms']
            data_sub_m['Duration'] = pd.Categorical(
                data_sub_m['Duration'],
                categories=['500ms', '1000ms', '1500ms'],
                ordered=True)

            # Create subplot
            fig, axs = plt.subplots(1,3, figsize=(8,6))

            # Loop over durations
            for d in range(len(durations)):
                    print("duration:",durations[d])

                    # Plot violin
                    pt.half_violinplot(
                          x = "time_bin", y = "value",
                          data = d_m.query(f"Duration =='{durations[d]}'"),
                          bw = .2, cut = 0.,
                          scale = "area", width = .6,
                          inner = None,
                          ax = axs[d])

                    # Add points
                    sns.stripplot(
                        x = "time_bin", y = "value",
                        data = d_m.query(f"Duration =='{durations[d]}'"),
                        edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0,
                        ax = axs[d])

                    # Add boxplot
                    sns.boxplot(
                        x = "time_bin", y = "value",
                        data = d_m.query(f"Duration =='{durations[d]}'"),
                        color = "black", width = .15, zorder = 10,
                        showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                        showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
                        saturation = 1,
                        ax = axs[d])

            for ax in axs.flat:
                if band == 'alpha':
                    ax.set_ylim([0.65, 1.35])
                elif band == 'gamma':
                    ax.set_ylim([0.9, 1.1])

            axs[0].set_ylabel('Power (a.u.)', fontsize='x-large')

            axs[0].set_xlabel(f'{durations[0]} duration', fontsize='x-large', fontweight='bold')
            axs[1].set_xlabel(f'{durations[1]} duration', fontsize='x-large', fontweight='bold')
            axs[2].set_xlabel(f'{durations[1]} duration', fontsize='x-large', fontweight='bold')

            plt.suptitle(f"{band}-band power: {label}", fontsize='xx-large', fontweight='bold')

            plt.tight_layout()

            # Save figure
            fname_fig = op.join(source_figure_root,
                                f"sourcedur-{band}_{label}_{tbins[0]}_{task_rel[:3]}_timebins_avg.png")
            fig.savefig(fname_fig, dpi=300)
            plt.close(fig)

    # Save table as .tsv
    bids_path_source = bids_path_source.copy().update(
        root=source_deriv_root,
        subject=f"groupphase{phase}",
        suffix=f"{tbins[0]}_{task_rel[:3]}_lmm_datatable",
        check=False)
    df_all.to_csv(bids_path_source.fpath,
                        sep="\t",
                        index=False)


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


def create_models(package="lmer"):
    if package == "stats_model":
        models = {  "null_model": {
                        "model": "value ~ 1",
                        "re_formula": None
                    },
                    "time_win": {
                        "model": "value ~ time_bin",
                        "re_formula": None
                    },
                    "duration": {
                        "model": "value ~ Duration",
                        "re_formula": None
                    },
                    "time_win_dur": {
                        "model": "value ~ time_bin + Duration",
                        "re_formula": None
                    },
                    "time_win_dur_iit": {
                        "model": "value ~ time_bin + Duration + iit_predictors",
                        "re_formula": None
                    },
                    "time_win_dur_gnw": {
                        "model": "value ~ time_bin + Duration + gnw_predictors",
                        "re_formula": None
                    },
                    "time_win_dur_cate_iit": {
                        "model": "value ~ time_bin + Duration + Category*iit_predictors",
                        "re_formula": None
                    },
                    "time_win_dur_cate_gnw": {
                        "model": "value ~ time_bin + Duration + Category*gnw_predictors",
                        "re_formula": None
                    }}
    elif package == "lmer":
        models = {
            "null_model": {
                "model": "value ~ 1 + (1|sub)",
                "re_formula": None
            },
            "time_win": {
                "model": "value ~ time_bin + (1|sub)",
                "re_formula": None
            },
            "duration": {
                "model": "value ~ Duration + (1|sub)",
                "re_formula": None
            },
            "time_win_dur": {
                "model": "value ~ time_bin + Duration + (1|sub)",
                "re_formula": None
            },
            "time_win_dur_iit": {
                "model": "value ~ time_bin + Duration + iit_predictors + (1|sub)",
                "re_formula": None
            },
            "time_win_dur_gnw": {
                "model": "value ~ time_bin + Duration + gnw_predictors + (1|sub)",
                "re_formula": None
            },
            "time_win_dur_cate_iit": {
                "model": "value ~ time_bin + Duration + Category * iit_predictors + (1|sub)",
                "re_formula": None
            },
            "time_win_dur_cate_gnw": {
                "model": "value ~ time_bin + Duration + Category * gnw_predictors + (1|sub)",
                "re_formula": None
            }}
    return models


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
        print(model)
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
            results = pd.concat([results,
                                 pd.DataFrame({
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
            })], ignore_index=True)
        elif package == "lmer":
            # Fit the model:
            mdl = Lmer(models[model]["model"], data=data)
            print(mdl.fit(REML=False))
            # Append the coefs to the results table:
            coefs = mdl.coefs
            results = pd.concat([results,
                                 pd.DataFrame({
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
            })], ignore_index=True)

            # In addition, run the anova on the model to extract the main effects:
            anova_res = mdl.anova()
            # For the null model, since there are no main effects, the anova results are empty:
            if len(anova_res) == 0:
                anova_results = pd.concat([anova_results, pd.DataFrame({
                    "subject": group.split("-")[0],
                    "analysis_name": "anova",
                    "model": model,
                    "group": group,
                    "conditions": np.nan,
                    "F-stat": np.nan,
                    "NumDF": np.nan,
                    "DenomDF": np.nan,
                    "p-value": np.nan,
                    "reject": np.nan,
                    "converged": [True] * len(coefs["Estimate"]),
                    "SS": np.nan,
                    "aic": mdl.AIC,
                    "bic": mdl.BIC
                }, index=[0])], ignore_index=True)
            else:
                anova_results = pd.concat([anova_results, pd.DataFrame({
                    "subject": group.split("-")[0],
                    "analysis_name": ["anova"] * len(anova_res),
                    "model": [model] * len(anova_res),
                    "group": [group] * len(anova_res),
                    "conditions": anova_res.index.values,
                    "F-stat": anova_res["F-stat"].to_list(),
                    "NumDF": anova_res["NumDF"].to_list(),
                    "DenomDF": anova_res["DenomDF"].to_list(),
                    "p-value": anova_res["P-val"].to_list(),
                    "reject": [True if p_val < alpha else False for p_val in anova_res["P-val"].to_list()],
                    "converged": [True] * len(anova_res),
                    "SS": anova_res["SS"].to_list(),
                    "aic": [mdl.AIC] * len(anova_res),
                    "bic": [mdl.BIC] * len(anova_res)
                })], ignore_index=True)

    return results, anova_results


def model_comparison(models_results, criterion="bic", test="linear_mixed_model"):
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


if __name__ == '__main__':
    for task_rel in ['Irrelevant', 'Relevant non-target']:
        for tbins in [ [[0.8,1.0],[1.3,1.5],[1.8,2.0]], [[1.0,1.2],[1.5,1.7],[2.0,2.2]] ]:
            run_source_dur_activation(task_rel, tbins)
