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
import seaborn as sns
import ptitprince as pt  #conda install -c conda-forge ptitprince

import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root

from S06_source_dur_lmm import (create_theories_predictors, fit_lmm,
                                model_comparison, create_models)

# Set params
visit_id = "V1"

task_rel = 'Irrelevant'
# task_rel = 'Relevant non-target'
tbins = [[0.8,1.0],[1.3,1.5],[1.8,2.0]]

debug = False


factor = ['Category', 'Task_relevance', "Duration"]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant target','Relevant non-target','Irrelevant'],
              ['500ms', '1000ms', '1500ms']]


# Set participant list
phase = 3

if debug:
    sub_list = ["CA124", "CA124"]
else:
    # Read the .txt file
    f = open(op.join(bids_root,
                  f'participants_MEG_phase{phase}_included.txt'), 'r').read()
    # Split text into list of elemetnts
    sub_list = f.split("\n")


def run_source_dur_activation():
    # Set directory paths
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur_ERF")
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
                        root=source_deriv_root[:-10],
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
    for band in ['ERF']:
        print('\nfreq_band:', band)

        for label in labels_names:
            print('\nlabel:', label)

            # Select band and label
            df_cond = df.query(f"band == '{band}' and label == '{label}' and Task_relevance == '{task_rel}'")

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
            # pd.crosstab(index=data_df["iit_predictors"],
            #             columns=[data_df["Duration"], data_df["time_bin"]],
            #             normalize='columns')
            # pd.crosstab(index=data_df["gnw_predictors"],
            #             columns=[data_df["Duration"], data_df["time_bin"]])

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


            # Plot ERF time courses

            # Plot 1a #
            # Group by category and duration and average across participants
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
                # ax.set_xlim([-.5, 2.4])
                # if band == 'alpha':
                #     ax.set_ylim([0.6, 1.4])
                # elif band == 'gamma':
                #     ax.set_ylim([0.9, 1.1])
                # ax.axvspan(.3, .5, color='grey', alpha=0.25)
                ax.axvspan(tbins[0][0], tbins[0][1], color='red', alpha=0.25)
                ax.axvspan(tbins[1][0], tbins[1][1], color='red', alpha=0.25)
                ax.axvspan(tbins[2][0], tbins[2][1], color='red', alpha=0.25)
                ax.legend(['500ms', '1000ms', '1500ms'], loc='lower left')

            axs[0].set_ylabel('Face', fontsize='x-large', fontweight='bold')
            axs[1].set_ylabel('Object', fontsize='x-large', fontweight='bold')
            axs[2].set_ylabel('Letter', fontsize='x-large', fontweight='bold')
            axs[3].set_ylabel('False-font', fontsize='x-large', fontweight='bold')
            plt.suptitle(f"{band}: time course over {label} source", fontsize='xx-large', fontweight='bold')

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
                # if band == 'alpha':
                #     v = [0.6, 1.4]
                # elif band == 'gamma':
                #     v = [0.9, 1.1]

                for d, data in zip(range(len(conditions[2])), [d500_m, d1000_m, d1500_m]):
                    im = axs[c,d].imshow(
                        data, cmap="RdYlBu_r",
                        # vmin=v[0], vmax=v[1],
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
            # if band == 'alpha':
            #     ax.set_ylim([0.6, 1.4])
            # elif band == 'gamma':
            #     ax.set_ylim([0.9, 1.1])
            # ax.axvspan(.3, .5, color='grey', alpha=0.25)
            ax.axvspan(tbins[0][0], tbins[0][1], color='red', alpha=0.25)
            ax.axvspan(tbins[1][0], tbins[1][1], color='red', alpha=0.25)
            ax.axvspan(tbins[2][0], tbins[2][1], color='red', alpha=0.25)
            ax.legend(['500ms', '1000ms', '1500ms'], loc='lower left')

            ax.set_ylabel('Activation (rms)', fontsize='x-large')
            plt.suptitle(f"{band}: time course over {label} source", fontsize='xx-large', fontweight='bold')

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
            # if band == 'alpha':
            #     v = [0.6, 1.4]
            # elif band == 'gamma':
            #     v = [0.9, 1.1]

            for ax, data in zip(axs.flat, [d500_m, d1000_m, d1500_m]):
                im = ax.imshow(
                    data, cmap="RdYlBu_r",
                    # vmin=v[0], vmax=v[1],
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


            # Plot ERF activation raincloud

            # Get indivisual data by condition
            data_sub_m = df_long.groupby(['sub','Category','Duration','time_bin'],as_index = False)["value"].mean()

            # Fix order of levels in duration variable
            data_sub_m['Duration'] = pd.Categorical(
                data_sub_m['Duration'],
                categories=['500ms', '1000ms', '1500ms'],
                ordered=True)

            # Loop over categories
            fig, axs = plt.subplots(4, 3, figsize=(8,8))
            for c in range(len(conditions[0])):
                print("condition:",conditions[0][c])

                # Get data
                d_m = data_sub_m.query(f"Category =='{conditions[0][c]}'")

                for d in range(len(tbins)):
                    print("time bin:",tbins[d])

                    # Plot violin
                    pt.half_violinplot(
                         x = "Duration", y = "value",
                         data = d_m.query(f"time_bin =='{tbins[d]}'"),
                         bw = .2, cut = 0.,
                         scale = "area", width = .6,
                         inner = None,
                         ax = axs[c,d])

                    # Add points
                    sns.stripplot(
                        x = "Duration", y = "value",
                        data = d_m.query(f"time_bin =='{tbins[d]}'"),
                        edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0,
                        ax = axs[c,d])

                    # Add boxplot
                    sns.boxplot(
                        x = "Duration", y = "value",
                        data = d_m.query(f"time_bin =='{tbins[d]}'"),
                        color = "black", width = .15, zorder = 10,
                        showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                        showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
                        saturation = 1,
                        ax = axs[c,d])

            # for ax in axs.flat:
            #     if band == 'alpha':
            #         ax.set_ylim([0.65, 1.35])
            #     elif band == 'gamma':
            #         ax.set_ylim([0.9, 1.1])

            axs[0,0].set_xlabel(None)
            axs[0,1].set_xlabel(None)
            axs[0,2].set_xlabel(None)
            axs[1,0].set_xlabel(None)
            axs[1,1].set_xlabel(None)
            axs[1,2].set_xlabel(None)
            axs[2,0].set_xlabel(None)
            axs[2,1].set_xlabel(None)
            axs[2,2].set_xlabel(None)

            axs[3,0].set_xlabel(f'{tbins[0]} time bin', fontsize='x-large', fontweight='bold')
            axs[3,1].set_xlabel(f'{tbins[1]} time bin', fontsize='x-large', fontweight='bold')
            axs[3,2].set_xlabel(f'{tbins[1]} time bin', fontsize='x-large', fontweight='bold')

            axs[0,0].set_ylabel('Face', fontsize='x-large', fontweight='bold')
            axs[1,0].set_ylabel('Object', fontsize='x-large', fontweight='bold')
            axs[2,0].set_ylabel('Letter', fontsize='x-large', fontweight='bold')
            axs[3,0].set_ylabel('False-font', fontsize='x-large', fontweight='bold')
            plt.suptitle(f"{band}: time bins over {label} source", fontsize='xx-large', fontweight='bold')

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
            data_sub_m['Duration'] = pd.Categorical(
                data_sub_m['Duration'],
                categories=['500ms', '1000ms', '1500ms'],
                ordered=True)

            # Create subplot
            fig, axs = plt.subplots(1,3, figsize=(8,6))

            # Loop over durations
            for d in range(len(tbins)):
                    print("time bin:",tbins[d])

                    # Plot violin
                    pt.half_violinplot(
                         x = "Duration", y = "value",
                         data = data_sub_m.query(f"time_bin =='{tbins[d]}'"),
                         bw = .2, cut = 0.,
                         scale = "area", width = .6,
                         inner = None,
                         ax = axs[d])

                    # Add points
                    sns.stripplot(
                        x = "Duration", y = "value",
                        data = data_sub_m.query(f"time_bin =='{tbins[d]}'"),
                        edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0,
                        ax = axs[d])

                    # Add boxplot
                    sns.boxplot(
                        x = "Duration", y = "value",
                        data = data_sub_m.query(f"time_bin =='{tbins[d]}'"),
                        color = "black", width = .15, zorder = 10,
                        showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                        showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
                        saturation = 1,
                        ax = axs[d])

            # for ax in axs.flat:
            #     if band == 'alpha':
            #         ax.set_ylim([0.65, 1.35])
            #     elif band == 'gamma':
            #         ax.set_ylim([0.9, 1.1])

            axs[0].set_ylabel('Activaiton (rms)', fontsize='x-large')

            axs[0].set_xlabel('0.8-1.0 time bin', fontsize='x-large')
            axs[1].set_xlabel('1.3-1.5 time bin', fontsize='x-large')
            axs[2].set_xlabel('1.8-2.0 time bin', fontsize='x-large')

            plt.suptitle(f"{band}: time bins over {label} source", fontsize='xx-large', fontweight='bold')

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


if __name__ == '__main__':
    run_source_dur_activation()
