"""
===========================
03. Artifact annotation
===========================

Detect and note ocular and muscle artifacts

@author: Oscar Ferrante oscfer88@gmail.com

"""  # noqa: E501

import os.path as op
import os
import matplotlib.pyplot as plt
import shutil

from fpdf import FPDF
import mne
from mne.preprocessing import annotate_muscle_zscore
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

from config.config import bids_root

def artifact_annotation(subject_id, visit_id, record="run", has_eeg=False, threshold_muscle=7):

    # Prepare PDF report
    pdf = FPDF(orientation="P", unit="mm", format="A4")

    # Set path to preprocessing derivatives
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    prep_figure_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    prep_report_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "reports")
    prep_code_root = op.join(prep_deriv_root,
                             f"sub-{subject_id}",f"ses-{visit_id}","meg",
                             "codes")

    # Find whether the recording has EEG data and set the suffix
    if has_eeg:
        suffix = "car"
    else:
        suffix = 'sss'

    print("Processing subject: %s" % subject_id)

    # Loop over runs
    data_path = os.path.join(bids_root,f"sub-{subject_id}",f"ses-{visit_id}","meg")

    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".json") and record in fname:

            # Set run
            if "run" in fname:
                run = f"{int(fname[-10]):02}"
            elif "rest" in fname:
                run = None
            print("  Run: %s" % run)

            # Set task
            if 'dur' in fname:
                bids_task = 'dur'
            elif 'vg' in fname:
                bids_task = 'vg'
            elif 'replay' in fname:
                bids_task = 'replay'
            elif "rest" in fname:
                bids_task = "rest"
            else:
                raise ValueError("Error: could not find the task for %s" % fname)

            # Set BIDS path
            bids_path_sss = mne_bids.BIDSPath(
                root=prep_deriv_root,
                subject=subject_id,
                datatype='meg',
                task=bids_task,
                run=run,
                session=visit_id,
                suffix=suffix,
                extension='.fif',
                check=False)

            # Read raw data
            raw = mne_bids.read_raw_bids(bids_path_sss).load_data()

            ###########################
            # Detect ocular artifacts #
            ###########################

            if has_eeg:
                # Resetting the EOG channel
                eog_ch = raw.copy().pick_types(meg=False, eeg=False, eog=True)
                if len(eog_ch.ch_names) < 2:
                    raw.set_channel_types({'BIO002':'eog'})
                    raw.rename_channels({'BIO002': 'EOG002'})

                # Find EOG events
                eog_events = mne.preprocessing.find_eog_events(raw)
                onsets = (eog_events[:, 0] - raw.first_samp) / raw.info['sfreq'] - 0.25
                durations = [0.5] * len(eog_events)
                descriptions = ['Blink'] * len(eog_events)

                # Annotate events
                annot_blink = mne.Annotations(
                    onsets,
                    durations,
                    descriptions)

                # Plot blink with EEG data
                eeg_picks = mne.pick_types(raw.info,
                                          meg=False,
                                          eeg=True,
                                          eog=True)
                fig = raw.plot(events=eog_events,
                              start=100,
                              order=eeg_picks)
                fname_fig = op.join(prep_figure_root,
                                   "03_%sr%s_artifact_blink.png" % (bids_task,run))
                fig.savefig(fname_fig)
                plt.close()

            ###########################
            # Detect muscle artifacts #
            ###########################

            # Notch filter
            raw_muscle = raw.copy().notch_filter([50, 100])

            # Choose one channel type, if there are axial gradiometers and magnetometers,
            # select magnetometers as they are more sensitive to muscle activity.
            annot_muscle, scores_muscle = annotate_muscle_zscore(
                raw_muscle,
                ch_type="mag",
                threshold=threshold_muscle,
                min_length_good=0.3,
                filter_freq=[110, 140])

            # Plot muscle z-scores across recording
            fig1, ax = plt.subplots()
            ax.plot(raw.times, scores_muscle)
            ax.axhline(y=threshold_muscle, color='r')
            ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity (threshold = %s)' % threshold_muscle)
            fname_fig1 = op.join(prep_figure_root,
                                "03_%sr%s_artifact_muscle.png" % (bids_task,run))
            fig1.savefig(fname_fig1)
            plt.close()

            # Add figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:-8])
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Muscle artifact power', 'B', ln=1)
            pdf.image(fname_fig1, 0, 45, pdf.epw*.8)

            #################
            # Detect breaks #
            #################

            if record == "run":
                # Get events
                events, event_id = mne.events_from_annotations(raw)

                # Detect breaks based on events
                annot_break = mne.preprocessing.annotate_break(
                    raw=raw,
                    events=events,
                    min_break_duration=15.0)

            ###########################

            # Contatenate blink and muscle artifact annotations
            if has_eeg:
                annot_artifact = annot_blink + annot_muscle
            else:
                annot_artifact = annot_muscle
            annot_artifact = mne.Annotations(onset = annot_artifact.onset + raw._first_time,
                                              duration = annot_artifact.duration,
                                              description = annot_artifact.description,
                                              orig_time = raw.info['meas_date'])

            # Add artifact annotations in raw
            if record == "run":
                raw.set_annotations(raw.annotations + annot_artifact + annot_break)
            elif record == "rest":
                raw.set_annotations(raw.annotations + annot_artifact)

            # View raw with annotations
            channel_picks = mne.pick_types(raw.info,
                                           meg='mag', eog=True)
            fig2 = raw.plot(duration=50,
                           start=100,
                           order=channel_picks)
            fname_fig2 = op.join(prep_figure_root,
                                "03_%sr%s_artifact_annot.png" % (bids_task,run))
            fig2.savefig(fname_fig2)
            plt.close()

            # Add figures to report
            pdf.ln(120)
            pdf.cell(0, 10, 'Data and annotations', 'B', ln=1)
            pdf.image(fname_fig2, 0, 175, pdf.epw)

            # Save data with annotated artifacts
            bids_path_annot = bids_path_sss.copy().update(
                suffix="annot",
                check=False)

            raw.save(bids_path_annot, overwrite=True)

    # Save code
    shutil.copy(__file__, prep_code_root)

    # Save report
    if record == "rest":
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report_rest.pdf'))
    else:
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report.pdf'))

def input_bool(message):
    value = input(message)
    if value == "True":
        return True
    if value == "False":
        return False

if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., CA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    has_eeg = input_bool("Has this recording EEG data? (True or False)\n>>> ")
    threshold_muscle = int(input("Set the threshold for muscle artifact? (default is 7)\n>>> "))
    artifact_annotation(subject_id, visit_id, has_eeg=has_eeg, threshold_muscle=threshold_muscle)
