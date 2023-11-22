import os
from pathlib import Path
from rsa.rsa_batch_runner import rsa_batch_runner
from Experiment1ActivationAnalysis.activation_analysis_batch_runner import activation_analysis_batch_runner
from category_selectivity_analysis.category_selectivity_batch_runner import category_selectivity_batch_runner
from visual_responsiveness_analysis.visual_responsiveness_batch_runner import visual_responsiveness_batch_runner

subjects_list = ["CE103", "CE107", "CE108", "CE109", "CE110", "CE112", "CE113", "CE115", "CE118",
                 "CE119", "CF102", "CF103", "CF104", "CF105", "CF106", "CF107", "CF109", "CF110",
                 "CF112", "CF113", "CF117", "CF119"]


def run_analysis_pipeline(bids_root, do_visual_responsiveness=False,
                          do_category_selectivity=False, do_activation_analysis=False, do_rsa=False):

    # Get the original directory:
    orig_dir = os.getcwd()
    # Launching all analysis:
    # Visual responsiveness
    if do_visual_responsiveness:
        # Create the results directory if it doesn't exists:
        if not os.path.isdir(Path(bids_root, "derivatives", "visual_responsiveness")):
            os.makedirs(Path(bids_root, "derivatives", "visual_responsiveness"))
        # Go to the visual responsiveness dir:
        os.chdir(Path(orig_dir, "visual_responsiveness_analysis"))
        # Launch the pipeline in a batch:
        visual_responsiveness_batch_runner()

    # Category selectivity:
    if do_category_selectivity:
        # Create the results dir if it doesn't exists already:
        if not os.path.isdir(Path(bids_root, "derivatives", "category_selectivity")):
            os.makedirs(Path(bids_root, "derivatives", "category_selectivity"))
        # Change dir to scripts dir:
        os.chdir(Path(orig_dir, "category_selectivity_analysis"))
        # Launch the pipeline in a batch:
        category_selectivity_batch_runner()

    # Activation analysis:
    if do_activation_analysis:
        # Create the results dir if it doesn't exists already:
        if not os.path.isdir(Path(bids_root, "derivatives", "activation_analysis")):
            os.makedirs(Path(bids_root, "derivatives", "activation_analysis"))
        # Change dir to scripts dir:
        os.chdir(Path(orig_dir, "Experiment1ActivationAnalysis"))
        # Launch the pipeline in a batch:
        activation_analysis_batch_runner(lmm=True, onset_offset=True, duration_tracking=True)

    # RSA:
    if do_rsa:
        # Create the results dir if it doesn't exists already:
        if not os.path.isdir(Path(bids_root, "derivatives", "rsa")):
            os.makedirs(Path(bids_root, "derivatives", "rsa"))
        # Change dir to scripts dir:
        os.chdir(Path(orig_dir, "rsa"))
        # Launch the pipeline in a batch:
        rsa_batch_runner()


if __name__ == "__main__":
    run_analysis_pipeline("bids_root",
                          do_visual_responsiveness=False,
                          do_category_selectivity=False,
                          do_activation_analysis=False,
                          do_rsa=False)
