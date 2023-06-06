# Proposed structure for the repository

**coglib** -> all analysis code including plotting_uniformization and any specific or common code that is used for analysis, each analysis pipeline should have a requirements list of dependencies.
> Eventually coglib is to be separated into it's own repo and library

**figures** -> The specific usage of the code in *coglib*. For each figure have a folder and a data_loading, data_processing, and plotting scripts, additional run.py script to bring everything together.

**sample_data** -> Sample data files.

coglib/
    ieeg/
    fmri/
    meeg/
    beh_et/
    utils.py

figures/
    figure1/
        load_data.py
        plot_data.py
        process_data.py
        run.py

sample_data/
